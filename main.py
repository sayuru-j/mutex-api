import os
import io
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import soundfile as sf
import json
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
from audoai.noise_removal import NoiseRemovalClient  # Added Audo AI import

# Create FastAPI app
app = FastAPI(title="MuteX API", 
              description="API for denoising audio files",
              version="1.0.0")

# Define the path to the config.json file
config_file_path = 'config.json'

# Load the configuration parameters from the config.json file
with open(config_file_path, 'r') as f:
    config = json.load(f)

# Create output directory
output_dir = "Denoised Output"
os.makedirs(output_dir, exist_ok=True)

# Initialize Audo AI client with API key
audo_api_key = '20a5cdca54029e516b0930002057b744'
noise_removal_client = NoiseRemovalClient(api_key=audo_api_key)

def normalize_spectrogram(spectrogram):
    """Normalize a spectrogram to the range [0, 1]."""
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)
    if max_val > min_val:
        spectrogram = (spectrogram - min_val) / (max_val - min_val)
    return spectrogram, min_val, max_val

def denormalize_spectrogram(spectrogram, original_min, original_max):
    """Denormalize a spectrogram back to its original range using the original minimum and maximum values."""
    spectrogram = spectrogram * (original_max - original_min) + original_min
    return spectrogram

def spectrogram_to_audio(spectrogram, phase):
    """Convert a spectrogram back to audio signal."""
    # Combine magnitude and phase
    complex_spec = spectrogram * np.exp(1j * phase)
    
    # Invert the STFT
    y = librosa.istft(complex_spec, hop_length=config['hop_length'], win_length=config['window_size'])
    
    return y

def denoise_audio_with_audo_ai(audio_data, sr):
    """Denoise audio using Audo AI API."""
    # Save the audio data to a temporary file
    temp_input_path = os.path.join(output_dir, "temp_input.wav")
    sf.write(temp_input_path, audio_data, sr, format='wav')
    
    # Process with Audo AI
    temp_output_path = os.path.join(output_dir, "temp_output.wav")
    result = noise_removal_client.process(temp_input_path)
    result.save(temp_output_path)
    
    # Read the denoised audio
    denoised_audio, sr = librosa.load(temp_output_path, sr=sr)
    
    # Compute spectrograms for visualization
    D_original = librosa.stft(audio_data, n_fft=config['window_size'], hop_length=config['hop_length'])
    D_original_db = librosa.amplitude_to_db(np.abs(D_original))
    
    D_denoised = librosa.stft(denoised_audio, n_fft=config['window_size'], hop_length=config['hop_length'])
    D_denoised_db = librosa.amplitude_to_db(np.abs(D_denoised))
    
    # Create visualization of spectrograms
    plt.figure(figsize=(15, 10))
    
    # Plot original spectrogram
    plt.subplot(2, 1, 1)
    librosa.display.specshow(D_original_db, sr=sr, hop_length=config['hop_length'], x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original Spectrogram')
    
    # Plot denoised spectrogram
    plt.subplot(2, 1, 2)
    librosa.display.specshow(D_denoised_db, sr=sr, hop_length=config['hop_length'], x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Denoised Spectrogram (Audo AI)')
    
    plt.tight_layout()
    
    # Save visualization to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Save audio to bytes buffer
    audio_buf = io.BytesIO()
    sf.write(audio_buf, denoised_audio, sr, format='wav')
    audio_buf.seek(0)
    
    # Clean up temporary files
    os.remove(temp_input_path)
    os.remove(temp_output_path)
    
    return {
        'denoised_audio': audio_buf,
        'spectrogram_image': buf,
        'sample_rate': sr
    }

def denoise_audio(audio_data, sr=None, use_bypass=False):
    """Denoise audio data and return the denoised audio and spectrograms."""
    # If bypass flag is set, use Audo AI API instead of local model
    if use_bypass:
        return denoise_audio_with_audo_ai(audio_data, sr or config['sample_rate'])
    
    # Otherwise, use the original local model approach
    # Load the model
    model_path = 'modelv1.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    model = load_model(model_path, compile=False)
    
    # Process audio
    sr = sr or config['sample_rate']
    if sr != config['sample_rate']:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=config['sample_rate'])
        sr = config['sample_rate']
    
    # Compute spectrogram
    D = librosa.stft(audio_data, n_fft=config['window_size'], hop_length=config['hop_length'])
    D_db = librosa.amplitude_to_db(np.abs(D))
    D_db_norm, min_val, max_val = normalize_spectrogram(D_db)
    
    # Get phase information (needed for reconstruction)
    phase = np.angle(D)
    
    # Prepare for model input
    D_db_norm_resized = tf.image.resize(D_db_norm[..., np.newaxis], 
                                       [config['input_shape'][0], config['input_shape'][1]]).numpy()
    
    # Predict denoised spectrogram
    denoised_spec_resized = model.predict(D_db_norm_resized[np.newaxis, ...])[0]
    
    # Resize back to original dimensions
    denoised_spec = tf.image.resize(denoised_spec_resized, [D_db.shape[0], D_db.shape[1]]).numpy()[..., 0]
    
    # Denormalize
    denoised_spec_db = denormalize_spectrogram(denoised_spec, min_val, max_val)
    
    # Convert back to magnitude
    denoised_mag = librosa.db_to_amplitude(denoised_spec_db)
    
    # Reconstruct audio with original phase
    denoised_audio = spectrogram_to_audio(denoised_mag, phase)
    
    # Create visualization of spectrograms
    plt.figure(figsize=(15, 10))
    
    # Plot original spectrogram
    plt.subplot(2, 1, 1)
    librosa.display.specshow(D_db, sr=sr, hop_length=config['hop_length'], x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original Spectrogram')
    
    # Plot denoised spectrogram
    plt.subplot(2, 1, 2)
    librosa.display.specshow(denoised_spec_db, sr=sr, hop_length=config['hop_length'], x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Denoised Spectrogram')
    
    plt.tight_layout()
    
    # Save visualization to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Save audio to bytes buffer
    audio_buf = io.BytesIO()
    sf.write(audio_buf, denoised_audio, sr, format='wav')
    audio_buf.seek(0)
    
    return {
        'denoised_audio': audio_buf,
        'spectrogram_image': buf,
        'sample_rate': sr
    }

@app.get("/")
async def root():
    return {"message": "MuteX API is running. Use /denoise endpoint to denoise audio."}

@app.post("/denoise/")
async def denoise_route(
    file: UploadFile = File(...),
    bp: bool = Query(False, description="Set to true to use Audo AI API instead of local model")
):
    """
    Denoise an uploaded audio file and return both the denoised audio and a visualization of the spectrograms.
    
    - Set 'bp' parameter to true to use the Audo AI API instead of the local model.
    
    Returns a JSON with URLs to download the denoised audio and view the spectrogram visualization.
    """
    if not file.filename.lower().endswith(('.wav', '.mp3')):
        raise HTTPException(status_code=400, detail="Only WAV and MP3 files are supported")
    
    try:
        # Read file content
        contents = await file.read()
        
        # Create a temporary file to save the uploaded audio
        temp_file = os.path.join(output_dir, f"temp_{file.filename}")
        with open(temp_file, "wb") as f:
            f.write(contents)
        
        # Load audio file
        audio_data, sr = librosa.load(temp_file, sr=None)
        
        # Remove temporary file
        os.remove(temp_file)
        
        # Process audio using specified method
        result = denoise_audio(audio_data, sr, use_bypass=bp)
        
        # Generate unique filenames
        base_filename = os.path.splitext(file.filename)[0]
        method_prefix = "audo_" if bp else "denoised_"
        denoised_filename = f"{method_prefix}{base_filename}.wav"
        spectrogram_filename = f"spectrogram_{method_prefix}{base_filename}.png"
        
        # Save files for potential future reference
        denoised_path = os.path.join(output_dir, denoised_filename)
        spectrogram_path = os.path.join(output_dir, spectrogram_filename)
        
        with open(denoised_path, "wb") as f:
            f.write(result['denoised_audio'].getvalue())
        
        with open(spectrogram_path, "wb") as f:
            f.write(result['spectrogram_image'].getvalue())
        
        # Prepare response with both files
        result['denoised_audio'].seek(0)
        result['spectrogram_image'].seek(0)
        
        # Return immediate response with denoised audio
        return StreamingResponse(
            result['denoised_audio'], 
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename={denoised_filename}"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.get("/denoise/{filename}/audio")
async def get_denoised_audio(filename: str):
    """Get a specific denoised audio file by filename."""
    # Check for both possible prefixes
    prefixes = ["denoised_", "audo_"]
    file_path = None
    
    for prefix in prefixes:
        path = os.path.join(output_dir, f"{prefix}{filename}.wav")
        if os.path.exists(path):
            file_path = path
            break
    
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(file_path, media_type="audio/wav")

@app.get("/denoise/{filename}/spectrogram")
async def get_spectrogram(filename: str):
    """Get a specific spectrogram image by filename."""
    # Check for different possible filename patterns
    prefixes = ["denoised_", "audo_"]
    file_path = None
    
    for prefix in prefixes:
        path = os.path.join(output_dir, f"spectrogram_{prefix}{filename}.png")
        if os.path.exists(path):
            file_path = path
            break
            
    # If not found, check traditional pattern
    if not file_path:
        path = os.path.join(output_dir, f"spectrogram_{filename}.png")
        if os.path.exists(path):
            file_path = path
    
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(file_path, media_type="image/png")

@app.get("/processed")
async def list_processed_files():
    """List all processed files."""
    files = []
    for f in os.listdir(output_dir):
        if f.startswith("denoised_") or f.startswith("audo_"):
            if f.endswith(".wav"):
                prefix = "denoised_" if f.startswith("denoised_") else "audo_"
                base_name = f.replace(prefix, "")
                file_name = os.path.splitext(base_name)[0]
                method = "Audo AI" if prefix == "audo_" else "Local Model"
                files.append({
                    "filename": file_name,
                    "method": method,
                    "audio_url": f"/denoise/{file_name}/audio",
                    "spectrogram_url": f"/denoise/{file_name}/spectrogram"
                })
    return {"processed_files": files}

@app.delete("/clear-output")
async def clear_output_directory():
    """Delete all files in the Denoised Output directory."""
    try:
        count = 0
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                count += 1
        return {
            "status": "success", 
            "message": f"Successfully deleted {count} files from the output directory.",
            "deleted_count": count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing output directory: {str(e)}")
    
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)