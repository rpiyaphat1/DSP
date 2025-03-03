import streamlit as st
import torch
import torchaudio
from demucs import pretrained
from demucs.apply import apply_model
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import tempfile

# Load Demucs model
model = pretrained.get_model(name="htdemucs")
model.eval()

def demucs_denoise(audio_path, sr):
    wav, sr = torchaudio.load(audio_path)
    wav = wav.mean(dim=0, keepdim=True)  # Convert to mono
    with torch.no_grad():
        sources = apply_model(model, wav, device='cpu', split=True)
    
    # Extract the 'noisy' and 'clean' components (assuming vocals as main sound)
    clean_audio = sources[0, 0]  # Main separated source (can be customized)
    return clean_audio.numpy(), sr

# Streamlit UI
st.title("🎵 AI-Powered Noise Reduction with Demucs")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(uploaded_file.read())
        tmpfile_path = tmpfile.name
    
    # Load audio
    y, sr = librosa.load(tmpfile_path, sr=None)
    
    # Show original waveform
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr)
    ax.set_title("Original Audio Waveform")
    st.pyplot(fig)
    
    # Apply Demucs for noise reduction
    clean_audio, sr = demucs_denoise(tmpfile_path, sr)
    
    # Show processed waveform
    fig, ax = plt.subplots()
    librosa.display.waveshow(clean_audio, sr=sr)
    ax.set_title("Denoised Audio Waveform")
    st.pyplot(fig)
    
    # Save processed audio
    output_filename = "denoised_audio.wav"
    sf.write(output_filename, clean_audio, sr)
    
    st.success("Processing Complete! 🎉")
    st.download_button(label="Download Processed Audio", data=open(output_filename, "rb"), file_name="denoised_audio.wav", mime="audio/wav")
