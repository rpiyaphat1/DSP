import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import os

# Install dependencies dynamically
os.system("pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu")
os.system("pip install demucs")

from demucs import pretrained
from demucs.apply import apply_model

# Load Demucs model
st.write("Loading Demucs model...")
try:
    model = pretrained.get_model('htdemucs')
    model.cpu().eval()
    st.success("Demucs model loaded successfully!")
except Exception as e:
    st.error(f"Error loading Demucs model: {e}")

# Function to remove noise using Demucs
def remove_noise(audio, sr):
    try:
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        sources = apply_model(model, audio_tensor, device='cpu', shifts=1)
        return sources[0, 0].numpy()  # Return denoised audio
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return audio

# Streamlit UI
st.title("🎵 AI-Powered Noise Removal with Demucs")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    # Load Audio
    try:
        y, sr = librosa.load(uploaded_file, sr=None)
        
        # Show Original Waveform
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr)
        ax.set_title("Original Audio Waveform")
        st.pyplot(fig)
        
        # Apply Demucs for Noise Removal
        denoised_audio = remove_noise(y, sr)
        
        # Show Processed Waveform
        fig, ax = plt.subplots()
        librosa.display.waveshow(denoised_audio, sr=sr)
        ax.set_title("Denoised Audio Waveform")
        st.pyplot(fig)
        
        # Save Processed Audio
        output_filename = "denoised_audio.wav"
        sf.write(output_filename, denoised_audio, sr)
        
        st.success("Noise Removal Complete! 🎉")
        st.download_button(label="Download Denoised Audio", data=open(output_filename, "rb"), file_name="denoised_audio.wav", mime="audio/wav")
    except Exception as e:
        st.error(f"Error loading or processing audio: {e}")
