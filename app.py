import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import torch
from demucs import pretrained
from demucs.apply import apply_model

# Load Demucs model
model = pretrained.get_model('htdemucs')
model.cpu().eval()

def remove_noise(audio, sr):
    # Convert audio to tensor
    audio_tensor = torch.tensor(audio).unsqueeze(0)
    # Apply Demucs model for noise removal
    sources = apply_model(model, audio_tensor, device='cpu', shifts=1)
    return sources[0, 0].numpy()  # Return only the 'vocals' stem (denoised audio)

# Streamlit UI
st.title("🎵 AI-Powered Noise Removal with Demucs")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    # Load Audio
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
