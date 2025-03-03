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

def separate_audio(audio, sr):
    # Convert audio to tensor
    audio_tensor = torch.tensor(audio).unsqueeze(0)
    # Apply Demucs model
    sources = apply_model(model, audio_tensor, device='cpu', shifts=1)
    return sources[0, 0].numpy()  # Return only the 'vocals' stem

# Streamlit UI
st.title("🎵 AI-Powered Noise Reduction with Demucs")

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
    
    # Apply Demucs for Noise Reduction
    filtered_audio = separate_audio(y, sr)
    
    # Show Processed Waveform
    fig, ax = plt.subplots()
    librosa.display.waveshow(filtered_audio, sr=sr)
    ax.set_title("Filtered Audio Waveform")
    st.pyplot(fig)
    
    # Save Processed Audio
    output_filename = "filtered_audio.wav"
    sf.write(output_filename, filtered_audio, sr)
    
    st.success("Processing Complete! 🎉")
    st.download_button(label="Download Processed Audio", data=open(output_filename, "rb"), file_name="filtered_audio.wav", mime="audio/wav")
