import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import butter, lfilter

# DSP Functions
def butter_lowpass_filter(data, cutoff=1500, fs=44100, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# Streamlit UI
st.title("🎵 DSP-Based Noise Reduction App")

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
    
    # Apply Noise Reduction
    filtered_audio = butter_lowpass_filter(y)
    
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
