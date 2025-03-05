import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import torchaudio
from scipy.ndimage import median_filter
from scipy.signal import spectrogram

# โหลดโมเดล AI DeepFilterNet
@st.cache_resource
def load_model():
    model = torch.hub.load("jansson/demucs", "demucs")  # หรือใช้ DeepFilterNet
    model.eval()
    return model

model = load_model()

# ฟังก์ชัน Spectral Gating (ลบ noise เพิ่มเติม)
def spectral_gate(y, sr, threshold=0.1):
    _, _, Sxx = spectrogram(y, fs=sr)
    mask = Sxx > (threshold * np.max(Sxx))  # สร้าง mask ให้ลบ noise ต่ำกว่า threshold
    y_clean = y * mask[:len(y)]
    return y_clean

# ฟังก์ชัน AI Noise Reduction (DeepFilterNet)
def denoise_with_ai(audio, sr):
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        denoised_audio = model(audio_tensor)
    return denoised_audio.squeeze(0).numpy()

# ฟังก์ชัน Median Filter (ช่วยลด noise ที่ยังเหลือ)
def smooth_audio(y):
    return median_filter(y, size=3)

# Streamlit UI
st.title("🎵 Advanced AI + DSP Noise Reduction")
st.write("🔍 อัปโหลดไฟล์เสียง แล้วลบ noise แบบสมจริง!")

uploaded_file = st.file_uploader("📤 อัปโหลดไฟล์เสียง (.wav)", type=["wav"])

if uploaded_file:
    y, sr = librosa.load(uploaded_file, sr=None)

    # แสดงคลื่นเสียงต้นฉบับ
    st.write("🎧 คลื่นเสียงต้นฉบับ")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    st.pyplot(fig)

    # ใช้ AI ลบ noise
    y_denoised = denoise_with_ai(y, sr)

    # ใช้ Spectral Gating เพื่อลบ noise เพิ่มเติม
    y_filtered = spectral_gate(y_denoised, sr)

    # Smooth เสียงที่ลบ noise ไปแล้ว
    y_final = smooth_audio(y_filtered)

    # แสดงคลื่นเสียงที่ลบ noise แล้ว
    st.write("🔇 คลื่นเสียงหลังจากลบ Noise")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y_final, sr=sr, ax=ax)
    st.pyplot(fig)

    # ดาวน์โหลดไฟล์เสียงที่ผ่านการลบ Noise
    output_file = "denoised_output.wav"
    sf.write(output_file, y_final, sr)
    st.download_button("📥 ดาวน์โหลดไฟล์เสียงที่ปรับปรุงแล้ว", data=open(output_file, "rb"), file_name="denoised_output.wav")
