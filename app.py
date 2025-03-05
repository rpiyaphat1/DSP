import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import wiener
import torch
import torchaudio

# ฟังก์ชัน Wiener Filter (ลบ noise ได้ดีกว่า bandpass)
def apply_wiener_filter(y):
    return wiener(y)

# ฟังก์ชัน AI ลบเสียงรบกวน (ใช้โมเดลที่ฝึกไว้)
def denoise_with_ai(audio, sr):
    model = torch.hub.load("facebook/demucs", "demucs")  # ใช้โมเดล Demucs
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        denoised_audio = model(audio_tensor)
    return denoised_audio.squeeze(0).numpy()

# Streamlit UI
st.title("🎵 AI Noise Reduction")
st.write("🔍 อัปโหลดไฟล์เสียงที่มีเสียงรบกวน แล้วดูผลลัพธ์หลังจากลบ noise!")

# อัปโหลดไฟล์เสียง
uploaded_file = st.file_uploader("📤 อัปโหลดไฟล์เสียง (.wav)", type=["wav"])

if uploaded_file:
    # โหลดเสียง
    y, sr = librosa.load(uploaded_file, sr=None)

    # แสดงคลื่นเสียงต้นฉบับ
    st.write("🎧 คลื่นเสียงต้นฉบับ")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    st.pyplot(fig)

    # ใช้ Wiener Filtering เพื่อลด noise เบื้องต้น
    y_filtered = apply_wiener_filter(y)

    # ใช้ AI ลบเสียงรบกวน (Demucs)
    y_denoised = denoise_with_ai(y_filtered, sr)

    # แสดงคลื่นเสียงที่ลบ Noise แล้ว
    st.write("🔇 คลื่นเสียงหลังจากลบ Noise")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y_denoised, sr=sr, ax=ax)
    st.pyplot(fig)

    # ดาวน์โหลดไฟล์เสียงที่ผ่านการลบ Noise
    output_file = "denoised_output.wav"
    sf.write(output_file, y_denoised, sr)
    st.download_button("📥 ดาวน์โหลดไฟล์เสียงที่ปรับปรุงแล้ว", data=open(output_file, "rb"), file_name="denoised_output.wav")
