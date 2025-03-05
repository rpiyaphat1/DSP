import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import torchaudio
from scipy.signal import butter, lfilter, spectrogram

# ฟังก์ชันสร้าง High-Pass Filter
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

# ฟังก์ชันใช้ High-Pass Filter
def apply_highpass_filter(y, cutoff=300, fs=44100, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    return lfilter(b, a, y)

# โหลดโมเดล AI Demucs
@st.cache_resource
def load_model():
    model = torch.hub.load("facebook/demucs", "demucs")  # หรือใช้ DeepFilterNet
    model.eval()
    return model

model = load_model()

# ฟังก์ชัน AI Noise Reduction (Demucs)
def denoise_with_ai(audio, sr):
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        denoised_audio = model(audio_tensor)
    return denoised_audio.squeeze(0).numpy()

# ฟังก์ชัน Spectral Gating
def spectral_gate(y, sr, threshold=0.1):
    _, _, Sxx = spectrogram(y, fs=sr)
    mask = Sxx > (threshold * np.max(Sxx))  # สร้าง mask ให้ลบ noise ต่ำกว่า threshold
    y_clean = y * mask[:len(y)]
    return y_clean

# Streamlit UI
st.title("🎵 High-Pass Filter + AI Noise Reduction")
st.write("🔍 อัปโหลดไฟล์เสียง แล้วลบ noise ความถี่ต่ำ!")

uploaded_file = st.file_uploader("📤 อัปโหลดไฟล์เสียง (.wav)", type=["wav"])

if uploaded_file:
    y, sr = librosa.load(uploaded_file, sr=None)

    # แสดงคลื่นเสียงต้นฉบับ
    st.write("🎧 คลื่นเสียงต้นฉบับ")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    st.pyplot(fig)

    # ใช้ High-Pass Filter
    y_highpass = apply_highpass_filter(y, cutoff=300, fs=sr)

    # ใช้ AI ลบเสียงรบกวน
    y_denoised = denoise_with_ai(y_highpass, sr)

    # ใช้ Spectral Gating เพื่อลบ noise เพิ่มเติม
    y_final = spectral_gate(y_denoised, sr)

    # แสดงคลื่นเสียงที่ลบ noise แล้ว
    st.write("🔇 คลื่นเสียงหลังจากลบ Noise")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y_final, sr=sr, ax=ax)
    st.pyplot(fig)

    # ดาวน์โหลดไฟล์เสียงที่ผ่านการลบ Noise
    output_file = "denoised_output.wav"
    sf.write(output_file, y_final, sr)
    st.download_button("📥 ดาวน์โหลดไฟล์เสียงที่ปรับปรุงแล้ว", data=open(output_file, "rb"), file_name="denoised_output.wav")
