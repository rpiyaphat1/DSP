import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import torch
import torchaudio

# ฟังก์ชันกรองเสียงรบกวนด้วย DSP (Butterworth filter)
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

# ฟังก์ชันลบเสียงรบกวนด้วย AI (ใช้โมเดลที่ฝึกมา)
def denoise_with_ai(audio):
    # โหลดโมเดล AI ที่ฝึกมา (เช่น Denoising Autoencoder)
    model = torch.jit.load("denoiser_model.pt")
    model.eval()

    # แปลงเสียงเป็น Tensor
    audio_tensor = torch.tensor(audio, dtype=torch.float32)
    denoised_audio = model(audio_tensor)
    
    return denoised_audio.detach().numpy()

# Streamlit UI
st.title("AI + DSP Noise Reduction App")
st.write("อัปโหลดไฟล์เสียงที่มีเสียงรบกวนและดูผลลัพธ์หลังจากกรองเสียง")

# อัปโหลดไฟล์เสียง
uploaded_file = st.file_uploader("อัปโหลดไฟล์เสียง (.wav)", type=["wav"])

if uploaded_file:
    # โหลดเสียง
    y, sr = librosa.load(uploaded_file, sr=None)

    # แสดงคลื่นเสียงต้นฉบับ
    st.write("🎵 คลื่นเสียงต้นฉบับ")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    st.pyplot(fig)

    # ใช้ DSP Filter
    y_filtered = apply_bandpass_filter(y, 300, 3400, sr)

    # ใช้ AI ลบเสียงรบกวน
    y_denoised = denoise_with_ai(y_filtered)

    # แสดงคลื่นเสียงที่ผ่านการลบ Noise
    st.write("🔇 คลื่นเสียงหลังจากลบ Noise")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y_denoised, sr=sr, ax=ax)
    st.pyplot(fig)

    # ดาวน์โหลดไฟล์เสียงที่ผ่านการลบ Noise
    output_file = "denoised_output.wav"
    sf.write(output_file, y_denoised, sr)
    st.download_button("📥 ดาวน์โหลดไฟล์เสียงที่ปรับปรุงแล้ว", data=open(output_file, "rb"), file_name="denoised_output.wav")

