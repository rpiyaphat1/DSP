import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import os
import scipy.signal as signal
import scipy.stats
import sounddevice as sd
import wave

# ✅ โหลดโมเดลที่ Train แล้ว
class BritishAccentScoreModel(nn.Module):
    def __init__(self, input_size):
        super(BritishAccentScoreModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # 1 Output = British Accent Score
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # ✅ ลด Overfitting
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return self.sigmoid(x) * 100  #  British Accent Score (0-100%)

# ✅ โหลดโมเดล
model = BritishAccentScoreModel(input_size=13)
model.load_state_dict(torch.load("british_accent_score_model.pth", map_location=torch.device('cpu')))
model.eval()

# ✅ ฟังก์ชันประมวลผลเสียงก่อนเข้าโมเดล (DSP Processing)
def preprocess_audio(file_path, sr=16000, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=sr)

    # ✅ Bandpass Filter (300Hz - 3400Hz)
    lowcut = 300.0
    highcut = 3400.0
    b, a = signal.butter(4, [lowcut / (sr / 2), highcut / (sr / 2)], btype='band')
    y_filtered = signal.filtfilt(b, a, y)

    # ✅ Normalization
    y_filtered = y_filtered / np.max(np.abs(y_filtered))

    # ✅ Extract MFCC + Delta MFCC
    mfcc = librosa.feature.mfcc(y=y_filtered, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_combined = np.concatenate((mfcc, mfcc_delta), axis=0)
    return np.mean(mfcc_combined, axis=1)  # คืนค่าเฉลี่ย MFCC + Delta MFCC

# ✅ ฟังก์ชันบันทึกเสียงจากไมค์
def record_audio(filename="recorded_audio.wav", duration=5, sr=16000):
    st.info("🎤 กำลังบันทึกเสียง... พูดออกมาเลย!")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype=np.int16)
    sd.wait()
    
    # ✅ บันทึกเป็นไฟล์ WAV
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sr)
        wf.writeframes(recording.tobytes())
    
    st.success("✅ บันทึกเสียงเสร็จแล้ว!")
    return filename

# ✅ สร้าง Streamlit UI
st.title("🎤 British Accent Detector")
st.write("กดปุ่มเพื่อบันทึกเสียง แล้วตรวจสอบความใกล้เคียงสำเนียง British")

if st.button("🎙️ กดเพื่อบันทึกเสียง (5 วินาที)"):
    recorded_file = record_audio()
    
    # ✅ ประมวลผลเสียง
    mfcc_features = preprocess_audio(recorded_file)
    if mfcc_features is not None:
        # ✅ แปลงเป็น Tensor
        X_test_tensor = torch.tensor(mfcc_features, dtype=torch.float32).unsqueeze(0)
        
        # ✅ ทำนาย British Accent Score
        with torch.no_grad():
            score_raw = model(X_test_tensor).item()
        
        # ✅ ปรับค่า Confidence ด้วย Temperature Scaling
        temperature = 2.5
        softmax_output = np.exp(score_raw / temperature) / np.sum(np.exp(score_raw / temperature))
        british_confidence = softmax_output * 100
        
        # ✅ คำนวณความใกล้เคียงสำเนียง British
        british_mean = 90  # ค่าเฉลี่ยของ British Accent
        british_std = 15   # ส่วนเบี่ยงเบนมาตรฐานของ British Accent
        british_similarity = scipy.stats.norm.pdf(score_raw, british_mean, british_std)
        british_similarity_percentage = (british_similarity / np.max(british_similarity)) * 100

        # ✅ แสดงผล
        st.success(f"🎯 British Accent Similarity: {british_similarity_percentage:.2f}%")
    else:
        st.error("⚠️ ไม่สามารถประมวลผลไฟล์เสียงนี้ได้")
