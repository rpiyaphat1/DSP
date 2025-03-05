import streamlit as st
import torch
import librosa
import numpy as np
import os
import scipy.signal as signal

# ✅ โหลดโมเดลที่ Train แล้ว
class BritishAccentScoreModel(nn.Module):
    def __init__(self, input_size):
        super(BritishAccentScoreModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # 1 Output = British Accent Score
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
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

    # ✅ Extract MFCC
    mfcc = librosa.feature.mfcc(y=y_filtered, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)  # คืนค่าเฉลี่ย MFCC Features

# ✅ สร้าง Streamlit UI
st.title("Did you have a British accent?")
st.write("อัปโหลดไฟล์เสียงเพื่อตรวจสอบความใกล้เคียงสำเนียง British")

uploaded_file = st.file_uploader("อัปโหลดไฟล์เสียง (WAV)", type=["wav"])

if uploaded_file is not None:
    # ✅ บันทึกไฟล์ชั่วคราว
    file_path = f"temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # ✅ ประมวลผลเสียง
    mfcc_features = preprocess_audio(file_path)
    if mfcc_features is not None:
        # ✅ แปลงเป็น Tensor
        X_test_tensor = torch.tensor(mfcc_features, dtype=torch.float32).unsqueeze(0)
        
        # ✅ ทำนาย British Accent Score
        with torch.no_grad():
            score = model(X_test_tensor).item()
        
        st.success(f"🎯 British Accent Score: {score:.2f}%")
    else:
        st.error("⚠️ ไม่สามารถประมวลผลไฟล์เสียงนี้ได้")
