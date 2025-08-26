# ===============================
# Music Genre Classification App
# ===============================

import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import librosa
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models

import pickle

# -------------------------------
# DATASET PATH
# -------------------------------
DATA_DIR = "Data/genres_original"

# -------------------------------
# UTILS
# -------------------------------
def find_audio_files(root=DATA_DIR):
    wavs = []
    for dirpath, dirnames, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".wav"):
                wavs.append(os.path.join(dirpath, f))
    return wavs

def get_label_from_path(path):
    parts = path.replace("\\","/").split("/")
    for i in range(len(parts)-2, -1, -1):
        if parts[i].lower() in ["blues","classical","country","disco","hiphop",
                                "jazz","metal","pop","reggae","rock"]:
            return parts[i].lower()
    return parts[-2].lower()

def extract_tabular_features(path, sr=22050, duration=30.0):
    y, sr = librosa.load(path, sr=sr, duration=duration, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features = []
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    features += [np.mean(spec_cent), np.std(spec_cent)]
    features += [np.mean(spec_bw), np.std(spec_bw)]
    features += [np.mean(rolloff), np.std(rolloff)]
    features += [np.mean(zcr), np.std(zcr)]
    features += [np.mean(chroma), np.std(chroma)]
    
    return np.array(features, dtype=np.float32).reshape(1,-1)

def mel_spectrogram_128x128(path, sr=22050, duration=30.0, n_mels=128, hop_length=512):
    y, sr = librosa.load(path, sr=sr, duration=duration, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    need = 128
    if S_db.shape[1] < need:
        S_db = np.pad(S_db, ((0,0),(0,need - S_db.shape[1])), mode="constant")
    else:
        S_db = S_db[:, :need]
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min())
    return S_db[np.newaxis, ..., np.newaxis].astype(np.float32)

def build_cnn(input_shape, n_classes):
    model = models.Sequential([
        layers.Conv2D(16, (3,3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(32, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# -------------------------------
# LOAD DATA & EXTRACT FEATURES
# -------------------------------
st.info("Loading dataset and extracting features... This may take a few minutes.")

all_wavs = find_audio_files()
labels = [get_label_from_path(p) for p in all_wavs]

# Tabular features
X_tab, y_tab = [], []
for p in all_wavs:
    try:
        X_tab.append(extract_tabular_features(p))
        y_tab.append(get_label_from_path(p))
    except:
        pass
X_tab = np.vstack(X_tab)
le = LabelEncoder()
y_tab_enc = le.fit_transform(y_tab)

Xtr, Xte, ytr, yte = train_test_split(X_tab, y_tab_enc, test_size=0.2, random_state=42, stratify=y_tab_enc)
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xte_s = scaler.transform(Xte)

# RandomForest
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(Xtr_s, ytr)

# CNN features
X_img, y_img = [], []
for p in all_wavs:
    try:
        X_img.append(mel_spectrogram_128x128(p))
        y_img.append(get_label_from_path(p))
    except:
        pass
X_img = np.array(X_img)
y_img_enc = le.transform(y_img)
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(X_img, y_img_enc, test_size=0.2, random_state=42, stratify=y_img_enc)

cnn = build_cnn(Xi_tr.shape[1:], len(le.classes_))
cnn.fit(Xi_tr, yi_tr, validation_data=(Xi_te, yi_te), epochs=5, batch_size=32)

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("🎵 Music Genre Classifier")
st.write("Upload a song and predict its genre using Tabular or CNN model.")

uploaded_file = st.file_uploader("Choose a .wav file", type="wav")
model_type = st.selectbox("Select model type:", ["Tabular (MFCC+Spectral)", "Image-based (CNN)"])

if uploaded_file is not None:
    temp_path = "temp.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if model_type == "Tabular (MFCC+Spectral)":
        features = extract_tabular_features(temp_path)
        features_scaled = scaler.transform(features)
        pred = rf.predict(features_scaled)
        genre = le.inverse_transform(pred)[0]
    else:  # CNN
        spec = mel_spectrogram_128x128(temp_path)
        pred = np.argmax(cnn.predict(spec), axis=1)
        genre = le.inverse_transform(pred)[0]
    
    st.success(f"Predicted genre: **{genre.upper()}**")
    
    if st.checkbox("Show Mel-Spectrogram"):
        import librosa.display
        y, sr = librosa.load(temp_path, sr=22050)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(8,4))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-Spectrogram')
        st.pyplot(plt)
    
    os.remove(temp_path)

# -------------------------------
# SAVE MODELS
# -------------------------------
pickle.dump(rf, open("rf_model.pkl","wb"))
pickle.dump(scaler, open("scaler.pkl","wb"))
pickle.dump(le, open("label_encoder.pkl","wb"))
cnn.save("cnn_model.h5")
