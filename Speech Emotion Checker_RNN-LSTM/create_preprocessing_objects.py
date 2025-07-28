#!/usr/bin/env python3
"""
Script to recreate the scaler and label_encoder from the dataset
when they weren't saved during training.
"""

import os
import numpy as np
import pandas as pd
import librosa
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf

# --- Parameters (same as in ser_model.py) ---
DATA_PATH = './ravdess/'
SAMPLING_RATE = 48000
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
MAX_PAD_LEN = 250

EMOTIONS = {
    '01': 'neutral',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '08': 'surprised'
}

def load_data(data_path):
    """Load audio file paths and their corresponding emotion labels."""
    audio_paths = []
    emotion_labels = []
    
    for actor_dir in os.listdir(data_path):
        actor_path = os.path.join(data_path, actor_dir)
        if os.path.isdir(actor_path):
            for file_name in os.listdir(actor_path):
                if file_name.endswith('.wav'):
                    parts = file_name.split('-')
                    emotion_code = parts[2]
                    if emotion_code in EMOTIONS:
                        audio_paths.append(os.path.join(actor_path, file_name))
                        emotion_labels.append(EMOTIONS[emotion_code])
                        
    return pd.DataFrame({'path': audio_paths, 'emotion': emotion_labels})

def extract_features(audio_path):
    """Extract MFCC features from an audio file."""
    try:
        audio, sr = librosa.load(audio_path, sr=SAMPLING_RATE, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        return mfccs.T
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def main():
    print("Creating preprocessing objects...")
    
    # Load data
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} audio files.")
    
    # Extract features for a sample to create scaler
    print("Extracting features for preprocessing object creation...")
    features_list = []
    
    # Extract features from first few files to create scaler
    for i, path in enumerate(df['path'].head(20)):  # Use first 20 files
        features = extract_features(path)
        if features is not None:
            features_list.append(features)
        print(f"Processed {i+1}/20 files", end='\r')
    
    print(f"\nExtracted features from {len(features_list)} files.")
    
    # Pad features
    padded_features = tf.keras.preprocessing.sequence.pad_sequences(
        features_list, maxlen=MAX_PAD_LEN, dtype='float32', padding='post', truncating='post'
    )
    
    # Create and fit scaler
    scaler = StandardScaler()
    scaler.fit(padded_features.reshape(-1, padded_features.shape[2]))
    
    # Create and fit label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(df['emotion'])
    
    # Save the objects
    print("Saving preprocessing objects...")
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    
    print("✅ Preprocessing objects created and saved successfully!")
    print(f"  - Scaler: models/scaler.pkl")
    print(f"  - Label Encoder: models/label_encoder.pkl")
    print(f"  - Emotions: {label_encoder.classes_}")

if __name__ == '__main__':
    main()
