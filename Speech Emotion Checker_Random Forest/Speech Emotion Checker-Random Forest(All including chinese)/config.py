# File: config.py

import os
import numpy as np

def create_dir_if_not_exists(path):
    """
    Helper function to create a directory if it doesn't already exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

# --- Path Configuration ---
# Path to the root directory of the RAVDESS dataset.
DATA_PATH = "./RAVDESS_data/"

# Path to the new Emotional Speech Dataset
EMOTION_SPEECH_PATH = "./Emotion Speech Dataset/"

# Path to save trained models and scalers
MODEL_PATH = "./models/"
create_dir_if_not_exists(MODEL_PATH)


# --- Emotion Mapping Configuration ---
# RAVDESS emotions are coded in the filename. This map translates the code to a readable name.
EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# We define our target emotions and map them to numerical labels.
# Updated to support multi-class emotion recognition (7 emotions from both datasets)
TARGET_EMOTION_MAPPING = {
    'neutral': 0,
    'calm': 1,
    'happy': 2,
    'sad': 3,
    'angry': 4,
    'fearful': 5,
    'surprise': 6
}

# Emotion labels for display purposes
EMOTION_LABELS = [
    'neutral',
    'calm', 
    'happy',
    'sad',
    'angry',
    'fearful',
    'surprise'
]

# Mapping for Emotion Speech Dataset folder names to our standard names
EMOTION_SPEECH_MAPPING = {
    'Neutral': 'neutral',
    'Happy': 'happy',
    'Sad': 'sad',
    'Angry': 'angry',
    'Surprise': 'surprise'
}

# --- Feature Configuration ---
# List of all feature names that will be extracted. This ensures consistency.
FEATURE_COLUMNS = (
    # MFCCs (40 dimensions)
    [f'mfcc_{i}' for i in range(1, 41)] +
    # Chroma (12 dimensions)
    [f'chroma_{i}' for i in range(1, 13)] +
    # RMS (1 dimension)
    ['rms_energy'] +
    # Zero Crossing Rate (1 dimension)
    ['zero_crossing_rate'] +
    # Melspectrogram (128 dimensions)
    [f'mel_{i}' for i in range(1, 129)] +
    # Spectral features (5 dimensions total)
    ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff'] +
    [f'spectral_contrast_{i}' for i in range(1, 8)] +
    ['spectral_flatness'] +
    # Pitch features (2 dimensions)
    ['pitch_mean', 'pitch_std']
)