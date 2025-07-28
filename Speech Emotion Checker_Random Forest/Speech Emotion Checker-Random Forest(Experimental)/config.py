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
# PLEASE UPDATE this path to where you have stored the RAVDESS dataset.
DATA_PATH = "./RAVDESS_data/"

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
# 'Confident' will be our positive class (1), and 'Nervous' will be our negative class (0).
TARGET_EMOTION_MAPPING = {
    'fearful': 0,  # Mapped to Nervous
    'sad': 0,      # Mapped to Nervous
    'neutral': 1,  # Mapped to Confident
    'calm': 1,     # Mapped to Confident
    'happy': 1     # Mapped to Confident
}

# --- Feature Configuration ---
# List of all feature names that will be extracted. This ensures consistency.
FEATURE_COLUMNS = [
    'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 
    'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'mfcc_14', 'mfcc_15', 'mfcc_16', 
    'mfcc_17', 'mfcc_18', 'mfcc_19', 'mfcc_20', 'mfcc_21', 'mfcc_22', 'mfcc_23', 'mfcc_24', 
    'mfcc_25', 'mfcc_26', 'mfcc_27', 'mfcc_28', 'mfcc_29', 'mfcc_30', 'mfcc_31', 'mfcc_32', 
    'mfcc_33', 'mfcc_34', 'mfcc_35', 'mfcc_36', 'mfcc_37', 'mfcc_38', 'mfcc_39', 'mfcc_40',
    'chroma_stft', 'rms_energy', 'zero_crossing_rate', 'melspectrogram',
    'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 
    'spectral_contrast', 'spectral_flatness', 'pitch_mean', 'pitch_std'
]