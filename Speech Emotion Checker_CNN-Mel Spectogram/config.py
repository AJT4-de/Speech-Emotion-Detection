# config.py

"""
Configuration file for the Speech Emotion Recognition (SER) model.
Centralizes all parameters for easy modification and tuning.
"""

import os

# 1. Dataset and Audio Parameters
# ---------------------------------
# List of paths to the datasets. This now includes both RAVDESS and ESD.
DATA_PATHS = [
    './datasets/RAVDESS/',
    './datasets/Emotion Speech Dataset/'
]

# Combined emotion mapping for both RAVDESS and ESD datasets.
EMOTIONS = {
    # RAVDESS emotions (by number)
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprise',

    # ESD emotions (by letter)
    'n': 'neutral',
    'h': 'happy',
    's': 'sad',
    'a': 'angry',
    'u': 'surprise'
}

# Audio processing parameters
SAMPLE_RATE = 22050  # Hz, standard for audio analysis
DURATION = 3         # seconds, to pad or truncate audio clips to a fixed length
N_MELS = 128         # Number of Mel bands to generate
HOP_LENGTH = 512     # Number of samples between successive frames
FMAX = 8000          # Maximum frequency for Mel-spectrogram

# 2. Model Training Parameters
# -----------------------------
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# 3. Model Saving and Loading
# ----------------------------
MODEL_SAVE_DIR = './models'
MODEL_NAME_PREFIX = 'ser_cnn'

# Path to a pre-trained model for incremental learning.
# Set to None to train a new model from scratch.
# Example: './models/ser_cnn_best_model.keras'
PRETRAINED_MODEL_PATH = None

# Create model save directory if it doesn't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)