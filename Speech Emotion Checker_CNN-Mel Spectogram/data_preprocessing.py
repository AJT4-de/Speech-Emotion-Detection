# data_preprocessing.py

"""
Handles loading audio data, extracting features (Mel-spectrograms),
and preparing it for the SER model.
"""

import os
import glob
import numpy as np
import librosa
import soundfile as sf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from config import DATA_PATHS, EMOTIONS, SAMPLE_RATE, DURATION, N_MELS, HOP_LENGTH, FMAX

def extract_emotion_from_filename(filename):
    """
    Extracts the emotion code from the RAVDESS or ESD filename.
    - RAVDESS example: '03-01-03-02-01-01-01.wav' -> '03' (happy)
    - ESD example: '0011_a_n_01.wav' -> 'a' (angry)
    """
    try:
        base_name = os.path.basename(filename)
        
        # Check for RAVDESS format (e.g., 03-01-03-...)
        if '-' in base_name:
            parts = base_name.split('-')
            if len(parts) > 2:
                emotion_code = parts[2]
                return EMOTIONS.get(emotion_code)

        # Check for ESD format (e.g., 0011_a_n_01.wav)
        elif '_' in base_name:
            parts = base_name.split('_')
            if len(parts) > 1:
                emotion_code = parts[1]
                return EMOTIONS.get(emotion_code)
                
    except (IndexError, KeyError):
        # Return None if the filename format is not recognized or part is missing
        return None
    
    return None


def load_and_extract_features():
    """
    Loads audio files, extracts Mel-spectrograms, and prepares them for the model.

    This function iterates through the specified data paths, processes each audio file to
    extract Mel-spectrograms, and normalizes them.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Processed and normalized spectrograms (features).
            - np.ndarray: Encoded emotion labels.
            - LabelEncoder: The fitted label encoder for decoding predictions.
    """
    features = []
    labels = []
    
    print("Starting data loading and feature extraction...")
    
    all_files = []
    for path in DATA_PATHS:
        # Use glob to find all .wav files recursively
        all_files.extend(glob.glob(os.path.join(path, '**', '*.wav'), recursive=True))

    print(f"Found {len(all_files)} audio files from paths: {DATA_PATHS}")

    for file_path in all_files:
        emotion = extract_emotion_from_filename(file_path)
        if emotion is None:
            continue

        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
            
            # Pad or truncate to the fixed duration
            target_length = int(DURATION * SAMPLE_RATE)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
            else:
                audio = audio[:target_length]

            # Generate Mel-spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio,
                sr=SAMPLE_RATE,
                n_mels=N_MELS,
                hop_length=HOP_LENGTH,
                fmax=FMAX
            )
            
            # Convert to decibels (log scale)
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            features.append(log_mel_spectrogram)
            labels.append(emotion)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Successfully processed {len(features)} files.")
    
    if not features:
        raise ValueError("No features were extracted. Check data paths and file formats.")

    # Convert lists to numpy arrays
    X = np.array(features)
    y = np.array(labels)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Reshape for CNN input (num_samples, n_mels, num_frames, 1)
    X = X[..., np.newaxis]
    
    # Note: Normalization (scaling) is done in train.py after splitting the data
    
    print("Feature extraction complete.")
    return X, y_encoded, label_encoder
