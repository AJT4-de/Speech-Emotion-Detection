# File: feature_extractor.py

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import FEATURE_COLUMNS

def extract_features(audio_path, n_mfcc=40):
    """
    Extracts a comprehensive set of audio features from a single audio file.

    Args:
        audio_path (str): Path to the audio file.
        n_mfcc (int): The number of MFCCs to compute.

    Returns:
        numpy.ndarray: A 1D array containing all extracted features, or None if extraction fails.
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, duration=3, sr=22050)
        
        # --- Standard Features ---
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        
        # --- Spectral Features ---
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T, axis=0)
        spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)
        spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        spec_flatness = np.mean(librosa.feature.spectral_flatness(y=y).T, axis=0)
        
        # --- Pitch (Fundamental Frequency) ---
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pitch_mean = np.mean(f0[voiced_flag]) if np.any(voiced_flag) else 0
        pitch_std = np.std(f0[voiced_flag]) if np.any(voiced_flag) else 0

        # Concatenate all features into a single vector
        features = np.hstack((
            mfccs, chroma, rms, zcr, mel,
            spec_centroid, spec_bw, spec_rolloff, spec_contrast, spec_flatness,
            [pitch_mean], [pitch_std]
        ))
        
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def process_all_features(dataframe):
    """
    Applies feature extraction to every file in the input DataFrame.

    Args:
        dataframe (pandas.DataFrame): DataFrame with a 'file_path' column.

    Returns:
        pandas.DataFrame: A new DataFrame containing the extracted features,
                          with columns named according to config.FEATURE_COLUMNS.
    """
    features_list = []
    
    # Use tqdm for a progress bar
    for index, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0], desc="Extracting Features"):
        audio_path = row['file_path']
        features = extract_features(audio_path)
        if features is not None:
            features_list.append(features)
            
    # Create a DataFrame from the list of features
    feature_df = pd.DataFrame(features_list, columns=FEATURE_COLUMNS)
    return feature_df