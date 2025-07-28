# File: data_loader.py

import os
import pandas as pd
from config import EMOTION_MAP, TARGET_EMOTION_MAPPING, EMOTION_SPEECH_MAPPING

def load_ravdess_data(data_path, target_mapping):
    """
    Scans the RAVDESS dataset directory, extracts emotion from filenames,
    and creates a DataFrame with file paths and their target labels.

    Args:
        data_path (str): The root path of the RAVDESS dataset.
        target_mapping (dict): A dictionary mapping emotion names to numerical labels.

    Returns:
        pandas.DataFrame: A DataFrame with 'file_path', 'label', and 'dataset' columns,
                          filtered to include only emotions present in target_mapping.
    """
    data = []
    
    # Walk through the directory structure
    for dirname, _, filenames in os.walk(data_path):
        for filename in filenames:
            if filename.endswith('.wav'):
                try:
                    # Filename format: 03-01-EMOTION-XX-XX-XX-ACTOR.wav
                    parts = filename.split('.')[0].split('-')
                    emotion_code = parts[2]
                    
                    # Map emotion code to emotion name
                    emotion_name = EMOTION_MAP.get(emotion_code)
                    
                    # If the emotion is one of our targets, add it to the list
                    if emotion_name in target_mapping:
                        label = target_mapping[emotion_name]
                        file_path = os.path.join(dirname, filename)
                        data.append({
                            "file_path": file_path, 
                            "label": label,
                            "dataset": "RAVDESS"
                        })
                except (IndexError, KeyError) as e:
                    print(f"Skipping RAVDESS file with unexpected format: {filename} ({e})")
                    continue
                    
    # Create DataFrame from the collected data
    df = pd.DataFrame(data)
    
    # Ensure labels are integers
    if not df.empty:
        df['label'] = df['label'].astype(int)
        
    return df

def load_emotion_speech_data(data_path, target_mapping, emotion_speech_mapping):
    """
    Scans the Emotion Speech Dataset directory and creates a DataFrame with file paths and labels.

    Args:
        data_path (str): The root path of the Emotion Speech Dataset.
        target_mapping (dict): A dictionary mapping emotion names to numerical labels.
        emotion_speech_mapping (dict): Mapping from dataset folder names to standard emotion names.

    Returns:
        pandas.DataFrame: A DataFrame with 'file_path', 'label', and 'dataset' columns.
    """
    data = []
    
    # Walk through the directory structure
    for speaker_dir in os.listdir(data_path):
        speaker_path = os.path.join(data_path, speaker_dir)
        if os.path.isdir(speaker_path):
            # Check each emotion folder
            for emotion_folder in os.listdir(speaker_path):
                emotion_path = os.path.join(speaker_path, emotion_folder)
                if os.path.isdir(emotion_path) and emotion_folder in emotion_speech_mapping:
                    
                    # Map folder name to standard emotion name
                    emotion_name = emotion_speech_mapping[emotion_folder]
                    
                    # If the emotion is one of our targets, process the files
                    if emotion_name in target_mapping:
                        label = target_mapping[emotion_name]
                        
                        # Process all wav files in this emotion folder
                        for filename in os.listdir(emotion_path):
                            if filename.endswith('.wav'):
                                file_path = os.path.join(emotion_path, filename)
                                data.append({
                                    "file_path": file_path,
                                    "label": label,
                                    "dataset": "EmotionSpeech"
                                })
    
    # Create DataFrame from the collected data
    df = pd.DataFrame(data)
    
    # Ensure labels are integers
    if not df.empty:
        df['label'] = df['label'].astype(int)
        
    return df

def load_combined_data(ravdess_path, emotion_speech_path, target_mapping):
    """
    Loads and combines data from both RAVDESS and Emotion Speech datasets.

    Args:
        ravdess_path (str): Path to RAVDESS dataset.
        emotion_speech_path (str): Path to Emotion Speech dataset.
        target_mapping (dict): A dictionary mapping emotion names to numerical labels.

    Returns:
        pandas.DataFrame: Combined DataFrame with all audio files from both datasets.
    """
    print("Loading RAVDESS dataset...")
    ravdess_df = load_ravdess_data(ravdess_path, target_mapping)
    print(f"Loaded {len(ravdess_df)} files from RAVDESS dataset")
    
    print("Loading Emotion Speech dataset...")
    emotion_speech_df = load_emotion_speech_data(emotion_speech_path, target_mapping, EMOTION_SPEECH_MAPPING)
    print(f"Loaded {len(emotion_speech_df)} files from Emotion Speech dataset")
    
    # Combine the datasets
    combined_df = pd.concat([ravdess_df, emotion_speech_df], ignore_index=True)
    print(f"Total combined dataset: {len(combined_df)} files")
    
    return combined_df