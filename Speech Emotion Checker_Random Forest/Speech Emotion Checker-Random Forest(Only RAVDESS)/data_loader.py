# File: data_loader.py

import os
import pandas as pd
from config import EMOTION_MAP, TARGET_EMOTION_MAPPING

def load_ravdess_data(data_path, target_mapping):
    """
    Scans the RAVDESS dataset directory, extracts emotion from filenames,
    and creates a DataFrame with file paths and their target labels.

    Args:
        data_path (str): The root path of the RAVDESS dataset.
        target_mapping (dict): A dictionary mapping emotion names to numerical labels.

    Returns:
        pandas.DataFrame: A DataFrame with 'file_path' and 'label' columns,
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
                        data.append({"file_path": file_path, "label": label})
                except (IndexError, KeyError) as e:
                    print(f"Skipping file with unexpected format: {filename} ({e})")
                    continue
                    
    # Create DataFrame from the collected data
    df = pd.DataFrame(data)
    
    # Ensure labels are integers
    if not df.empty:
        df['label'] = df['label'].astype(int)
        
    return df