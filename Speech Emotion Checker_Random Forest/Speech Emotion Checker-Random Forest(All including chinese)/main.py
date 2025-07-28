# File: main.py

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Import our custom modules
from config import DATA_PATH, EMOTION_SPEECH_PATH, TARGET_EMOTION_MAPPING, MODEL_PATH
from data_loader import load_combined_data
from feature_extractor import process_all_features
from model_trainer import train_model, evaluate_model

def main(use_subset=True, subset_size=5000):
    """
    Main function to orchestrate the SER project workflow.
    
    Args:
        use_subset (bool): Whether to use a subset of data for faster training
        subset_size (int): Size of subset to use if use_subset is True
    """
    print("--- Starting Multi-Dataset Speech Emotion Recognition Project ---")

    # 1. Load Data from both datasets
    print("\nStep 1: Loading and combining datasets...")
    raw_data_df = load_combined_data(DATA_PATH, EMOTION_SPEECH_PATH, TARGET_EMOTION_MAPPING)
    if raw_data_df.empty:
        print("No data loaded. Please check paths in config.py and your datasets.")
        return
    print(f"Loaded {len(raw_data_df)} audio files for emotion classification.")
    print("Class distribution:\n", raw_data_df['label'].value_counts().sort_index())
    print("Dataset distribution:\n", raw_data_df['dataset'].value_counts())
    
    # Use subset for faster training if requested
    if use_subset and len(raw_data_df) > subset_size:
        print(f"\nUsing subset of {subset_size} files for faster training...")
        # Stratified sampling to maintain emotion distribution
        raw_data_df = raw_data_df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(min(len(x), subset_size // len(raw_data_df['label'].unique())), 
                              random_state=42)
        ).reset_index(drop=True)
        print(f"Subset size: {len(raw_data_df)} files")
        print("Subset class distribution:\n", raw_data_df['label'].value_counts().sort_index())

    # 2. Extract Features
    print("\nStep 2: Extracting features from audio files...")
    # This can take a while depending on the number of files and CPU.
    feature_df = process_all_features(raw_data_df)
    
    # Combine features with labels
    full_df = pd.concat([feature_df, raw_data_df['label']], axis=1)
    full_df.dropna(inplace=True) # Drop rows if feature extraction failed for any file
    
    print(f"Feature extraction complete. Shape of feature set: {full_df.shape}")

    # 3. Prepare Data for Modeling
    X = full_df.drop('label', axis=1)
    y = full_df['label']
    
    # Split data into training and testing sets
    # test_size=0.25 means 25% of the data is for testing
    # stratify=y ensures the class distribution is the same in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\nStep 3: Data split into training ({len(X_train)}) and testing ({len(X_test)}) sets.")

    # 4. Train Model
    print("\nStep 4: Training the model...")
    model, scaler = train_model(X_train, y_train)

    # 5. Evaluate Model
    print("\nStep 5: Evaluating the model on the test set...")
    evaluate_model(model, scaler, X_test, y_test)

    # 6. Save Model and Scaler
    print("\nStep 6: Saving the trained model and scaler...")
    joblib.dump(model, f"{MODEL_PATH}ser_model.pkl")
    joblib.dump(scaler, f"{MODEL_PATH}ser_scaler.pkl")
    print(f"Model and scaler saved to '{MODEL_PATH}' directory.")
    
    print("\n--- Project Workflow Complete ---")
    print("You can now use 'predict.py' to classify new audio files.")


if __name__ == "__main__":
    main()