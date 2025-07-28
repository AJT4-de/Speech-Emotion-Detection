# File: predict.py

import joblib
import numpy as np
import os
from feature_extractor import extract_features
from config import MODEL_PATH, EMOTION_LABELS

# Define the inverse mapping from label to emotion name
LABEL_TO_EMOTION = {i: emotion for i, emotion in enumerate(EMOTION_LABELS)}

def predict_emotion(audio_path, model, scaler):
    """
    Predicts the emotion from a single audio file.

    Args:
        audio_path (str): Path to the audio file.
        model: The trained machine learning model.
        scaler: The trained StandardScaler.

    Returns:
        str: The predicted emotion name, or None on error.
    """
    try:
        # Extract features from the new audio file
        features = extract_features(audio_path)
        if features is None:
            return "Could not extract features."

        # Reshape features to a 2D array for the scaler and model
        features = features.reshape(1, -1)
        
        # Scale the features using the loaded scaler
        scaled_features = scaler.transform(features)
        
        # Predict the probability for each class
        prediction_proba = model.predict_proba(scaled_features)
        
        # Get the predicted label and the confidence score
        predicted_label = np.argmax(prediction_proba)
        confidence_score = prediction_proba[0][predicted_label]
        
        predicted_emotion = LABEL_TO_EMOTION[predicted_label]
        
        # Also show top 3 predictions for more insight
        top_3_indices = np.argsort(prediction_proba[0])[-3:][::-1]
        top_3_emotions = [(LABEL_TO_EMOTION[i], prediction_proba[0][i]) for i in top_3_indices]
        
        result = f"Predicted Emotion: {predicted_emotion} (Confidence: {confidence_score:.2f})\n"
        result += "Top 3 predictions:\n"
        for i, (emotion, prob) in enumerate(top_3_emotions, 1):
            result += f"  {i}. {emotion}: {prob:.2f}\n"
        
        return result
        
    except Exception as e:
        return f"An error occurred during prediction: {e}"

def main_predictor():
    """
    Main function to run the prediction CLI.
    """
    print("--- Multi-Emotion Speech Recognition Predictor ---")
    print("Supported emotions: " + ", ".join(EMOTION_LABELS))
    
    try:
        # Load the saved model and scaler
        model = joblib.load(f"{MODEL_PATH}ser_model.pkl")
        scaler = joblib.load(f"{MODEL_PATH}ser_scaler.pkl")
        print("Model and scaler loaded successfully.")
    except FileNotFoundError:
        print("\nERROR: Model or scaler not found.")
        print("Please run the 'main.py' script first to train and save the model.")
        return

    while True:
        audio_path = input("\nEnter the path to an audio file (or 'q' to quit): ").strip()
        if audio_path.lower() == 'q':
            break
        
        if not os.path.exists(audio_path):
            print("File not found. Please check the path and try again.")
            continue
            
        # Get the prediction
        result = predict_emotion(audio_path, model, scaler)
        print(result)

if __name__ == '__main__':
    # This allows you to run prediction independently after training the model.
    main_predictor()