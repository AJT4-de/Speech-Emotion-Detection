# File: validate_model.py

import joblib
import os
import glob
from predict import predict_emotion
from config import MODEL_PATH, EMOTION_MAP, TARGET_EMOTION_MAPPING
import numpy as np

def extract_true_emotion_from_filename(filename):
    """Extract the true emotion from RAVDESS filename"""
    # RAVDESS filename format: 03-02-{emotion}-01-01-01-01.wav
    parts = os.path.basename(filename).split('-')
    if len(parts) >= 3:
        emotion_code = parts[2]
        return EMOTION_MAP.get(emotion_code, 'unknown')
    return 'unknown'

def validate_model_performance():
    """Validate the model on a subset of test files"""
    
    # Load the model and scaler
    try:
        model = joblib.load(f"{MODEL_PATH}ser_model.pkl")
        scaler = joblib.load(f"{MODEL_PATH}ser_scaler.pkl")
        print("Model and scaler loaded successfully.")
    except FileNotFoundError:
        print("ERROR: Model or scaler not found.")
        return
    
    # Get a sample of files from different actors for testing
    test_files = []
    for actor in ['Actor_01', 'Actor_02', 'Actor_03']:
        actor_path = f"RAVDESS_data/{actor}/*.wav"
        actor_files = glob.glob(actor_path)
        # Take first 5 files from each actor
        test_files.extend(actor_files[:5])
    
    print(f"\nTesting on {len(test_files)} files...")
    print("-" * 80)
    
    correct_predictions = 0
    total_predictions = 0
    results = []
    
    for file_path in test_files:
        # Get true emotion
        true_emotion = extract_true_emotion_from_filename(file_path)
        
        # Skip if we can't map the emotion or it's not in our target emotions
        if true_emotion == 'unknown' or true_emotion not in TARGET_EMOTION_MAPPING:
            continue
            
        # Get prediction
        result = predict_emotion(file_path, model, scaler)
        
        # Extract predicted emotion from result string
        lines = result.split('\n')
        predicted_emotion = lines[0].split(': ')[1].split(' (')[0]
        confidence = float(lines[0].split('Confidence: ')[1].split(')')[0])
        
        # Check if prediction is correct
        is_correct = predicted_emotion == true_emotion
        if is_correct:
            correct_predictions += 1
        total_predictions += 1
        
        # Store result
        results.append({
            'file': os.path.basename(file_path),
            'true': true_emotion,
            'predicted': predicted_emotion,
            'confidence': confidence,
            'correct': is_correct
        })
        
        # Print result
        status = "✓" if is_correct else "✗"
        print(f"{status} {os.path.basename(file_path):<25} | True: {true_emotion:<8} | Pred: {predicted_emotion:<8} | Conf: {confidence:.2f}")
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    print("-" * 80)
    print(f"Validation Results:")
    print(f"Correct predictions: {correct_predictions}/{total_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Show confusion matrix style breakdown
    print("\nBreakdown by emotion:")
    emotion_stats = {}
    for result in results:
        true_emotion = result['true']
        if true_emotion not in emotion_stats:
            emotion_stats[true_emotion] = {'correct': 0, 'total': 0}
        emotion_stats[true_emotion]['total'] += 1
        if result['correct']:
            emotion_stats[true_emotion]['correct'] += 1
    
    for emotion in sorted(emotion_stats.keys()):
        stats = emotion_stats[emotion]
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {emotion:<10}: {stats['correct']}/{stats['total']} ({acc:.1%})")

if __name__ == "__main__":
    validate_model_performance()
