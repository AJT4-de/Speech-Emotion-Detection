import numpy as np
import librosa
import joblib
import tensorflow as tf
import glob
import os
from tensorflow.keras.models import load_model

# --- Re-define necessary parameters from the training script ---
SAMPLING_RATE = 48000
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
MAX_PAD_LEN = 250

# --- Load the saved components ---
print("Loading model and pre-processing objects...")

# Find the most recent trained model
model_pattern = 'models/ser_model_*_val_acc_*.keras'
saved_models = glob.glob(model_pattern)

if saved_models:
    # Get the most recent model file
    model_path = max(saved_models, key=os.path.getctime)
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
else:
    # Fallback to the simple model name if no timestamped model found
    model_path = 'models/ser_bilstm_model.keras'
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)
    else:
        raise FileNotFoundError("No trained model found. Please run ser_model.py first to train a model.")

# Load preprocessing objects
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
print("✅ All components loaded successfully.")

def extract_features(audio_path):
    """Extracts MFCC features from an audio file."""
    try:
        audio, sr = librosa.load(audio_path, sr=SAMPLING_RATE, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        return mfccs.T
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def predict_emotion(audio_path, model, scaler, label_encoder):
    """Predicts the emotion of a single audio file."""
    # 1. Extract features
    features = extract_features(audio_path)
    if features is None:
        return "Error processing audio."

    # 2. Pad/truncate features
    padded_features = tf.keras.preprocessing.sequence.pad_sequences(
        [features], maxlen=MAX_PAD_LEN, dtype='float32', padding='post', truncating='post'
    )
    
    # 3. Scale the features
    scaled_features = scaler.transform(padded_features.reshape(-1, padded_features.shape[2]))
    scaled_features = scaled_features.reshape(padded_features.shape)

    # 4. Make prediction
    prediction_probs = model.predict(scaled_features)
    predicted_label_index = np.argmax(prediction_probs, axis=1)[0]

    # 5. Decode the label to an emotion string
    predicted_emotion = label_encoder.inverse_transform([predicted_label_index])[0]
    
    return predicted_emotion

# --- Main execution ---
if __name__ == '__main__':
    # 🎤 Ask user for the audio file path
    print("🎵 Emotion Recognition System")
    print("=" * 40)
    print("Please provide the path to your audio file.")
    print("Examples:")
    print("  - ./ravdess/Actor_01/03-02-01-01-01-01-01.wav")
    print("  - C:\\path\\to\\your\\audio\\file.wav")
    print("  - /path/to/your/audio/file.wav")
    print("")
    
    AUDIO_FILE_PATH = input("Enter audio file path: ").strip()
    
    # Remove quotes if user wrapped the path in quotes
    if AUDIO_FILE_PATH.startswith('"') and AUDIO_FILE_PATH.endswith('"'):
        AUDIO_FILE_PATH = AUDIO_FILE_PATH[1:-1]
    if AUDIO_FILE_PATH.startswith("'") and AUDIO_FILE_PATH.endswith("'"):
        AUDIO_FILE_PATH = AUDIO_FILE_PATH[1:-1]
    
    # Check if file exists
    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"❌ Audio file not found: {AUDIO_FILE_PATH}")
        print("Please make sure the file path is correct and the file exists.")
        print("Supported formats: .wav files")
        exit(1)
    
    print(f"\n🔍 Processing audio file: {AUDIO_FILE_PATH}")
    
    # Predict the emotion
    emotion = predict_emotion(AUDIO_FILE_PATH, model, scaler, label_encoder)
    
    print(f"\n🎵 Audio File: {AUDIO_FILE_PATH}")
    print(f"🎭 Predicted Emotion: {emotion}")