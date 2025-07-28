# Test prediction script
import joblib
from predict import predict_emotion
from config import MODEL_PATH

def test_single_prediction():
    """Test the prediction on a single audio file"""
    
    # Load the model and scaler
    try:
        model = joblib.load(f"{MODEL_PATH}ser_model.pkl")
        scaler = joblib.load(f"{MODEL_PATH}ser_scaler.pkl")
        print("Model and scaler loaded successfully.")
    except FileNotFoundError:
        print("ERROR: Model or scaler not found.")
        return
    
    # Test with a sample audio file
    test_file = "RAVDESS_data/Actor_01/03-02-01-01-01-01-01.wav"
    print(f"\nTesting prediction on: {test_file}")
    
    result = predict_emotion(test_file, model, scaler)
    print("\nPrediction Result:")
    print(result)
    
    # Test with another file
    test_file2 = "RAVDESS_data/Actor_01/03-02-02-01-01-01-01.wav"
    print(f"\nTesting prediction on: {test_file2}")
    
    result2 = predict_emotion(test_file2, model, scaler)
    print("\nPrediction Result:")
    print(result2)

if __name__ == "__main__":
    test_single_prediction()
