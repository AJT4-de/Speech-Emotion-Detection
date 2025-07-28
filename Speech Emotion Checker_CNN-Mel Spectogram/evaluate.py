# evaluate.py

"""
Evaluates the trained SER model on the test set.
Loads the best model and computes metrics like accuracy, classification report,
and confusion matrix.
"""

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from config import MODEL_SAVE_DIR, MODEL_NAME_PREFIX, EMOTIONS

def evaluate_model():
    """Loads the trained model and evaluates it on the test set."""
    print("--- Starting Model Evaluation ---")
    
    # 1. Load the test data and label encoder
    try:
        X_test = np.load(os.path.join(MODEL_SAVE_DIR, 'X_test.npy'))
        y_test = np.load(os.path.join(MODEL_SAVE_DIR, 'y_test.npy'))
        encoder_classes = np.load(os.path.join(MODEL_SAVE_DIR, 'label_encoder.npy'), allow_pickle=True)
    except FileNotFoundError:
        print("Error: Test data not found. Please run train.py first to generate test data.")
        return

    # 2. Load the best saved model
    model_path = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME_PREFIX}_best_model.keras')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train the model first.")
        return
        
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")

    # 3. Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {loss:.4f}")

    # 4. Generate predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 5. Generate and print classification report
    print("\nClassification Report:")
    # Ensure target names are correctly ordered based on the encoder
    target_names = [EMOTIONS[label] for label in encoder_classes] if isinstance(encoder_classes[0], str) and encoder_classes[0].isdigit() else encoder_classes
    
    # Use integer labels if they are not mapped back to names
    try:
        report = classification_report(y_test, y_pred, target_names=target_names)
    except ValueError:
        # Fallback if target_names do not match labels
        report = classification_report(y_test, y_pred)
        
    print(report)

    # 6. Generate and plot confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot
    cm_path = os.path.join(MODEL_SAVE_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.show()
    print(f"Confusion matrix saved to {cm_path}")

if __name__ == '__main__':
    evaluate_model()