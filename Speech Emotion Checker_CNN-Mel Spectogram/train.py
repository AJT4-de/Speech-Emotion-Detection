# train.py

"""
Main script for training the Speech Emotion Recognition model.
Handles data loading, splitting, model building/loading, training, and saving.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from config import (
    EPOCHS, BATCH_SIZE, MODEL_SAVE_DIR, MODEL_NAME_PREFIX,
    PRETRAINED_MODEL_PATH, EMOTIONS
)
from data_preprocessing import load_and_extract_features
from model import build_cnn_model

def check_gpu():
    """Check for GPU availability and configure memory growth."""
    print("🔍 Checking GPU availability...")
    
    # Check if TensorFlow was built with CUDA support
    if not tf.test.is_built_with_cuda():
        print("⚠️ TensorFlow was not built with CUDA support. GPU acceleration not available.")
        return
    
    # Get list of available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU(s) available: {len(gpus)}. Memory growth enabled.")
            print(f"   GPU device(s): {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(f"❌ Error setting up GPU: {e}")
    else:
        print("⚠️ No GPU detected. This could be due to:")
        print("   • Missing CUDA/cuDNN libraries")
        print("   • Incompatible GPU compute capability")
        print("   • GPU drivers not installed")
        print("   • No GPU hardware present")
        print("")
        print("💡 For your NVIDIA GeForce 940MX, you would need:")
        print("   • CUDA 11.2")
        print("   • cuDNN 8.1")
        print("")
        print("🚀 The model will train on CPU. This is fine for this dataset size!")
        print("   CPU training will be slower but still functional.")
        print("")
        
        # Show CPU information
        import psutil
        cpu_count = psutil.cpu_count()
        print(f"🖥️ Using CPU with {cpu_count} cores for training.")

def train_model():
    """Main function to orchestrate the model training process."""
    check_gpu()

    # 1. Load and prepare data
    X, y, label_encoder = load_and_extract_features()
    num_classes = len(np.unique(y))

    # 2. Split data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # 3. Normalize the data
    # Fit scaler only on the training data to prevent data leakage
    scaler = StandardScaler()
    h, w, c = X_train.shape[1], X_train.shape[2], X_train.shape[3]
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, h * w * c)).reshape(-1, h, w, c)
    X_val_scaled = scaler.transform(X_val.reshape(-1, h * w * c)).reshape(-1, h, w, c)
    X_test_scaled = scaler.transform(X_test.reshape(-1, h * w * c)).reshape(-1, h, w, c)
    
    # Save the test set for final evaluation
    np.save(os.path.join(MODEL_SAVE_DIR, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(MODEL_SAVE_DIR, 'y_test.npy'), y_test)
    np.save(os.path.join(MODEL_SAVE_DIR, 'label_encoder.npy'), label_encoder.classes_)
    print("Test data saved for evaluation.")
    
    # 4. Build or load model
    if PRETRAINED_MODEL_PATH and os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"Resuming training from model: {PRETRAINED_MODEL_PATH}")
        model = load_model(PRETRAINED_MODEL_PATH)
    else:
        print("Building a new model from scratch.")
        input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2], 1)
        model = build_cnn_model(input_shape, num_classes)

    # 5. Define callbacks for training
    model_checkpoint_path = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME_PREFIX}_best_model.keras')
    
    callbacks = [
        # Stop training if validation loss doesn't improve for 15 epochs
        EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True),
        # Save the best model based on validation accuracy
        ModelCheckpoint(filepath=model_checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        # Reduce learning rate if validation loss plateaus for 5 epochs
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-6)
    ]

    # 6. Train the model
    print("\n--- Starting Model Training ---")
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_scaled, y_val),
        callbacks=callbacks
    )
    print("--- Model Training Finished ---\n")

    # 7. Plot and save training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    
    history_plot_path = os.path.join(MODEL_SAVE_DIR, 'training_history.png')
    plt.savefig(history_plot_path)
    plt.show()
    print(f"Training history plot saved to {history_plot_path}")

if __name__ == '__main__':
    train_model()