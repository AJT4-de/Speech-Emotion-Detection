# model.py

"""
Defines the CNN architecture for the Speech Emotion Recognition model.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from config import LEARNING_RATE

def build_cnn_model(input_shape, num_classes):
    """
    Builds and compiles the CNN model for emotion recognition.

    Args:
        input_shape (tuple): The shape of the input data (n_mels, num_frames, 1).
        num_classes (int): The number of emotion classes for the output layer.

    Returns:
        keras.Model: The compiled CNN model.
    """
    model = Sequential([
        Input(shape=input_shape),

        # Layer 1
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Layer 2
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Layer 3
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        # Flatten the features to feed into the Dense layers
        Flatten(),

        # Dense Layers
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.5),

        # Output Layer
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("CNN Model built successfully.")
    model.summary()
    return model