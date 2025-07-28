# File: model_trainer.py

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from config import EMOTION_LABELS

def train_model(X_train, y_train):
    """
    Trains a Random Forest model on the provided training data.
    It first scales the features using a StandardScaler.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training labels.

    Returns:
        tuple: A tuple containing the trained model and the fitted scaler.
    """
    # Initialize and fit the scaler on the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Initialize the model
    # n_estimators is the number of trees in the forest.
    # random_state ensures reproducibility.
    # For multi-class classification, we use more trees and better parameters
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle potential class imbalance
    )
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    print("Model training complete.")
    return model, scaler

def evaluate_model(model, scaler, X_test, y_test):
    """
    Evaluates the trained model on the test set.

    Args:
        model: The trained classifier.
        scaler: The fitted scaler from the training phase.
        X_test (pd.DataFrame or np.ndarray): Test features.
        y_test (pd.Series or np.ndarray): Test labels.

    Returns:
        float: The accuracy score of the model on the test data.
    """
    # Scale the test data using the *same* scaler
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=EMOTION_LABELS)
    
    print(f"Model Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(report)
    
    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=EMOTION_LABELS, 
                yticklabels=EMOTION_LABELS)
    plt.title('Confusion Matrix - Multi-Emotion Classification')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return accuracy