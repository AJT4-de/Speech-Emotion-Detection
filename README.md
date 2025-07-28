# Speech Emotion Detection Project

A comprehensive machine learning project implementing multiple approaches for Speech Emotion Recognition (SER) using different deep learning and machine learning techniques.

## 🎯 Project Overview

This project explores three different approaches to speech emotion recognition:
- **CNN with Mel Spectrograms**: Convolutional Neural Network using visual representation of audio
- **Random Forest**: Traditional machine learning with hand-crafted audio features
- **RNN-LSTM**: Recurrent Neural Network for sequential audio pattern recognition

## 📊 Datasets

The project utilizes two primary datasets:
- **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**: Professional actors expressing emotions
- **Emotion Speech Dataset (ESD)**: Additional emotional speech samples including Chinese language data

### Supported Emotions
- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust (RAVDESS only)
- Surprise

## 🏗️ Project Structure

```
Speech Emotion Detection/
├── Speech Emotion Checker_CNN-Mel Spectogram/
│   ├── config.py                 # Configuration parameters
│   ├── data_preprocessing.py     # Audio preprocessing pipeline
│   ├── model.py                  # CNN model architecture
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Model evaluation
│   ├── requirements.txt          # Dependencies
│   ├── datasets/                 # Dataset storage
│   └── models/                   # Trained models and results
│
├── Speech Emotion Checker_Random Forest/
│   ├── Speech Emotion Checker-Random Forest(All including chinese)/
│   │   ├── config.py             # Configuration
│   │   ├── data_loader.py        # Data loading utilities
│   │   ├── feature_extractor.py  # Audio feature extraction
│   │   ├── model_trainer.py      # Random Forest training
│   │   ├── predict.py            # Prediction interface
│   │   ├── validate_model.py     # Model validation
│   │   └── requirements.txt      # Dependencies
│   │
│   ├── Speech Emotion Checker-Random Forest(Experimental)/
│   │   └── [Advanced experimental features and optimizations]
│   │
│   └── Speech Emotion Checker-Random Forest(Only RAVDESS)/
│       └── [RAVDESS-only implementation]
│
└── Speech Emotion Checker_RNN-LSTM/
    ├── ser_model.py              # LSTM model implementation
    ├── predict.py                # Prediction script
    ├── create_preprocessing_objects.py  # Preprocessing setup
    ├── requirements.txt          # Dependencies
    └── models/                   # Trained models
```

## 🚀 Quick Start

### Prerequisites

Ensure you have Python 3.8+ installed on your system.

### Setup Instructions

1. **Clone the repository** (or navigate to the project directory)
   ```bash
   cd "Speech Emotion Detection"
   ```

2. **Choose your preferred approach** and navigate to the corresponding directory:

   **For CNN with Mel Spectrograms:**
   ```bash
   cd "Speech Emotion Checker_CNN-Mel Spectogram"
   pip install -r requirements.txt
   ```

   **For Random Forest:**
   ```bash
   cd "Speech Emotion Checker_Random Forest/Speech Emotion Checker-Random Forest(All including chinese)"
   pip install -r requirements.txt
   ```

   **For RNN-LSTM:**
   ```bash
   cd "Speech Emotion Checker_RNN-LSTM"
   pip install -r requirements.txt
   ```

3. **Prepare datasets**: Place your audio files in the appropriate dataset directories

4. **Train the model**:
   - CNN: `python train.py`
   - Random Forest: `python main.py`
   - RNN-LSTM: `python ser_model.py`

5. **Make predictions**:
   - CNN: `python evaluate.py`
   - Random Forest: `python predict.py`
   - RNN-LSTM: `python predict.py`

## 🔬 Approaches Comparison

### 1. CNN with Mel Spectrograms
- **Technology**: TensorFlow/Keras
- **Input**: Mel-spectrogram images (128 mel bands)
- **Architecture**: Convolutional layers for feature extraction
- **Strengths**: Good for capturing spectral patterns
- **Best for**: Visual representation of audio features

### 2. Random Forest
- **Technology**: Scikit-learn
- **Input**: Hand-crafted audio features (196 features)
- **Features**: MFCC, spectral features, rhythm features
- **Strengths**: Interpretable, fast training
- **Best for**: Traditional ML approach with feature engineering

### 3. RNN-LSTM
- **Technology**: TensorFlow/Keras
- **Input**: Sequential audio features
- **Architecture**: LSTM layers for temporal modeling
- **Strengths**: Captures temporal dependencies
- **Best for**: Sequential pattern recognition

## 📈 Performance Results

### Random Forest (All Datasets)
- **Overall Test Accuracy**: 72%
- **Total Samples**: 3,938 audio files
- **Best Performance**: Calm emotions (100% accuracy)
- **Challenge**: Neutral emotion recognition

### CNN Mel-Spectrogram
- **Training Approach**: Image-based CNN
- **Input Size**: 128x130 mel-spectrogram images
- **Architecture**: Multi-layer CNN with pooling

### RNN-LSTM
- **Sequential Processing**: Time-series audio analysis
- **Memory**: Long Short-Term Memory for temporal patterns

## 🛠️ Key Features

- **Multi-dataset Support**: Works with RAVDESS and ESD datasets
- **Multiple Architectures**: Compare different ML/DL approaches
- **Preprocessing Pipeline**: Comprehensive audio preprocessing
- **Feature Engineering**: Extensive audio feature extraction
- **Model Evaluation**: Detailed performance analysis
- **Prediction Interface**: Easy-to-use prediction scripts

## 📋 Dependencies

### Common Requirements
- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Librosa (audio processing)
- Matplotlib (visualization)
- SoundFile

### Deep Learning Specific
- TensorFlow 2.15.0
- Keras 2.15.0

### Machine Learning Specific
- Joblib
- Seaborn
- Tqdm

## 📁 Dataset Structure

Expected dataset structure:
```
datasets/
├── RAVDESS/
│   └── Actor_XX/
│       └── 03-01-XX-XX-XX-XX-XX.wav
└── Emotion Speech Dataset/
    └── [emotion files with appropriate naming]
```

## 🔧 Configuration

Each approach has its own configuration file:
- CNN: `config.py` - Audio parameters, model settings
- Random Forest: `config.py` - Feature extraction, emotion mapping
- RNN-LSTM: Model parameters embedded in scripts

## 📊 Usage Examples

### Training a Model
```bash
# CNN approach
cd "Speech Emotion Checker_CNN-Mel Spectogram"
python train.py

# Random Forest approach
cd "Speech Emotion Checker_Random Forest/Speech Emotion Checker-Random Forest(All including chinese)"
python main.py
```

### Making Predictions
```bash
# Interactive prediction
python predict.py

# Batch validation
python validate_model.py
```

## 🚧 Future Improvements

1. **Data Augmentation**: Audio augmentation techniques
2. **Ensemble Methods**: Combine multiple approaches
3. **Real-time Processing**: Live audio emotion detection
4. **Cross-lingual Support**: Better multi-language recognition
5. **Deployment**: Web interface or API development

## 🤝 Contributing

This project is part of a Masters in Applied Artificial Intelligence program. Contributions and suggestions are welcome for educational purposes.

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- RAVDESS dataset creators
- Emotion Speech Dataset contributors
- TensorFlow and Scikit-learn communities

---

**Note**: This project demonstrates different approaches to speech emotion recognition. Each method has its own strengths and is suitable for different use cases. The Random Forest approach provides good interpretability, while the deep learning approaches (CNN and RNN-LSTM) may capture more complex patterns in the audio data.
# Speech-Emotion-Detection
