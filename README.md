# Speech Emotion Detection Project

A comprehensive machine learning project implementing multiple approaches for Speech Emotion Recognition (SER) using different deep learning and machine learning techniques.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)](https://tensorflow.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4.2-green)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-Educational-purple)](#)

## 🎯 Project Overview

This project explores three different state-of-the-art approaches to speech emotion recognition, each leveraging different aspects of audio signal processing and machine learning:

- **🖼️ CNN with Mel Spectrograms**: Convolutional Neural Network treating audio as visual spectrograms
- **🌳 Random Forest**: Traditional machine learning with comprehensive hand-crafted audio features
- **🔄 RNN-LSTM**: Recurrent Neural Network capturing temporal dependencies in audio sequences

### 🎓 Academic Context
This project is part of a **Masters in Applied Artificial Intelligence** program, demonstrating practical implementation of various AI/ML techniques for real-world audio analysis challenges.

## 📊 Datasets

The project utilizes two high-quality emotional speech datasets:

### Primary Datasets
- **🎭 RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**
  - Professional actors expressing 8 different emotions
  - High-quality recordings with consistent acoustic conditions
  - 1,440 audio files from 24 professional actors
  - Balanced gender representation (12 male, 12 female)

- **🌏 Emotion Speech Dataset (ESD)**
  - Multi-language emotional speech samples
  - Includes Chinese language data for cross-linguistic analysis
  - Additional emotional expressions for robust training

### Supported Emotions
| Emotion | RAVDESS Code | ESD Code | Description |
|---------|--------------|----------|-------------|
| 😐 Neutral | 01 | n | Baseline emotional state |
| 😌 Calm | 02 | - | Relaxed, peaceful |
| 😊 Happy | 03 | h | Joyful, pleased |
| 😢 Sad | 04 | s | Sorrowful, melancholy |
| 😠 Angry | 05 | a | Frustrated, irritated |
| 😨 Fearful | 06 | - | Afraid, anxious |
| 🤢 Disgust | 07 | - | Repulsed, disgusted (RAVDESS only) |
| 😲 Surprise | 08 | u | Astonished, amazed |

**Total Emotions Supported**: 7-8 emotions (depending on dataset)

## 🏗️ Project Structure

```
Speech Emotion Detection/
├── 📄 README.md                  # This comprehensive guide
├── 🚫 .gitignore                 # Git ignore patterns for datasets/models
├── 
├── 🖼️ Speech Emotion Checker_CNN-Mel Spectogram/
│   ├── config.py                 # Configuration parameters
│   ├── data_preprocessing.py     # Audio preprocessing pipeline
│   ├── model.py                  # CNN model architecture
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Model evaluation
│   ├── requirements.txt          # Dependencies
│   ├── 📊 datasets/              # Dataset storage (ignored by git)
│   │   ├── RAVDESS/
│   │   └── Emotion Speech Dataset/
│   └── 🤖 models/                # Trained models and results
│       ├── ser_cnn_best_model.keras
│       ├── label_encoder.npy
│       ├── confusion_matrix.png
│       └── training_history.png
│
├── 🌳 Speech Emotion Checker_Random Forest/
│   ├── 🌐 Speech Emotion Checker-Random Forest(All including chinese)/
│   │   ├── config.py             # Configuration
│   │   ├── data_loader.py        # Data loading utilities
│   │   ├── feature_extractor.py  # Audio feature extraction (196 features)
│   │   ├── model_trainer.py      # Random Forest training
│   │   ├── predict.py            # Interactive prediction interface
│   │   ├── validate_model.py     # Comprehensive model validation
│   │   ├── test_prediction.py    # Simple testing script
│   │   ├── requirements.txt      # Dependencies
│   │   ├── README_Results.md     # Detailed performance analysis
│   │   ├── 📊 RAVDESS_data/      # RAVDESS dataset (ignored by git)
│   │   ├── 📊 Emotion Speech Dataset/ # ESD dataset (ignored by git)
│   │   └── 🤖 models/            # Trained models
│   │       ├── ser_model.pkl     # Random Forest model
│   │       └── ser_scaler.pkl    # Feature scaler
│   │
│   ├── 🧪 Speech Emotion Checker-Random Forest(Experimental)/
│   │   ├── advanced_*.py         # Advanced implementations
│   │   ├── optimized_*.py        # Performance optimizations
│   │   ├── analyze_dataset.py    # Dataset analysis tools
│   │   └── [Additional experimental features]
│   │
│   └── 🎯 Speech Emotion Checker-Random Forest(Only RAVDESS)/
│       └── [RAVDESS-only implementation for comparison]
│
└── 🔄 Speech Emotion Checker_RNN-LSTM/
    ├── ser_model.py              # LSTM model implementation
    ├── predict.py                # Prediction script
    ├── create_preprocessing_objects.py  # Preprocessing setup
    ├── requirements.txt          # Dependencies
    ├── features_cache.pkl        # Cached features (ignored by git)
    ├── 📊 ravdess/               # Dataset (ignored by git)
    └── 🤖 models/                # Trained models (ignored by git)
```

### 📁 Key Directories
- **📊 Dataset folders**: Automatically ignored by Git (see `.gitignore`)
- **🤖 Model folders**: Contain trained models and artifacts
- **📄 Config files**: Centralized configuration for each approach
- **🧪 Experimental**: Advanced features and optimization attempts

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** 
- **Git** (for cloning and version control)
- **Audio drivers** (for audio processing)
- **CUDA** (optional, for GPU acceleration with deep learning models)

### 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AJT4-de/Speech-Emotion-Detection.git
   cd Speech-Emotion-Detection
   ```

2. **Choose your preferred approach** and navigate to the corresponding directory:

   **🖼️ For CNN with Mel Spectrograms:**
   ```bash
   cd "Speech Emotion Checker_CNN-Mel Spectogram"
   pip install -r requirements.txt
   ```

   **🌳 For Random Forest (Recommended for beginners):**
   ```bash
   cd "Speech Emotion Checker_Random Forest/Speech Emotion Checker-Random Forest(All including chinese)"
   pip install -r requirements.txt
   ```

   **🔄 For RNN-LSTM:**
   ```bash
   cd "Speech Emotion Checker_RNN-LSTM"
   pip install -r requirements.txt
   ```

3. **📊 Prepare datasets**: 
   - Download RAVDESS and ESD datasets
   - Place audio files in the appropriate dataset directories
   - Follow the expected folder structure (see Dataset Structure section)

4. **🎯 Train the model**:
   ```bash
   # CNN approach
   python train.py
   
   # Random Forest approach  
   python main.py
   
   # RNN-LSTM approach
   python ser_model.py
   ```

5. **🔮 Make predictions**:
   ```bash
   # Interactive prediction interface
   python predict.py
   
   # Batch evaluation
   python evaluate.py  # (CNN)
   python validate_model.py  # (Random Forest)
   ```

4. **Train the model**:
   - CNN: `python train.py`
   - Random Forest: `python main.py`
   - RNN-LSTM: `python ser_model.py`

5. **Make predictions**:
   - CNN: `python evaluate.py`
   - Random Forest: `python predict.py`
   - RNN-LSTM: `python predict.py`

## 🔬 Approaches Comparison

| Feature | 🖼️ CNN Mel-Spectrogram | 🌳 Random Forest | 🔄 RNN-LSTM |
|---------|-------------------------|------------------|--------------|
| **Technology** | TensorFlow/Keras | Scikit-learn | TensorFlow/Keras |
| **Input Type** | 2D Mel-spectrogram images | 196 hand-crafted features | Sequential audio features |
| **Architecture** | Convolutional layers + pooling | Ensemble of decision trees | LSTM layers + dense |
| **Training Time** | Medium-Long (GPU recommended) | Fast | Medium-Long |
| **Interpretability** | Low | High | Low |
| **Memory Usage** | High | Low | Medium |
| **Real-time Capability** | Medium | High | Medium |

### 1. 🖼️ CNN with Mel Spectrograms
- **Strengths**: Excellent for capturing spectral patterns, leverages computer vision techniques
- **Input Processing**: Converts audio to 128-band mel-spectrograms (128x130 images)
- **Best For**: Complex pattern recognition, when you have sufficient training data
- **Innovation**: Treats audio analysis as an image classification problem

### 2. 🌳 Random Forest
- **Strengths**: Fast training, interpretable results, robust to overfitting
- **Feature Set**: 196 comprehensive audio features (MFCC, spectral, temporal, rhythm)
- **Best For**: Quick prototyping, when interpretability is crucial, limited computational resources
- **Reliability**: Most consistent performer across different datasets

### 3. 🔄 RNN-LSTM
- **Strengths**: Captures temporal dependencies, natural fit for sequential audio data
- **Memory Mechanism**: Long Short-Term Memory for learning long-range dependencies
- **Best For**: When temporal patterns are crucial, speech sequence modeling
- **Advanced**: Handles variable-length sequences naturally

## 📈 Performance Results

### 🏆 Model Performance Summary

| Model | Overall Accuracy | Training Time | Best Emotion | Challenging Emotion |
|-------|------------------|---------------|--------------|-------------------|
| 🌳 **Random Forest** | **92%** | ⚡ Fast | Calm (100%) | Neutral |
| 🔄 **RNN-LSTM** | **89%** | 🕐 Medium | Calm (94%) | Anger |
| 🖼️ **CNN Mel-Spec** | **87%** | 🕐 Medium | Happy(92%)| Neutral |

### 🌳 Random Forest (All Datasets) - *Recommended*
- **✅ Overall Test Accuracy**: 92%
- **📊 Total Samples**: 3,938 audio files
- **🎯 Feature Count**: 196 hand-crafted audio features
- **🏅 Best Performance**: Calm emotions (100% accuracy)
- **⚠️ Challenge**: Neutral emotion recognition (often misclassified as fearful)
- **💡 Strengths**: Fast inference, interpretable results, robust performance

### 🖼️ CNN Mel-Spectrogram
- **✅ Overall Test Accuracy**: 87%
- **🖼️ Input Processing**: 128x130 mel-spectrogram images
- **🏗️ Architecture**: Multi-layer CNN with pooling layers
- **💻 Requirements**: GPU recommended for training
- **💡 Strengths**: Good pattern recognition, end-to-end learning

### 🔄 RNN-LSTM
- **✅ Overall Test Accuracy**: 89%
- **⏰ Sequential Processing**: Time-series audio analysis
- **🧠 Memory**: Long Short-Term Memory for temporal patterns
- **🔗 Architecture**: LSTM + Dense layers
- **💡 Strengths**: Temporal dependency modeling, sequence learning

### 📊 Detailed Performance Analysis

**Random Forest Validation Results** (RAVDESS Subset):
- **Files Tested**: 15 samples from 3 actors
- **Breakdown by Emotion**:
  - ✅ Calm: 100% accuracy (3/3 correct)
  - ❌ Neutral: 0% accuracy (consistently misclassified as fearful)

**Key Insights**:
- **Best Overall**: Random Forest shows highest accuracy and fastest training
- **Most Reliable**: Random Forest with consistent cross-dataset performance  
- **Most Innovative**: CNN treating audio as visual problem
- **Best for Sequences**: RNN-LSTM for temporal pattern analysis

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

The project expects datasets to be organized as follows:

```
datasets/
├── RAVDESS/
│   └── Actor_01/ to Actor_24/
│       └── 03-01-[emotion]-[intensity]-[statement]-[repetition]-[actor].wav
│           # Example: 03-01-06-01-02-01-12.wav
│           # Format: [Modality]-[Channel]-[Emotion]-[Intensity]-[Statement]-[Rep]-[Actor]
│           # Emotion codes: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
│
└── Emotion Speech Dataset/
    ├── [Chinese emotional speech files]
    └── [Additional language variants]
```

### 🎵 Audio File Naming Convention (RAVDESS)
- **Modality**: 03 (audio-only)
- **Channel**: 01 (speech)
- **Emotion**: 01-08 (see emotion mapping above)
- **Intensity**: 01 (normal), 02 (strong)
- **Statement**: 01 ("Kids are talking by the door"), 02 ("Dogs are sitting by the door")
- **Repetition**: 01 (1st repetition), 02 (2nd repetition)  
- **Actor**: 01-24 (actor identifier)

### 📂 Important Notes
- Datasets are automatically excluded from Git commits (see `.gitignore`)
- Download datasets separately and place in appropriate folders
- Ensure proper folder structure for automated data loading

## � Technical Documentation

### �🔧 Configuration Files
Each approach includes detailed configuration:
- **CNN**: `config.py` - Audio parameters (sample rate, mel bands, training epochs)
- **Random Forest**: `config.py` - Feature extraction settings, emotion mappings
- **RNN-LSTM**: Embedded parameters in model scripts

### 🎵 Audio Processing Pipeline
1. **Audio Loading**: Librosa-based audio file loading
2. **Preprocessing**: Normalization, resampling to 22kHz
3. **Feature Extraction**: 
   - CNN: Mel-spectrogram generation (128 bands)
   - Random Forest: 196 traditional features (MFCC, spectral, temporal)
   - RNN-LSTM: Sequential feature vectors
4. **Data Augmentation**: Optional noise addition and temporal shifts

### 🧠 Model Architectures

**CNN Architecture:**
```
Input (128x130x1) → Conv2D → MaxPool → Conv2D → MaxPool → 
Flatten → Dense(128) → Dropout → Dense(7) → Softmax
```

**Random Forest Configuration:**
```
n_estimators=100, max_depth=None, min_samples_split=2
Feature scaling: StandardScaler
196 features: 13 MFCC + spectral + temporal + rhythm features
```

**RNN-LSTM Structure:**
```
Input → LSTM(64) → Dropout → LSTM(32) → Dense(64) → 
Dropout → Dense(7) → Softmax
```

## 🔍 Troubleshooting

### Common Issues & Solutions

**Dataset Loading Problems:**
```bash
# Ensure proper folder structure
ls datasets/RAVDESS/Actor_01/  # Should show .wav files
```

**Memory Issues (CNN/RNN):**
```python
# Reduce batch size in config.py
BATCH_SIZE = 16  # Instead of 32
```

**Audio Processing Errors:**
```bash
# Install additional audio codecs
pip install soundfile librosa[display]
```

**Model Loading Failures:**
```python
# Check model file existence
import os
print(os.path.exists('models/ser_model.pkl'))
```

### Performance Optimization Tips
- **GPU Usage**: Enable CUDA for TensorFlow models
- **Feature Caching**: Use cached features for faster training
- **Parallel Processing**: Utilize multiprocessing for feature extraction

## 📊 Usage Examples

### 🎯 Training Models

**Random Forest (Recommended for beginners):**
```bash
cd "Speech Emotion Checker_Random Forest/Speech Emotion Checker-Random Forest(All including chinese)"
python main.py
# Output: Trained model saved to models/ser_model.pkl
```

**CNN with Mel-Spectrograms:**
```bash
cd "Speech Emotion Checker_CNN-Mel Spectogram"
python train.py
# Output: Model checkpoints and training visualizations
```

**RNN-LSTM:**
```bash
cd "Speech Emotion Checker_RNN-LSTM"
python ser_model.py
# Output: Sequential model with temporal learning
```

### 🔮 Making Predictions

**Interactive Prediction Interface:**
```bash
python predict.py
# Follow prompts to input audio file path
# Get real-time emotion prediction with confidence scores
```

**Batch Validation:**
```bash
# Random Forest comprehensive validation
python validate_model.py

# CNN model evaluation  
python evaluate.py

# Test on specific files
python test_prediction.py
```

### 🧪 Advanced Usage

**Experimental Features (Random Forest):**
```bash
cd "Speech Emotion Checker_Random Forest/Speech Emotion Checker-Random Forest(Experimental)"
python advanced_main.py              # Advanced feature extraction
python analyze_dataset.py            # Dataset analysis and visualization
python optimized_main.py             # Performance-optimized version
```

**Dataset Analysis:**
```bash
python check_emotions.py             # Verify emotion distribution
python debug_features.py             # Feature extraction debugging
```

## 🚧 Future Improvements & Research Directions

### 🔬 Immediate Enhancements
1. **🎚️ Data Augmentation**: 
   - Audio augmentation (noise injection, pitch shifting, time stretching)
   - Synthetic sample generation for data balancing
   
2. **🤖 Model Optimization**:
   - Hyperparameter tuning with Grid/Random Search
   - Ensemble methods combining all three approaches
   - Cross-validation for robust evaluation

3. **🎯 Feature Engineering**:
   - Additional audio features (spectral contrast, harmony, percussive)
   - Deep feature extraction using pre-trained models
   - Multi-scale temporal features

### 🌟 Advanced Research Directions
4. **🌐 Cross-Lingual Analysis**:
   - Language-independent emotion recognition
   - Transfer learning across languages
   - Cultural emotion expression analysis

5. **⚡ Real-Time Processing**:
   - Streaming audio emotion detection
   - Mobile deployment optimization
   - Edge computing implementation

6. **🎭 Multi-Modal Fusion**:
   - Combine audio with facial expression analysis
   - Text sentiment + speech emotion correlation
   - Physiological signal integration

### 🏭 Production Deployment
7. **🌐 Web Interface Development**:
   - Flask/Django web application
   - RESTful API for emotion detection
   - Real-time dashboard with visualizations

8. **📱 Mobile Application**:
   - React Native/Flutter implementation
   - On-device inference optimization
   - Privacy-preserving local processing

### 📚 Academic Extensions
9. **🔍 Interpretability Research**:
   - LIME/SHAP analysis for deep learning models
   - Feature importance visualization
   - Decision boundary analysis

10. **📊 Benchmark Comparisons**:
    - Comparison with state-of-the-art models
    - Cross-dataset evaluation protocols
    - Standardized evaluation metrics

## 🤝 Contributing

This project is part of a **Masters in Applied Artificial Intelligence** program. We welcome contributions from:

### 👥 How to Contribute
- **🐛 Bug Reports**: Open issues for any bugs found
- **💡 Feature Requests**: Suggest improvements or new features  
- **📖 Documentation**: Help improve documentation and examples
- **🔬 Research**: Share experimental results and findings
- **📊 Datasets**: Contribute additional emotional speech datasets

### 🔄 Development Workflow
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### 📋 Contribution Guidelines
- Follow existing code style and structure
- Include comprehensive comments and documentation
- Test your changes across different approaches
- Update README if adding new features

## � License

This project is developed for **educational and research purposes** as part of academic coursework.

### Usage Rights
- ✅ **Educational Use**: Free for academic and learning purposes
- ✅ **Research**: Permitted for academic research and publications
- ✅ **Modification**: Feel free to adapt and improve
- ⚠️ **Commercial Use**: Please contact for commercial licensing

## 🙏 Acknowledgments

### 📚 Datasets
- **RAVDESS**: Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
- **Emotion Speech Dataset**: Contributors to the multi-language emotional speech corpus

### 🛠️ Technology Stack
- **TensorFlow/Keras Team**: Deep learning framework
- **Scikit-learn Community**: Machine learning library  
- **Librosa Developers**: Audio analysis toolkit
- **Python Community**: Programming language and ecosystem

### 🎓 Academic Context
- **University**: Masters in Applied Artificial Intelligence Program
- **Course**: Individual Project in Speech Emotion Recognition
- **Focus Area**: Multi-approach comparison in AI/ML techniques

### 🌟 Special Thanks
- Academic supervisors and mentors
- Open-source community for tools and libraries
- Researchers advancing emotion recognition field
- Fellow students providing feedback and collaboration

---

## 📞 Contact & Support

For questions, suggestions, or collaboration opportunities:
- **GitHub Issues**: For technical problems and feature requests
- **Academic Inquiries**: Related to research methodology and findings
- **Collaboration**: Open to academic partnerships and improvements

---

**⭐ If this project helps your research or learning, please consider giving it a star!**

---

*This project demonstrates the application of multiple AI/ML approaches to a real-world problem, showcasing the strengths and trade-offs of different techniques in speech emotion recognition.*
