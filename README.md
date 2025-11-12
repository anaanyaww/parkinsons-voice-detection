#  Parkinson's Disease Voice Detection

AI-powered early detection system for Parkinson's disease using voice analysis and machine learning.

![Demo](demo_screenshot.png)

## Overview

This application uses advanced machine learning (XGBoost) to analyze voice recordings and detect early indicators of Parkinson's disease. The system achieves **93.85% cross-validation accuracy** and is designed as a non-invasive screening tool.

## Features

- **Multiple Voice Tests**: Sustained vowel, counting, sentence reading
- **Real-time Analysis**: Record or upload audio files
- **Professional Visualization**: Waveform display and feature analysis
- **High Accuracy**: 92.31% test accuracy, 97% recall
- **Explainable AI**: Feature importance and confidence scores
- **User-Friendly Interface**: Clean, intuitive Streamlit web app

## Technical Details

### Model Performance
- **Algorithm**: XGBoost Ensemble
- **Training Data**: UCI Parkinson's Dataset (195 samples)
- **Test Accuracy**: 92.31%
- **Cross-Validation**: 93.85% ± 2.05%
- **Recall**: 97% (catches 97% of Parkinson's cases)
- **Precision**: 93%

### Features Extracted
22 acoustic features including:
- Pitch variations (F0, Jitter)
- Amplitude variations (Shimmer)
- Harmonics-to-Noise Ratio (HNR)
- MFCCs and spectral features

## Installation

### Prerequisites
- Python 3.11+
- Microphone access (for live recording)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/parkinsons-voice-detection.git
cd parkinsons-voice-detection
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Dependencies
```
streamlit
librosa
soundfile
numpy
pandas
scikit-learn
xgboost
matplotlib
seaborn
```

##  Usage

1. **Select a voice test** (Sustained vowel recommended)
2. **Record your voice** or upload an audio file
3. **Click "Analyze Voice"** to process
4. **View results** with confidence scores and insights

##  Important Disclaimers

- This is a **screening tool**, NOT a medical diagnosis
- Trained on adults aged 50-80 years
- High sensitivity may produce false positives for younger individuals
- Always consult healthcare professionals for proper evaluation

##  Model Development

We evaluated multiple models:
- XGBoost (selected for best performance)
- Random Forest
- SVM
- Voting Ensemble

XGBoost was chosen for its superior accuracy and consistency across cross-validation folds.

##  Hackathon

Developed for **Hacksphere 2.0** by BITS Pilani Hyderabad

## License

This project is for educational and research purposes only.



## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Parkinson's dataset
- BITS Pilani Hyderabad for organizing Hacksphere 2.0