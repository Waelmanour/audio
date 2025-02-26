# Audio Deepfake Detector

A deep learning-based application that detects whether an audio file is authentic or artificially generated (deepfake).

## Features

- Upload and analyze audio files (WAV, MP3, OGG formats)
- Real-time audio deepfake detection
- Visual representation of analysis results
- User-friendly web interface

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train_model.py
```

3. Run the application locally:
```bash
streamlit run app.py
```

## Deploying to Streamlit Cloud

1. Create a Streamlit Cloud account at https://streamlit.io/cloud

2. Connect your GitHub repository to Streamlit Cloud

3. Deploy the app by selecting the repository and the main file (app.py)

4. Make sure to upload the trained model file (audio_deepfake_detector.h5) to the model/ directory

## Project Structure

- `app.py`: Main Streamlit application
- `train_model.py`: Script for training the deepfake detection model
- `requirements.txt`: Project dependencies
- `model/`: Directory for storing the trained model
- `for-2sec/for-2seconds/`: Dataset directory containing training and testing data

## Model Architecture

The model uses a deep neural network to analyze audio features:
- Input: Mel-frequency cepstral coefficients (MFCCs)
- Hidden layers with dropout for regularization
- Binary classification output (authentic vs deepfake)

## Dataset

The model is trained on a dataset organized in the following structure:
```
for-2sec/for-2seconds/
├── training/
│   ├── real/
│   └── fake/
├── testing/
│   ├── real/
│   └── fake/
└── validation/
    ├── real/
    └── fake/
```