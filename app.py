import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Audio Deepfake Detector",
    page_icon="🎵",
    layout="wide"
)

# Title and description
st.title("Audio Deepfake Detector 🎵")
st.markdown("""
    Upload an audio file to check if it's authentic or potentially a deepfake.
    Supported formats: WAV, MP3, OGG
""")

# File uploader
audio_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg'])

def process_audio(audio_data, sr):
    # Extract features (MFCC)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

def load_model():
    # Placeholder for model loading
    # TODO: Replace with actual model path once trained
    model_path = Path("model/audio_deepfake_detector.h5")
    if model_path.exists():
        return tf.keras.models.load_model(str(model_path))
    return None

if audio_file is not None:
    # Display audio player
    st.audio(audio_file)
    
    # Add a spinner during processing
    with st.spinner("Analyzing audio..."): 
        try:
            # Load and process audio
            audio_data, sr = librosa.load(audio_file)
            features = process_audio(audio_data, sr)
            
            # Load model and make prediction
            model = load_model()
            
            if model is not None:
                # Reshape features for model input
                features = np.expand_dims(features, axis=0)
                prediction = model.predict(features)
                
                # Display results
                st.subheader("Analysis Results")
                probability = float(prediction[0][0])
                
                # Create a progress bar for visualization
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Authenticity Score")
                    st.progress(1 - probability)
                    
                with col2:
                    st.markdown("### Prediction")
                    if probability > 0.5:
                        st.error("⚠️ Likely Deepfake")
                    else:
                        st.success("✅ Likely Authentic")
                
                # Additional details
                st.markdown("### Detailed Analysis")
                st.write(f"Deepfake Probability: {probability:.2%}")
            else:
                st.warning("Model not found. Please train the model first.")
                
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")

# Add information about the project
st.markdown("---")
st.markdown("""
### About this Detector

This audio deepfake detector uses deep learning to analyze audio files and determine if they are authentic or artificially generated. 

The model analyzes various audio features including:
- Mel-frequency cepstral coefficients (MFCCs)
- Spectral characteristics
- Temporal patterns

For best results, use high-quality audio recordings in supported formats.
""")