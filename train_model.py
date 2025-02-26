import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import librosa
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

def extract_features(audio_path):
    try:
        # Load audio file
        audio_data, sr = librosa.load(audio_path)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        
        return mfcc_scaled
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def create_model(input_shape):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_dataset(real_dir, fake_dir):
    features = []
    labels = []
    
    # Process real audio files
    for audio_file in Path(real_dir).glob('*.wav'):
        feature = extract_features(str(audio_file))
        if feature is not None:
            features.append(feature)
            labels.append(0)  # 0 for real
    
    # Process fake audio files
    for audio_file in Path(fake_dir).glob('*.wav'):
        feature = extract_features(str(audio_file))
        if feature is not None:
            features.append(feature)
            labels.append(1)  # 1 for fake
    
    return np.array(features), np.array(labels)

def main():
    # Set paths
    base_dir = Path('for-2sec/for-2seconds')
    train_dir = base_dir / 'training'
    test_dir = base_dir / 'testing'
    
    # Prepare training data
    print("Preparing training data...")
    X_train, y_train = prepare_dataset(
        train_dir / 'real',
        train_dir / 'fake'
    )
    
    # Prepare test data
    print("Preparing test data...")
    X_test, y_test = prepare_dataset(
        test_dir / 'real',
        test_dir / 'fake'
    )
    
    # Create and train model
    print("Creating model...")
    model = create_model(X_train.shape[1])
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save model
    model_dir = Path('model')
    model_dir.mkdir(exist_ok=True)
    model.save(model_dir / 'audio_deepfake_detector.h5')
    print("\nModel saved successfully!")

if __name__ == '__main__':
    main()