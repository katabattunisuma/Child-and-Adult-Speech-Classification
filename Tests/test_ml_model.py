"""from tensorflow.keras.models import load_model

loaded_model = load_model(
    "/Users/sumakatabattuni/Documents/Child-and-Adult-Speech-Classification/trained_models/overlap_unet_model"
)
import librosa
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
)


def spectrogram_to_audio(spectrogram, hop_length=512):
    return librosa.istft(spectrogram, hop_length=hop_length)


def load_audio(file_path, sr=22050, duration=5):
    # Load and pad/truncate the audio file
    audio, _ = librosa.load(file_path, sr=sr, duration=duration)
    return audio


def load_and_preprocess(file_path):
    audio = load_audio(file_path)
    spectrogram = audio_to_spectrogram(audio)
    return spectrogram


def audio_to_spectrogram(audio, n_fft=1024, hop_length=512, target_shape=(128, 128)):
    # Generate spectrogram
    spectrogram = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(spectrogram)

    # Pad or truncate to the target shape
    height, width = spectrogram.shape
    target_height, target_width = target_shape

    # Truncate if necessary
    if height > target_height:
        spectrogram = spectrogram[:target_height, :]
    if width > target_width:
        spectrogram = spectrogram[:, :target_width]

    # Pad if necessary
    pad_height = max(0, target_height - height)
    pad_width = max(0, target_width - width)

    spectrogram = np.pad(
        spectrogram, ((0, pad_height), (0, pad_width)), mode="constant"
    )

    return spectrogram


def unet_model(input_size=(128, 128, 1)):
    inputs = Input(input_size)

    # Downsampling
    conv1 = Conv2D(16, 3, activation="relu", padding="same")(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Upsampling
    up1 = UpSampling2D(size=(2, 2))(pool1)
    conv2 = Conv2D(1, 3, activation="relu", padding="same")(up1)

    model = tf.keras.Model(inputs=inputs, outputs=conv2)
    return model


def predict_baby_voice(mixed_file, model):
    mixed_spectrogram = load_and_preprocess(mixed_file)
    mixed_spectrogram = mixed_spectrogram.reshape(1, 128, 128, 1)
    baby_voice_spectrogram = model.predict(mixed_spectrogram)

    # Reshape and convert back to audio
    baby_voice_spectrogram = baby_voice_spectrogram.reshape(128, 128)
    baby_voice_audio = spectrogram_to_audio(baby_voice_spectrogram)

    return baby_voice_audio


# Example Usage
# prediction = predict_baby_voice("path/to/a/mixed/audio/file")
import soundfile as sf


def save_audio(audio, file_path, sr=22050):
    sf.write(file_path, audio, sr)


# Example usage
extracted_audio = predict_baby_voice(
    "/Users/sumakatabattuni/Documents/Child-and-Adult-Speech-Classification/Data/Overlapped Speech/Overlap /audio_5.wav",
    loaded_model,
)
save_audio(
    extracted_audio,
    "/Users/sumakatabattuni/Documents/Child-and-Adult-Speech-Classification/Data/Extracted Baby Voice/extracted_baby_voice2.wav",
)
"""
import joblib
import random
import os
import librosa
import numpy as np
import pandas as pd
from feature_extractor import FeatureExtractor
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

# Load the trained model
model_path = "trained_models/rf_625.joblib"
# model = joblib.load(model_path)
model = joblib.load(model_path)
# Directories
audio_dir = "/Users/sumakatabattuni/Documents/Child_Speech_Classification/Processed Data/Adult Speech"
audio_dir2 = "/Users/sumakatabattuni/Documents/Child_Speech_Classification/Processed Data/Child Speech"

all_actual_labels = []
all_predicted_labels = []

# List all audio files
audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
audio_files2 = [f for f in os.listdir(audio_dir2) if f.endswith(".wav")]
# Initialize lists to store accuracies
accuracies = []


# random.shuffle(audio_files)
for audio_file in audio_files2:
    # Load and preprocess audio
    print(audio_file)
    audio_path = os.path.join(audio_dir2, audio_file)
    # audio, sr = librosa.load(audio_path, sr=None)
    signal, sr = FeatureExtractor.load_audio(audio_path)
    features = FeatureExtractor.extract_mfcc(signal, sr)

    # Load actual labels and one-hot encode them

    actual_labels = 0
    # actual_labels_one_hot = to_categorical(actual_labels, num_classes=2)

    predicted_labels = model.predict(features.reshape(1, -1))

    # Calculate and store accuracy
    accuracy = np.mean(predicted_labels == actual_labels)
    accuracies.append(accuracy)
    print("==========================")
    print("actual ", actual_labels)
    print("predicted ", predicted_labels)
    print(accuracy)
    all_actual_labels.append(actual_labels)
    all_predicted_labels.append(predicted_labels)
    # conf_matrix = confusion_matrix(actual_labels, predicted_labels)
    # print(f"Confusion Matrix for {audio_file}:")
    # print(conf_matrix)
print("finished adult files")

# Process each file
for audio_file in audio_files:
    # Load and preprocess audio
    print(audio_file)
    audio_path = os.path.join(audio_dir, audio_file)
    signal, sr = FeatureExtractor.load_audio(audio_path)
    features = FeatureExtractor.extract_mfcc(signal, sr)

    # Load actual labels and one-hot encode them

    actual_labels = 1
    # actual_labels_one_hot = to_categorical(actual_labels, num_classes=2)

    predicted_labels = model.predict(features.reshape(1, -1))

    # Calculate and store accuracy
    accuracy = np.mean(predicted_labels == actual_labels)
    accuracies.append(accuracy)
    print("==========================")
    print("actual ", actual_labels)
    print("predicted ", predicted_labels)
    print(accuracy)
    all_actual_labels.append(actual_labels)
    all_predicted_labels.append(predicted_labels)
    # conf_matrix = confusion_matrix(actual_labels, predicted_labels)
    # print(f"Confusion Matrix for {audio_file}:")
    # print(conf_matrix)
print("finished adult files")


# Calculate overall accuracy
overall_accuracy = np.mean(accuracies)
print(f"Overall accuracy: {overall_accuracy * 100:.2f}%")
overall_conf_matrix = confusion_matrix(all_actual_labels, all_predicted_labels)
print("Overall Confusion Matrix:")
print(overall_conf_matrix)
