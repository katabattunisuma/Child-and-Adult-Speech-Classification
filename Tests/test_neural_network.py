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
model_path = "CNN_RNN_new_balance_300.h5"
# model = joblib.load(model_path)
model = load_model(model_path)
# Directories
audio_dir = "Processed Data/Adult Speech"
audio_dir2 = "Processed Data/Child Speech"

all_actual_labels = []
all_predicted_labels = []

# List all audio files
audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
audio_files2 = [f for f in os.listdir(audio_dir2) if f.endswith(".wav")]
# Initialize lists to store accuracies
accuracies = []


def preprocess_spectogram(audio, max_pad_len=174):
    mfcc_features = []
    # np.array([librosa.feature.melspectrogram(y=chunk) for chunk in chunks])

    spectrogram = librosa.feature.melspectrogram(y=audio)

    if spectrogram.shape[1] > max_pad_len:
        # Truncate the spectrogram to max_pad_len
        spectrogram = spectrogram[:, :max_pad_len]
    else:
        # Calculate pad_width and pad the spectrogram
        pad_width = max_pad_len - spectrogram.shape[1]
        spectrogram = np.pad(
            spectrogram, pad_width=((0, 0), (0, pad_width)), mode="constant"
        )
    mfcc_features.append(spectrogram)
    # features = FeatureExtractor.extract_feature(audio_data=signal)

    mfcc_features = np.array(mfcc_features)
    # Flatten the MFCC to have 2D array for the ML model
    # mfcc_features_flattened = np.array([mfcc.flatten() for mfcc in mfcc_features])
    return mfcc_features


# random.shuffle(audio_files)
for audio_file in audio_files2:
    # Load and preprocess audio

    print(audio_file)
    audio_path = os.path.join(audio_dir2, audio_file)
    audio, sr = librosa.load(audio_path, sr=None)
    preprocessed_chunks = preprocess_spectogram(audio)

    actual_labels = 0

    predicted_probabilities = model.predict(preprocessed_chunks)
    print(predicted_probabilities)

    predicted_labels = np.argmax(predicted_probabilities, axis=1)

    # Calculate and store accuracy
    accuracy = np.mean(predicted_labels == actual_labels)
    accuracies.append(accuracy)

    print("actual ", actual_labels)
    print("predicted ", predicted_labels)
    print(accuracy)
    all_actual_labels.append(actual_labels)
    all_predicted_labels.append(predicted_labels)

print("finished adult files")
# Process each file
for audio_file in audio_files:
    # Load and preprocess audio
    print(audio_file)
    audio_path = os.path.join(audio_dir, audio_file)
    audio, sr = librosa.load(audio_path, sr=None)
    preprocessed_chunks = preprocess_spectogram(audio)

    # Load actual labels and one-hot encode them

    actual_labels = 1
    # actual_labels_one_hot = to_categorical(actual_labels, num_classes=2)

    predicted_probabilities = model.predict(preprocessed_chunks)
    print(predicted_probabilities)
    # Get the predicted labels with the highest probability
    predicted_labels = np.argmax(predicted_probabilities, axis=1)

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
