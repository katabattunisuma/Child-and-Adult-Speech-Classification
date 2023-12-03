import joblib
import os
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
