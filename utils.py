import os
import numpy as np
import random
from feature_extractor import FeatureExtractor


class DatasetLoader:
    @staticmethod
    def load_dataset_and_labels(
        child_folder, adult_folder, target_length=16000, max_adult_files=1000
    ):
        child_files = [
            os.path.join(child_folder, f)
            for f in os.listdir(child_folder)
            if f.endswith(".wav")
        ]

        # Get a list of all adult files
        adult_files = [
            os.path.join(adult_folder, f)
            for f in os.listdir(adult_folder)
            if f.endswith(".mp3")
        ]

        # Randomly shuffle the list and select the first 1000 (or the max_adult_files limit)
        random.shuffle(adult_files)
        adult_files = adult_files[:max_adult_files]

        X = []
        y = []

        # Load child data
        for file in child_files:
            signal, sr = FeatureExtractor.load_audio(file, target_length)
            features = FeatureExtractor.extract_mfcc(signal, sr)
            X.append(features)
            y.append(0)  # Label for child

        # Load adult data
        for file in adult_files:
            signal, sr = FeatureExtractor.load_audio(file, target_length)
            features = FeatureExtractor.extract_mfcc(signal, sr)
            X.append(features)
            y.append(1)  # Label for adult

        return np.array(X), np.array(y)
