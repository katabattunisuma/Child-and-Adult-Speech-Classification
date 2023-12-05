# feature_extractor.py
import librosa
import numpy as np
from pydub import AudioSegment
import pandas as pd
import librosa.feature as lrf


class FeatureExtractor:
    @staticmethod
    def load_audio(file_path, target_length=16000):
        print(file_path)
        if file_path.endswith(".mp3"):
            audio = AudioSegment.from_mp3(file_path)
            file_path = file_path.replace(".mp3", ".wav")
            audio.export(file_path, format="wav")

        signal, sr = librosa.load(file_path, sr=None)
        if len(signal) > target_length:
            signal = signal[:target_length]
        elif len(signal) < target_length:
            signal = np.pad(
                signal, (0, max(0, target_length - len(signal))), "constant"
            )
        return signal, sr

    @staticmethod
    def extract_mfcc(signal, sr=22050, n_mfcc=13):
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed

    @staticmethod
    def extract_feature(audio_data):
        RATE = 44100
        FRAME = 512

        zcr = lrf.zero_crossing_rate(
            audio_data, frame_length=FRAME, hop_length=int(FRAME / 2)
        )
        feature_zcr = np.mean(zcr)

        mfcc = lrf.mfcc(y=audio_data, sr=RATE, n_mfcc=13)
        feature_mfcc = np.mean(mfcc, axis=1)

        spectral_centroid = lrf.spectral_centroid(
            y=audio_data, sr=RATE, hop_length=int(FRAME / 2)
        )
        feature_spectral_centroid = np.mean(spectral_centroid)

        spectral_bandwidth = lrf.spectral_bandwidth(
            y=audio_data, sr=RATE, hop_length=int(FRAME / 2)
        )
        feature_spectral_bandwidth = np.mean(spectral_bandwidth)

        spectral_rolloff = lrf.spectral_rolloff(
            y=audio_data, sr=RATE, hop_length=int(FRAME / 2), roll_percent=0.90
        )
        feature_spectral_rolloff = np.mean(spectral_rolloff)

        spectral_flatness = lrf.spectral_flatness(
            y=audio_data, hop_length=int(FRAME / 2)
        )
        feature_spectral_flatness = np.mean(spectral_flatness)

        features = np.append(
            [
                feature_zcr,
                feature_spectral_centroid,
                feature_spectral_bandwidth,
                feature_spectral_rolloff,
                feature_spectral_flatness,
            ],
            feature_mfcc,
        )

        return features
