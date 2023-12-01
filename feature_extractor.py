# feature_extractor.py
import librosa
import numpy as np
from pydub import AudioSegment


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
