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


baby_folder = "/Users/sumakatabattuni/Documents/Child-and-Adult-Speech-Classification/Data/Overlapped Speech/Child Speech"
adult_folder = "/Users/sumakatabattuni/Documents/Child-and-Adult-Speech-Classification/Data/Overlapped Speech/Adult Speech"
mixed_folder = "/Users/sumakatabattuni/Documents/Child-and-Adult-Speech-Classification/Data/Overlapped Speech/Overlap "

file_names = os.listdir(baby_folder)  # Assuming file names are aligned across folders

x_train = []
y_train = []

for file_name in file_names:
    print(file_name)
    mixed_spectrogram = load_and_preprocess(os.path.join(mixed_folder, file_name))
    baby_spectrogram = load_and_preprocess(os.path.join(baby_folder, file_name))

    x_train.append(mixed_spectrogram)
    y_train.append(baby_spectrogram)

# Convert to numpy arrays and reshape for the model
x_train = np.array(x_train).reshape(-1, 128, 128, 1)
y_train = np.array(y_train).reshape(-1, 128, 128, 1)


model = unet_model()
model.compile(optimizer="adam", loss="mean_squared_error")

model.fit(x_train, y_train, epochs=50, batch_size=32)


def predict_baby_voice(mixed_file, model):
    mixed_spectrogram = load_and_preprocess(mixed_file)
    mixed_spectrogram = mixed_spectrogram.reshape(1, 128, 128, 1)
    baby_voice_spectrogram = model.predict(mixed_spectrogram)

    # Reshape and convert back to audio
    baby_voice_spectrogram = baby_voice_spectrogram.reshape(128, 128)
    baby_voice_audio = spectrogram_to_audio(baby_voice_spectrogram)

    return baby_voice_audio


model.save(
    "/Users/sumakatabattuni/Documents/Child-and-Adult-Speech-Classification/trained_models/overlap_unet_model"
)
print("saved model")


# Example Usage
# prediction = predict_baby_voice("path/to/a/mixed/audio/file")
import soundfile as sf


def save_audio(audio, file_path, sr=22050):
    sf.write(file_path, audio, sr)


# Example usage
extracted_audio = predict_baby_voice(
    "/Users/sumakatabattuni/Documents/Child-and-Adult-Speech-Classification/Data/Overlapped Speech/Overlap /audio_3.wav",
    model,
)
save_audio(
    extracted_audio,
    "/Users/sumakatabattuni/Documents/Child-and-Adult-Speech-Classification/Data/Extracted Baby Voice/extracted_baby_voice.wav",
)

"""The current model's performance is not optimal, potentially due to the constraints of a limited training dataset. 
This presents an opportunity for future improvements and could be a focus for subsequent development."""
