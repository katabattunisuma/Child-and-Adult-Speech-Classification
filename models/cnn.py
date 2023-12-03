import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
    Flatten,
    LSTM,
    Dense,
    Reshape,
)

from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf


def load_files(folder_path, label, total_files=-1, max_pad_len=174):
    features = []
    for file in os.listdir(folder_path)[:total_files]:
        print(file)
        if file.endswith(".wav") or file.endswith(".mp3"):
            path = os.path.join(folder_path, file)
            audio, sample_rate = librosa.load(path, sr=None)
            # Use keyword arguments for melspectrogram
            spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)

            # Check if the spectrogram is longer than max_pad_len
            if spectrogram.shape[1] > max_pad_len:
                # Truncate the spectrogram to max_pad_len
                spectrogram = spectrogram[:, :max_pad_len]
            else:
                # Calculate pad_width and pad the spectrogram
                pad_width = max_pad_len - spectrogram.shape[1]
                spectrogram = np.pad(
                    spectrogram, pad_width=((0, 0), (0, pad_width)), mode="constant"
                )

            features.append([spectrogram, label])
    return features


# Load baby and adult audio files
baby_folder = "Data/Child Speech"  # Update this path
adult_folder = "Data/Adult Speech"  # Update this path

baby_data = load_files(baby_folder, 0)  # Label for baby files is 0
adult_data = load_files(adult_folder, 1, 400)  # Label for adult files is 1

# Combine and shuffle the data
all_data = baby_data + adult_data
np.random.shuffle(all_data)

# Split the features and labels
X, y = zip(*all_data)
X = np.array(
    [x.reshape(128, 174, 1) for x in X]
)  # Update shape based on your spectrogram
y = to_categorical(y, num_classes=2)


# Define a scheduler function
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def build_unet_cnn_rnn_model(input_shape):
    # U-Net encoder
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    pool1 = MaxPooling2D((2, 2))(conv1)

    # Bottleneck
    conv_mid = Conv2D(32, (3, 3), activation="relu", padding="same")(pool1)

    # U-Net decoder (symmetric to encoder)
    up1 = UpSampling2D((2, 2))(conv_mid)
    up_conv1 = Conv2D(16, (3, 3), activation="relu", padding="same")(up1)

    # Concatenation for U-Net
    concat = concatenate([conv1, up_conv1])

    # Flattening for RNN
    flatten = Flatten()(concat)

    # RNN Layer
    rnn_out = LSTM(64, return_sequences=False)(Reshape((-1, 64))(flatten))

    # Final Dense Layers
    dense = Dense(64, activation="relu")(rnn_out)
    outputs = Dense(2, activation="softmax")(dense)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model (use the build_unet_cnn_rnn_model function)
model = build_unet_cnn_rnn_model((128, 174, 1))  # Update the input shape if needed

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
# model.fit(X_train, y_train, epochs=30, callbacks=[LearningRateScheduler(scheduler)])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Include the plotting code to visualize training and validation accuracy and loss.

# Assuming 'model' is your trained Keras model
model.save("CNN_RNN_new_balance_400.h5")  # Saves the model to a HDF5 file
