from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


class NeuralNetworkModel:
    def __init__(self, input_shape):
        self.model = Sequential(
            [
                Dense(256, input_shape=input_shape, activation="relu"),
                Dropout(0.5),
                Dense(128, activation="relu"),
                Dropout(0.5),
                Dense(64, activation="relu"),
                Dense(2, activation="softmax"),
            ]
        )
        self.model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(
            X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val)
        )

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
