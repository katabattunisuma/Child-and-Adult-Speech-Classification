from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load


class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        return accuracy_score(y_test, predictions)

    def save(self, filename):
        dump(self.model, filename)

    def load(self, filename):
        self.model = load(filename)
