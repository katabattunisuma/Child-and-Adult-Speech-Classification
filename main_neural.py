from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from utils import DatasetLoader
from models.neural_network import NeuralNetworkModel

# Load dataset
baby_folder = "/Users/sumakatabattuni/Documents/Child-and-Adult-Speech-Classification/Data/Child Speech"
adult_folder = "/Users/sumakatabattuni/Documents/Child-and-Adult-Speech-Classification/Data/Adult Speech"
X, y = DatasetLoader.load_dataset_and_labels(baby_folder, adult_folder)

# Encode labels for neural network
y_categorical = to_categorical(y)


X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# NeuralNetworkModel
nn_model = NeuralNetworkModel(input_shape=(X_train_nn.shape[1],))
nn_model.train(X_train_nn, y_train_nn, X_test_nn, y_test_nn)

model_dir = os.path.join(os.getcwd(), "trained_models")
os.makedirs(model_dir, exist_ok=True)  # Create the directory if it does not exist

# Save the trained model in the specified directory
model_filename = os.path.join(model_dir, "trained_random_forest_model.joblib")
nn_model.save(model_filename)
print(f"Model saved as {model_filename}")


nn_scores = nn_model.evaluate(X_test_nn, y_test_nn)
print(f"Neural Network Accuracy: {nn_scores[1] * 100}%")
