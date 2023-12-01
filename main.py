# main.py
from sklearn.model_selection import train_test_split
from utils import DatasetLoader
from models.random_forest import RandomForestModel
import os

# Load dataset
baby_folder = "/Users/sumakatabattuni/Documents/Child-and-Adult-Speech-Classification/Data/Child Speech"
adult_folder = "/Users/sumakatabattuni/Documents/Child-and-Adult-Speech-Classification/Data/Adult Speech"
X, y = DatasetLoader.load_dataset_and_labels(baby_folder, adult_folder)


# Split the dataset for both models
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# RandomForestModel
rf_model = RandomForestModel()
rf_model.train(X_train, y_train)

# Define the directory for saving the model
model_dir = os.path.join(os.getcwd(), "trained_models")
os.makedirs(model_dir, exist_ok=True)  # Create the directory if it does not exist

"""model_filename = os.path.join(
    model_dir, "trained_random_forest_personalized.model.joblib"
)"""

# Save the trained model in the specified directory
model_filename = os.path.join(model_dir, "trained_random_forest_model.joblib")
rf_model.save(model_filename)
print(f"Model saved as {model_filename}")

# Evaluate the model
rf_accuracy = rf_model.evaluate(X_test, y_test)
print(f"Random Forest Accuracy: {rf_accuracy}")
