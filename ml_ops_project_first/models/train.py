# models/train_model.py
# This script trains a simple Logistic Regression model and saves it.
import os
import pickle

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


def train_model():
    """
    this code trains simple logistic model and saves file to directory by using pickles
    """
    print("Starting model training...")

    # Load a dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Initialize and train a Logistic Regression model
    model = LogisticRegression(max_iter=200)  # Increased max_iter for convergence
    model.fit(X, y)

    # Use os.path.join to create a path to the 'models' directory
    model_filename = "trained_model.pkl"
    model_path = os.path.join("models", model_filename)

    # Serialize the trained model
    try:
        with open(model_path, "wb") as file:
            pickle.dump(model, file)
        print(f"Model successfully saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
    return model_path


if __name__ == "__main__":
    train_model()
