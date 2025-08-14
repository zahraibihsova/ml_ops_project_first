# this script is for loading model and using it to make predictions.
import pickle

# Assumeing this is the previously trained model
model_filename = "trained_model.pkl"

try:
    # Open the file in binary read mode ('rb')
    with open(model_filename, "rb") as file:
        # Load the model using pickle.load()
        loaded_model = pickle.load(file)
    print(f"Model successfully loaded from {model_filename}")


except FileNotFoundError:
    print(f"Error: Model file '{model_filename}' not found.")
except Exception as e:
    print(f"An error occurred during deserialization: {e}")