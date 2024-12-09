import numpy as np
import random
import tensorflow as tf

def set_random_seed(seed_value=42):
    """
    Set the random seed for reproducibility.

    Args:
        seed_value (int): The seed value to use for randomness (default is 42).
    """
    # Set seed for random number generators
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    # If using TensorFlow/Keras, also set the random seed for them
    tf.random.set_seed(seed_value)
    
    # For other libraries like PyTorch, additional seeding might be needed
    # Example for PyTorch:
    # torch.manual_seed(seed_value)
    # torch.cuda.manual_seed_all(seed_value)

    print(f"Random seed set to {seed_value} for reproducibility.")

def model_summary(model):
    """
    Print the summary of a model (especially useful for neural networks).

    Args:
        model: The trained model (e.g., a scikit-learn or deep learning model).
    """
    try:
        # For deep learning models (e.g., using Keras or TensorFlow)
        model.summary()
    except AttributeError:
        # If the model doesn't have the `summary` method (e.g., scikit-learn models)
        print("Model summary is not available for this model type.")
    
def save_model_version(model, filename, version="1.0"):
    """
    Save the model with version information in the filename.

    Args:
        model: The trained model to save.
        filename (str): The base filename (without extension).
        version (str): The version of the model (default is "1.0").
    """
    versioned_filename = f"{filename}_v{version}.pkl"
    with open(versioned_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved as {versioned_filename}")

def load_model_version(filename):
    """
    Load a model based on the versioned filename.

    Args:
        filename (str): The versioned model filename.

    Returns:
        model: The loaded model.
    """
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model
    except FileNotFoundError:
        print(f"Error: The model file {filename} was not found.")
        return None

def check_model_overfitting(model, X_train, y_train, X_val, y_val, threshold=0.1):
    """
    Check if a model is overfitting based on training and validation performance.

    Args:
        model: The trained model.
        X_train (np.array or pd.DataFrame): The training feature data.
        y_train (np.array or pd.Series): The training target data.
        X_val (np.array or pd.DataFrame): The validation feature data.
        y_val (np.array or pd.Series): The validation target data.
        threshold (float): The acceptable difference between training and validation error (default is 0.1).
    
    Returns:
        bool: True if overfitting is detected, otherwise False.
    """
    # Calculate training error
    train_pred = model.predict(X_train)
    train_error = mean_squared_error(y_train, train_pred)

    # Calculate validation error
    val_pred = model.predict(X_val)
    val_error = mean_squared_error(y_val, val_pred)

    # Check if the model is overfitting
    if abs(train_error - val_error) > threshold:
        print("Warning: Potential overfitting detected!")
        return True
    else:
        print("No significant overfitting detected.")
        return False

def plot_feature_importances(model, feature_names):
    """
    Plot feature importances for tree-based models.

    Args:
        model: A trained tree-based model (e.g., RandomForest, XGBoost).
        feature_names (list): List of feature names.

    Returns:
        matplotlib figure: The plot of feature importances.
    """
    import matplotlib.pyplot as plt
    
    # Check if the model has feature_importances_ attribute (e.g., RandomForest)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        raise AttributeError("Model doesn't have 'feature_importances_' attribute.")
    
    # Sort the feature importances
    sorted_idx = np.argsort(importances)[::-1]
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance of the Model')
    plt.show()

def save_model_as_tf(model, filename):
    """
    Save a trained model as a TensorFlow model (for deep learning models).

    Args:
        model: The trained deep learning model (e.g., Keras or TensorFlow model).
        filename (str): The path where the model will be saved.
    """
    model.save(filename)
    print(f"Model saved as TensorFlow format at {filename}")

def load_tf_model(filename):
    """
    Load a TensorFlow model from the given file.

    Args:
        filename (str): The path to the saved model.

    Returns:
        model: The loaded TensorFlow model.
    """
    loaded_model = tf.keras.models.load_model(filename)
    print(f"TensorFlow model loaded from {filename}")
    return loaded_model
