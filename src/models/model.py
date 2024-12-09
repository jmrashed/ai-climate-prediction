import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def train_model(model, X, y, test_size=0.2, random_state=42, grid_search=False, param_grid=None):
    """
    Train the model on the given data.

    Args:
        model: The model to train.
        X (pd.DataFrame or np.array): The feature data.
        y (pd.Series or np.array): The target data.
        test_size (float): Proportion of the data to use for testing (default is 0.2).
        random_state (int): Seed for random number generator (default is 42).
        grid_search (bool): Whether to perform hyperparameter tuning using grid search (default is False).
        param_grid (dict): The grid of hyperparameters to search over (if grid_search=True).

    Returns:
        model: The trained model.
        X_train, X_test, y_train, y_test: The training and test data splits.
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Hyperparameter tuning with GridSearchCV (if enabled)
    if grid_search and param_grid:
        grid_search_model = GridSearchCV(model, param_grid, cv=5)
        grid_search_model.fit(X_train, y_train)
        model = grid_search_model.best_estimator_  # Use the best model from grid search

    # Train the model
    model.fit(X_train, y_train)

    # Return the trained model and the data splits
    return model, X_train, X_test, y_train, y_test


def save_model(model, filename):
    """
    Save the trained model to a file.

    Args:
        model: The trained model to save.
        filename (str): The path to save the model.
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def load_model(filename):
    """
    Load a trained model from a file.

    Args:
        filename (str): The path to the model file.

    Returns:
        model: The loaded model.
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model


def save_model_joblib(model, filename):
    """
    Save the trained model using joblib (useful for large models).

    Args:
        model: The trained model to save.
        filename (str): The path to save the model.
    """
    joblib.dump(model, filename)


def load_model_joblib(filename):
    """
    Load a trained model using joblib.

    Args:
        filename (str): The path to the model file.

    Returns:
        model: The loaded model.
    """
    return joblib.load(filename)


def model_metrics(model, X_test, y_test):
    """
    Calculate common regression metrics for a model's predictions.

    Args:
        model: The trained model.
        X_test (pd.DataFrame or np.array): The test features.
        y_test (pd.Series or np.array): The true target values.

    Returns:
        dict: A dictionary containing MAE, MSE, RMSE, and R^2 metrics.
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }
