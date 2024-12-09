from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using test data and common regression metrics.

    Args:
        model: The trained model to evaluate.
        X_test (pd.DataFrame or np.array): The feature data for testing.
        y_test (pd.Series or np.array): The true target values for testing.

    Returns:
        dict: A dictionary containing evaluation metrics (MAE, MSE, RMSE, R^2).
    """
    # Predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Return as a dictionary
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

def cross_validate_model(model, X, y, cv=5):
    """
    Perform cross-validation to evaluate the model's performance.

    Args:
        model: The model to cross-validate.
        X (pd.DataFrame or np.array): The feature data.
        y (pd.Series or np.array): The target values.
        cv (int): Number of cross-validation folds (default is 5).

    Returns:
        dict: A dictionary containing cross-validation results (mean score and std deviation).
    """
    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    
    # Return mean and standard deviation of cross-validation scores
    return {
        'mean_score': np.mean(scores),
        'std_dev': np.std(scores)
    }

def calculate_metrics(y_true, y_pred):
    """
    Calculate common regression metrics from true and predicted values.

    Args:
        y_true (pd.Series or np.array): The true target values.
        y_pred (pd.Series or np.array): The predicted target values.

    Returns:
        dict: A dictionary containing metrics (MAE, MSE, RMSE, R^2).
    """
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }
