import pytest
import numpy as np
import pandas as pd
from src.models.model import train_model, evaluate_model, save_model, load_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Example test data for model training and evaluation
test_data = {
    'temperature': [22.5, 25.0, 19.0, 30.5, 28.0, 23.5],
    'humidity': [50, 55, 60, 65, 58, 60],
    'precipitation': [0.1, 0.2, 0.0, 0.3, 0.4, 0.1],
    'target': [12.0, 13.0, 11.5, 14.2, 15.3, 12.8]  # Hypothetical target (e.g., climate prediction)
}

df = pd.DataFrame(test_data)
X = df[['temperature', 'humidity', 'precipitation']]  # Features
y = df['target']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Dummy model for testing purposes
model = RandomForestRegressor(n_estimators=100, random_state=42)

def test_train_model():
    """Test the model training functionality."""
    # Train the model using the training data
    trained_model = train_model(X_train, y_train, model)
    
    # Check if the model is fitted (it should have the `feature_importances_` attribute)
    assert hasattr(trained_model, 'feature_importances_'), "Model was not trained successfully"
    
    # Optionally: Check if the model performs well on training data
    train_predictions = trained_model.predict(X_train)
    train_mse = mean_squared_error(y_train, train_predictions)
    assert train_mse < 1.0, f"Training MSE is too high: {train_mse}"

def test_evaluate_model():
    """Test the model evaluation functionality."""
    # Train the model first
    trained_model = train_model(X_train, y_train, model)
    
    # Evaluate the model on test data
    evaluation_result = evaluate_model(trained_model, X_test, y_test)
    
    # Check if evaluation returns the expected metric (MSE in this case)
    assert 'mse' in evaluation_result, "Evaluation result should include 'mse'"
    assert evaluation_result['mse'] < 2.0, f"MSE is too high: {evaluation_result['mse']}"

def test_save_and_load_model():
    """Test saving and loading of the model."""
    # Train the model first
    trained_model = train_model(X_train, y_train, model)
    
    # Save the model
    save_model(trained_model, 'model.pkl')
    
    # Load the model
    loaded_model = load_model('model.pkl')
    
    # Check if the model was loaded correctly (it should have the same attributes)
    assert hasattr(loaded_model, 'feature_importances_'), "Model was not loaded correctly"
    
    # Check if the loaded model performs similarly to the original model
    original_preds = trained_model.predict(X_test)
    loaded_preds = loaded_model.predict(X_test)
    assert np.allclose(original_preds, loaded_preds, atol=1e-5), "Predictions from loaded model differ too much"

def test_invalid_model_input():
    """Test the behavior when invalid inputs are passed to the model."""
    # Pass invalid data (e.g., empty dataframe)
    with pytest.raises(ValueError, match="Data cannot be empty"):
        train_model(pd.DataFrame(), pd.Series(), model)
    
    # Pass invalid model type (e.g., a non-scikit-learn model)
    with pytest.raises(TypeError, match="Invalid model type"):
        train_model(X_train, y_train, "not_a_model")

# Additional test for model evaluation metrics
def test_model_evaluation_metrics():
    """Test the evaluation metrics provided by the evaluate_model function."""
    # Train the model first
    trained_model = train_model(X_train, y_train, model)
    
    # Evaluate the model
    result = evaluate_model(trained_model, X_test, y_test)
    
    # Ensure that the result contains expected keys and values
    assert 'mse' in result, "Model evaluation result should contain MSE"
    assert 'r2' in result, "Model evaluation result should contain R^2 score"
    assert result['mse'] >= 0, "MSE should be non-negative"
    assert result['r2'] >= 0, "R^2 should be non-negative"

