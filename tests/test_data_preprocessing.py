import pytest
import pandas as pd
from src.data_preprocessing.preprocess import clean_data, feature_engineering

# Example test data
test_data = {
    'temperature': [22.5, 25.0, 19.0, None, 30.5],
    'humidity': [50, 55, 60, 65, None],
    'precipitation': [0.1, None, 0.0, 0.3, 0.2]
}

# Convert test data to a pandas DataFrame
df = pd.DataFrame(test_data)

def test_clean_data():
    """Test the data cleaning function."""
    cleaned_df = clean_data(df)
    
    # Check that the data is cleaned and missing values are handled
    assert cleaned_df.isnull().sum().sum() == 0, "Missing values were not properly handled"
    
    # You can add other checks based on the cleaning process (e.g., removing outliers, etc.)
    assert 'temperature' in cleaned_df.columns, "Temperature column missing after cleaning"
    assert 'humidity' in cleaned_df.columns, "Humidity column missing after cleaning"

def test_feature_engineering():
    """Test the feature engineering function."""
    engineered_df = feature_engineering(df)
    
    # Check that new features are created (example: if a 'day_of_week' feature is added)
    assert 'day_of_week' in engineered_df.columns, "Day of week feature was not created"
    
    # Check if the feature engineering function has preserved other necessary columns
    assert 'temperature' in engineered_df.columns, "Temperature column missing after feature engineering"
    assert 'humidity' in engineered_df.columns, "Humidity column missing after feature engineering"

# You can also test edge cases, such as passing empty data or data with specific issues.
def test_empty_dataframe():
    """Test the behavior when passing an empty DataFrame."""
    empty_df = pd.DataFrame()
    
    cleaned_empty_df = clean_data(empty_df)
    assert cleaned_empty_df.empty, "The cleaned empty dataframe should still be empty"
    
def test_invalid_data():
    """Test the behavior with invalid data (e.g., non-numeric values)."""
    invalid_data = {
        'temperature': ['high', 'low', 'medium', 'medium', 'high'],
        'humidity': [50, 55, 60, 'low', 45]
    }
    invalid_df = pd.DataFrame(invalid_data)
    
    with pytest.raises(ValueError, match="Invalid data type in 'temperature' column"):
        clean_data(invalid_df)
