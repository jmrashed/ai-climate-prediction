import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def clean_data(df):
    """
    Perform basic data cleaning tasks such as removing duplicates,
    handling missing values, and ensuring data types are consistent.

    Args:
        df (pd.DataFrame): The raw data to clean.

    Returns:
        pd.DataFrame: The cleaned data.
    """
    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df = handle_missing_values(df)

    # Convert date column to datetime (if exists)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Drop rows with invalid or NaT dates
    df = df.dropna(subset=['date'])

    return df

def handle_missing_values(df):
    """
    Handle missing values in the dataset using SimpleImputer.

    Args:
        df (pd.DataFrame): The data to process.

    Returns:
        pd.DataFrame: The data with imputed missing values.
    """
    imputer = SimpleImputer(strategy='mean')  # Using mean imputation
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:  # Only impute numerical columns
            df[column] = imputer.fit_transform(df[[column]])

    return df

def transform_features(df):
    """
    Perform feature engineering, such as creating new features
    or transforming existing ones.

    Args:
        df (pd.DataFrame): The data to process.

    Returns:
        pd.DataFrame: The data with transformed features.
    """
    # Example: Adding a new feature based on existing columns
    if 'temperature' in df.columns and 'precipitation' in df.columns:
        df['temp_precipitation_ratio'] = df['temperature'] / (df['precipitation'] + 1)  # Avoid division by zero

    return df

def scale_data(df, scaling_method='standard'):
    """
    Scale the numerical features of the dataset using the chosen scaling method.

    Args:
        df (pd.DataFrame): The data to scale.
        scaling_method (str): The scaling method ('standard' or 'minmax').

    Returns:
        pd.DataFrame: The scaled data.
    """
    scaler = None

    # Select the scaling method
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()

    # Apply scaling only to numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

def split_data(df, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Args:
        df (pd.DataFrame): The data to split.
        test_size (float): The proportion of data to include in the test set.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Assuming the target variable is 'target', change it if necessary
    if 'target' not in df.columns:
        raise ValueError("'target' column not found in the dataframe.")

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test
