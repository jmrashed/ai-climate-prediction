# src/config.py

import os

# Set paths for directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, 'external')
MODEL_DIR = os.path.join(BASE_DIR, 'outputs', 'model')
RESULTS_DIR = os.path.join(BASE_DIR, 'outputs', 'results')
LOGS_DIR = os.path.join(BASE_DIR, 'outputs', 'logs')

# Hyperparameters for model training
HYPERPARAMETERS = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'dropout_rate': 0.3,
    'optimizer': 'adam'
}

# Paths for configuration files
CONFIG_FILE_PATH = os.path.join(BASE_DIR, 'config.json')
LOG_FILE_PATH = os.path.join(LOGS_DIR, 'training.log')

# Model configurations
MODEL_CONFIG = {
    'input_dim': 64,  # Example input dimensions
    'hidden_layers': [128, 64],  # Example hidden layer architecture
    'output_dim': 1,  # For regression, or number of classes for classification
    'activation': 'relu'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',  # Logging level (DEBUG, INFO, WARNING, ERROR)
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}

# Other configurations
RANDOM_SEED = 42  # For reproducibility
TRAINING_SET_SIZE = 0.8  # 80% of data for training, 20% for validation

# API or external data sources (if applicable)
API_KEYS = {
    'weather_api': 'your-weather-api-key',
    'data_source_2': 'your-api-key-here'
}

# Whether to use GPU or CPU for training (set to True/False)
USE_GPU = True

# Add any additional project-specific configuration below
