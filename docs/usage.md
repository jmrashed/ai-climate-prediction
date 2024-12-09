# Usage Guide for AI Climate Prediction

This document outlines how to use the **AI Climate Prediction** project, including how to load data, preprocess it, train machine learning models, evaluate them, generate predictions, and visualize the results.

## Table of Contents

1. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
2. [Training a Model](#training-a-model)
3. [Evaluating a Model](#evaluating-a-model)
4. [Generating Predictions](#generating-predictions)
5. [Visualizing Results](#visualizing-results)

---

## Data Loading and Preprocessing

### Step 1: Load Raw Data

Raw climate data should be stored in the `data/raw` directory. To load the data, simply use the preprocessing module in `src/data_preprocessing/preprocess.py`.

#### Example:

```python
from src.data_preprocessing.preprocess import load_raw_data

# Load the raw data
raw_data = load_raw_data("data/raw/climate_data.csv")

# Check the first few rows
print(raw_data.head())
```

### Step 2: Clean and Preprocess the Data

Once the data is loaded, you can clean and preprocess it (e.g., removing missing values, scaling, feature engineering) using the preprocessing functions.

#### Example:

```python
from src.data_preprocessing.preprocess import clean_data, preprocess_data

# Clean the data (remove or impute missing values, remove outliers, etc.)
cleaned_data = clean_data(raw_data)

# Preprocess the data (e.g., feature scaling, encoding)
processed_data = preprocess_data(cleaned_data)

# Save processed data to the `data/processed` folder
processed_data.to_csv("data/processed/cleaned_climate_data.csv", index=False)
```

---

## Training a Model

Once the data is processed, you can train a machine learning model. This project uses several models, including regression and classification algorithms.

### Step 1: Define the Model

You can define the model in the `src/models/model.py` file. For example, let’s use a simple `RandomForestRegressor`.

#### Example:

```python
from src.models.model import train_model

# Specify the preprocessed data for training
X_train, y_train = processed_data.drop('target', axis=1), processed_data['target']

# Train the model
model = train_model(X_train, y_train)

# Save the trained model to the `outputs/model` directory
model.save("outputs/model/random_forest_model.pkl")
```

### Step 2: Hyperparameter Tuning (Optional)

If you want to tune hyperparameters for better performance, you can adjust the model’s parameters before training.

Example of tuning hyperparameters using `GridSearchCV` (if applicable):

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None]
}

# Define the model and GridSearchCV
model = RandomForestRegressor()
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

# Train the model with hyperparameter tuning
grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
best_model = grid_search.best_estimator_

# Save the best model
best_model.save("outputs/model/tuned_random_forest_model.pkl")
```

---

## Evaluating a Model

After training, you can evaluate the model's performance on a test dataset.

### Step 1: Evaluate the Model

To evaluate the model, use the evaluation functions in the `src/models/evaluation.py` file.

#### Example:

```python
from src.models.evaluation import evaluate_model

# Load the test data
X_test, y_test = test_data.drop('target', axis=1), test_data['target']

# Evaluate the model
evaluation_results = evaluate_model(model, X_test, y_test)

# Print the evaluation results
print(evaluation_results)
```

### Step 2: Metrics

You can get common metrics like RMSE, MAE, R-squared, etc.

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Example of computing metrics
y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")
```

---

## Generating Predictions

Once the model is trained and evaluated, you can use it to generate predictions.

### Step 1: Generate Predictions

Use the `scripts/generate_predictions.py` script to generate predictions on new or unseen data.

#### Example:

```python
from src.models.model import load_model

# Load the trained model
model = load_model("outputs/model/random_forest_model.pkl")

# Generate predictions
new_data = load_data("data/new_climate_data.csv")
predictions = model.predict(new_data)

# Save predictions to a file
predictions.to_csv("outputs/results/predictions.csv", index=False)
```

### Step 2: Post-Processing (Optional)

You can post-process the predictions to ensure that they match the desired output format, such as converting them to a CSV or plotting them for visualization.

---

## Visualizing Results

Use the `src/visualization/plot.py` file to create visualizations of your results.

### Step 1: Visualize Model Performance

You can visualize the model's performance using plots like scatter plots, residual plots, or error distributions.

#### Example:

```python
import matplotlib.pyplot as plt
from src.visualization.plot import plot_residuals

# Plot the residuals
plot_residuals(y_test, y_pred)

# Show the plot
plt.show()
```

### Step 2: Visualize Climate Trends

You can also visualize climate trends over time or analyze how different climate variables are correlated.

```python
from src.visualization.plot import plot_climate_trends

# Plot climate trends for a specific variable
plot_climate_trends(processed_data, "temperature")

# Show the plot
plt.show()
```

---

## Conclusion

This guide provides a basic usage overview for the **AI Climate Prediction** project. You can preprocess data, train and evaluate models, generate predictions, and visualize results using the provided scripts and functions.

If you encounter any issues or have further questions, please refer to the [README.md](README.md) or feel free to open an issue on the GitHub repository.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 