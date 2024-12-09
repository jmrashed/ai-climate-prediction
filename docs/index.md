# AI Climate Prediction

Welcome to the **AI Climate Prediction** project! 🌍 This project aims to utilize advanced machine learning models and data analytics to predict and analyze climate-related data, with a focus on sustainability and climate action.

## Overview

The **AI Climate Prediction** project focuses on using artificial intelligence (AI) and data analysis techniques to predict climate patterns and variables. By leveraging historical climate data and machine learning models, this project aims to provide actionable insights that can help guide sustainable decision-making in climate science and environmental conservation.

### Key Features:
- **Data Analysis & Preprocessing**: Clean and preprocess raw climate data for analysis.
- **Modeling & Predictions**: Train and test machine learning models to predict climate outcomes (e.g., temperature, rainfall, air quality).
- **Visualization**: Visualize model performance and climate trends.
- **Sustainability Focus**: Use AI to drive actionable insights for sustainable solutions and climate change mitigation.

## Getting Started

Follow these steps to set up the project and start contributing:

### Prerequisites

- Python 3.7+
- Conda (recommended for managing environments)
- Git (for version control)
- Docker (optional, for containerization)

### Installation

#### 1. Clone the repository

```bash
git clone https://github.com/jmrashed/ai-climate-prediction.git
cd ai-climate-prediction
```

#### 2. Create a virtual environment using Conda (or use your preferred environment manager)

```bash
conda env create -f environment.yml
```

Alternatively, you can use `pip` and `requirements.txt` if you're not using Conda.

```bash
pip install -r requirements.txt
```

#### 3. Activate the environment

```bash
conda activate ai-climate-prediction
```

### Running the Project

1. **Data Preparation**:
   - The raw climate data can be found in the `data/raw` folder.
   - Preprocessed data will be saved in the `data/processed` folder after cleaning.

2. **Model Training**:
   - You can train machine learning models by running the training script or using the provided Jupyter notebooks.
   - The notebooks for training models can be found under `notebooks/model_training/`.

   Example to train a model:

   ```bash
   python scripts/train_model.py
   ```

3. **Model Evaluation**:
   - Evaluate the trained models using the provided `scripts/evaluate_model.py`.

   ```bash
   python scripts/evaluate_model.py
   ```

4. **Generating Predictions**:
   - After training, use the `scripts/generate_predictions.py` script to generate predictions from the trained model.

   ```bash
   python scripts/generate_predictions.py
   ```

### Visualizations

The `notebooks/exploratory_data_analysis/eda.ipynb` contains various data exploration techniques, while the `src/visualization/plot.py` contains functions for visualizing model predictions and climate trends.

```bash
# Run the Jupyter notebook for data exploration
jupyter notebook notebooks/exploratory_data_analysis/eda.ipynb
```

### Docker Setup (Optional)

If you prefer to run the project inside a Docker container, you can build and run the Docker image.

```bash
docker build -t ai-climate-prediction .
docker run -p 8888:8888 ai-climate-prediction
```

This will start a Jupyter notebook server within a Docker container, accessible at `http://localhost:8888`.

## Contributing

We welcome contributions to improve this project! Here are some ways you can contribute:

- **Bug Fixes**: Help identify and resolve bugs.
- **Feature Requests**: Suggest new features or improvements.
- **Documentation**: Update or enhance documentation.
- **Model Improvements**: Add more machine learning models or improve current models.

Please fork the repository, create a branch, and submit a pull request (PR) with your proposed changes. Be sure to follow the existing code style and include tests or documentation for any changes.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Acknowledgments

- **Data**: The data used for this project is sourced from [Climate Data Source] (insert your data source).
- **Machine Learning Models**: We utilize popular libraries such as `scikit-learn`, `XGBoost`, and `TensorFlow` for model development.
- **Visualization**: Visualizations are created using libraries like `Matplotlib`, `Seaborn`, and `Plotly`.

For any questions or feedback, feel free to open an issue or contact the repository owner at [jmrashed@gmail.com](mailto:jmrashed@gmail.com).

---
Let's use AI to build a sustainable future! 🌱 