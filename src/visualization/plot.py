import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_data_distribution(data, feature_names=None, bins=30):
    """
    Plot the distribution of data for given features.

    Args:
        data (pd.DataFrame or np.array): The data to plot (could be a DataFrame or Numpy array).
        feature_names (list of str, optional): The feature names for labeling the axes (default is None).
        bins (int, optional): The number of bins for the histogram (default is 30).
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=feature_names)

    plt.figure(figsize=(10, 6))
    for i, feature in enumerate(data.columns):
        sns.histplot(data[feature], bins=bins, kde=True, label=feature, color=sns.color_palette("husl")[i])
    
    plt.title('Data Distribution')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_feature_importance(importances, feature_names, top_n=10):
    """
    Plot the top N most important features of a model.

    Args:
        importances (np.array): Array of feature importances.
        feature_names (list of str): The feature names.
        top_n (int, optional): Number of top features to plot (default is 10).
    """
    # Sort the features by importance
    sorted_idx = np.argsort(importances)[::-1]

    # Select top N features
    top_idx = sorted_idx[:top_n]
    top_features = [feature_names[i] for i in top_idx]
    top_importances = importances[top_idx]

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.barh(top_features, top_importances, color=sns.color_palette("viridis", len(top_features)))
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

def plot_model_performance(train_scores, val_scores, epochs):
    """
    Plot the performance of a model over epochs (e.g., training and validation scores).

    Args:
        train_scores (list or np.array): The training scores over epochs.
        val_scores (list or np.array): The validation scores over epochs.
        epochs (list or np.array): The list of epoch numbers.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_scores, label='Training', color='blue')
    plt.plot(epochs, val_scores, label='Validation', color='red')
    plt.title('Model Performance Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

def plot_confusion_matrix(conf_matrix, labels, normalize=False):
    """
    Plot a confusion matrix as a heatmap.

    Args:
        conf_matrix (np.array): The confusion matrix (2D array).
        labels (list of str): The class labels.
        normalize (bool, optional): If True, normalize the confusion matrix values (default is False).
    """
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, cbar=False, square=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

def plot_loss_curve(train_loss, val_loss, epochs):
    """
    Plot the loss curves for training and validation.

    Args:
        train_loss (list or np.array): The training loss values over epochs.
        val_loss (list or np.array): The validation loss values over epochs.
        epochs (list or np.array): The list of epoch numbers.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', color='red')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_roc_curve(fpr, tpr, auc_score):
    """
    Plot the ROC curve for binary classification.

    Args:
        fpr (np.array): The false positive rate values.
        tpr (np.array): The true positive rate values.
        auc_score (float): The AUC score for the ROC curve.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
