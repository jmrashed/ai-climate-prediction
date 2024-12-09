# src/visualization/__init__.py

# This file makes the 'visualization' directory a Python package
# You can add any initial setup or default imports here if necessary

from .plot import plot_data_distribution, plot_feature_importance, plot_model_performance

__all__ = [
    'plot_data_distribution',
    'plot_feature_importance',
    'plot_model_performance',
]

# You can also add any version information or metadata about the module
__version__ = '1.0'
