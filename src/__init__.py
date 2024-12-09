# src/__init__.py

# This file makes the 'src' directory a Python package.
# Any initial setup for the src package can go here, or you can leave it empty.
# You can also import key modules or packages that will be used frequently.

# Example: Importing core modules to simplify the interface
from .data_preprocessing import preprocess
from .models import model, evaluation, utils
from .visualization import plot

# Optional: Provide metadata about the package
__version__ = '1.0'
__author__ = 'Md Rasheduzzaman'
