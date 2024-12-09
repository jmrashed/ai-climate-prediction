from setuptools import setup, find_packages

# Read the contents of your README file for long description
with open("README.md", "r") as fh:
    long_description = fh.read()

# Define the setup
setup(
    name="ai-climate-prediction",  # Name of the project
    version="0.1.0",               # Version number
    author="Your Name",            # Author name
    author_email="jmrashed@example.com",  # Author email
    description="AI-based Climate Prediction using Machine Learning",  # Short description of the project
    long_description=long_description,  # Long description from README.md
    long_description_content_type="text/markdown",  # Content type of the README
    url="https://github.com/jmrashed/ai-climate-prediction",  # Project URL (e.g., GitHub link)
    packages=find_packages(),  # Automatically find and include all packages
    classifiers=[  # Classifiers to categorize the project (optional)
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum Python version required
    install_requires=[  # Dependencies for the project
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tensorflow>=2.5.0",  # Or pytorch, depending on the model framework
        "jupyter",  # For Jupyter notebooks
        "requests",  # For API data collection (if applicable)
        "pytest",  # For testing purposes
    ],
    extras_require={  # Optional dependencies (e.g., for development)
        "dev": [
            "black",  # Code formatting
            "flake8",  # Linting
            "pre-commit",  # Git hooks
            "tox",  # Testing framework
            "pytest-cov",  # For coverage reporting in tests
        ],
        "docs": [
            "sphinx",  # Documentation generator
            "sphinx_rtd_theme",  # Read-the-docs theme
        ]
    },
    entry_points={  # Optional, if you plan to create command line tools
        "console_scripts": [
            "train-model=src.scripts.train_model:main",
            "evaluate-model=src.scripts.evaluate_model:main",
        ]
    },
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
    zip_safe=False,  # To avoid using zips if you have compiled extensions
)
