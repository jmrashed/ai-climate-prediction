import pytest
import matplotlib.pyplot as plt
import pandas as pd
from src.visualization.plot import plot_temperature_over_time, plot_precipitation_histogram
from io import BytesIO

# Example data for visualization
test_data = {
    'date': pd.date_range(start="2023-01-01", periods=5, freq='D'),
    'temperature': [22.5, 24.0, 19.5, 23.0, 25.0],
    'precipitation': [0.1, 0.2, 0.0, 0.3, 0.4]
}

df = pd.DataFrame(test_data)

def test_plot_temperature_over_time():
    """Test the temperature over time plot."""
    # Generate the plot
    fig = plot_temperature_over_time(df)
    
    # Check if the figure is created (it should be a matplotlib figure object)
    assert isinstance(fig, plt.Figure), "Returned object is not a matplotlib Figure"
    
    # Check if the plot contains the expected title and labels
    ax = fig.gca()
    assert ax.get_title() == "Temperature Over Time", "Plot title is incorrect"
    assert ax.get_xlabel() == "Date", "X-axis label is incorrect"
    assert ax.get_ylabel() == "Temperature (°C)", "Y-axis label is incorrect"

def test_plot_precipitation_histogram():
    """Test the precipitation histogram plot."""
    # Generate the plot
    fig = plot_precipitation_histogram(df)
    
    # Check if the figure is created
    assert isinstance(fig, plt.Figure), "Returned object is not a matplotlib Figure"
    
    # Check if the plot contains the expected title and labels
    ax = fig.gca()
    assert ax.get_title() == "Precipitation Histogram", "Plot title is incorrect"
    assert ax.get_xlabel() == "Precipitation (mm)", "X-axis label is incorrect"
    assert ax.get_ylabel() == "Frequency", "Y-axis label is incorrect"

def test_plot_empty_dataframe():
    """Test the behavior when plotting with an empty dataframe."""
    empty_df = pd.DataFrame(columns=['date', 'temperature', 'precipitation'])
    
    # Test if plotting the temperature over time with an empty dataframe raises a ValueError
    with pytest.raises(ValueError, match="Dataframe is empty"):
        plot_temperature_over_time(empty_df)
    
    # Test if plotting the precipitation histogram with an empty dataframe raises a ValueError
    with pytest.raises(ValueError, match="Dataframe is empty"):
        plot_precipitation_histogram(empty_df)

def test_plot_invalid_column():
    """Test the behavior when a dataframe with missing expected columns is passed."""
    invalid_df = pd.DataFrame({
        'date': pd.date_range(start="2023-01-01", periods=5, freq='D'),
        'wrong_column': [10, 20, 30, 40, 50]
    })
    
    # Test if plotting temperature over time with missing 'temperature' column raises an error
    with pytest.raises(KeyError, match="'temperature' column is missing"):
        plot_temperature_over_time(invalid_df)
    
    # Test if plotting precipitation histogram with missing 'precipitation' column raises an error
    with pytest.raises(KeyError, match="'precipitation' column is missing"):
        plot_precipitation_histogram(invalid_df)

def test_plot_saved_to_file():
    """Test if the plot can be saved to a file correctly."""
    fig = plot_temperature_over_time(df)
    
    # Save the figure to a BytesIO object (in-memory file) to simulate saving to disk
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    
    # Check if the file has been saved by checking the file size
    buf_size = len(buf.getvalue())
    assert buf_size > 0, "Saved plot file is empty"
    buf.close()

