"""
Utility functions for the mortgage rate prediction project.
"""
import os
import matplotlib.pyplot as plt


def setup_directories(dirs):
    """
    Create directories if they don't exist.
    
    Args:
        dirs (list): List of directory paths to create
    """
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created/verified: {directory}")

def plot_time_series(df, columns, title='Time Series Plot', figsize=(12, 6), date_column='Date', save_path=None):
    """
    Plot time series data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to plot
        title (str): Plot title
        figsize (tuple): Figure size
        date_column (str): Name of date column
        save_path (str): Path to save plot
    """
    plt.figure(figsize=figsize)
    
    if date_column in df.columns:
        x = df[date_column]
    else:
        x = df.index
    
    for column in columns:
        if column in df.columns:
            plt.plot(x, df[column], label=column)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
