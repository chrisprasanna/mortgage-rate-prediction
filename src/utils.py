"""
Utility functions for the mortgage rate prediction project.
"""
import os
import matplotlib.pyplot as plt
import pandas as pd

def setup_directories(dirs):
    """
    Create directories if they don't exist.
    
    Args:
        dirs (list): List of directory paths to create
    """
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created/verified: {directory}")
