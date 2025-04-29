"""
Mortgage Rate Prediction Package
"""
from .data_processor import MortgageDataProcessor
from .model_trainer import MortgageRateModelTrainer
from .model_evaluator import MortgageRateModelEvaluator
from .utils import setup_directories, plot_time_series
from . import config

__version__ = "0.1.0"
__author__ = "Chris Prasanna"

# Export main classes and functions
__all__ = [
    'MortgageDataProcessor',
    'MortgageRateModelTrainer',
    'MortgageRateModelEvaluator',
    'setup_directories',
    'plot_time_series',
    'config'
]