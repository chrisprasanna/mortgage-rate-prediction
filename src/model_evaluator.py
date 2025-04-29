"""
Model evaluation module for mortgage rate prediction.
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class MortgageRateModelEvaluator:
    """
    Class for evaluating mortgage rate prediction models.
    """
    def __init__(self, target_col='MORTGAGE30US_diff'):
        """
        Initialize the model evaluator.
        
        Args:
            target_col (str): Target column for prediction
        """
        self.target_col = target_col
        self.models = {}
        self.evaluation_results = {}
        
    def load_models(self, models_dir='models'):
        """
        Load trained models from disk.
        
        Args:
            models_dir (str): Directory containing saved models
        """
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        
        for model_file in model_files:
            model_name = model_file.replace('_pipeline.pkl', '').replace('_', ' ').title()
            model_path = os.path.join(models_dir, model_file)
            
            try:
                model = joblib.load(model_path)
                self.models[model_name] = model
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                
        return self.models
    
    def evaluate_models(self, test_data):
        """
        Evaluate models on test data.
        
        Args:
            test_data (pd.DataFrame): Test data
            
        Returns:
            pd.DataFrame: Evaluation metrics
        """
        if 'Date' in test_data.columns:
            X_test = test_data.drop(columns=[self.target_col, 'Date'])
        else:
            X_test = test_data.drop(columns=[self.target_col])
            
        y_test = test_data[self.target_col]
        
        results = {}
        
        for model_name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[model_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            
            print(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            
        # Create DataFrame with results
        results_df = pd.DataFrame(results).T
        self.evaluation_results = results_df
        
        return results_df
    
    def plot_actual_vs_predicted(self, test_data, model_name=None, save_path=None):
        """
        Create a scatter plot of actual vs predicted values.
        
        Args:
            test_data (pd.DataFrame): Test data
            model_name (str): Name of model to evaluate (uses first model if None)
            save_path (str): Path to save plot
        """
        if not self.models:
            print("No models loaded. Use load_models() first.")
            return
            
        # Use specified model or first one
        if model_name is None:
            model_name = list(self.models.keys())[0]
            
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return
            
        model = self.models[model_name]
        
        # Prepare data
        if 'Date' in test_data.columns:
            X_test = test_data.drop(columns=[self.target_col, 'Date'])
        else:
            X_test = test_data.drop(columns=[self.target_col])
            
        y_test = test_data[self.target_col]
        y_pred = model.predict(X_test)
        
        # Create scatter plot
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
        plt.title(f'Actual vs Predicted: {model_name}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
            
    def plot_residuals(self, test_data, model_name=None, save_path=None):
        """
        Plot residuals.
        
        Args:
            test_data (pd.DataFrame): Test data
            model_name (str): Name of model to evaluate (uses first model if None)
            save_path (str): Path to save plot
        """
        if not self.models:
            print("No models loaded. Use load_models() first.")
            return
            
        # Use specified model or first one
        if model_name is None:
            model_name = list(self.models.keys())[0]
            
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return
            
        model = self.models[model_name]
        
        # Prepare data
        if 'Date' in test_data.columns:
            X_test = test_data.drop(columns=[self.target_col, 'Date'])
            dates = test_data['Date']
        else:
            X_test = test_data.drop(columns=[self.target_col])
            dates = pd.Series(range(len(test_data)))
            
        y_test = test_data[self.target_col]
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Residuals over time
        ax1.plot(dates, residuals, 'o', alpha=0.7)
        ax1.axhline(y=0, color='r', linestyle='-')
        ax1.set_title('Residuals Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Residual')
        ax1.grid(True)
        
        # Residual distribution
        sns.histplot(residuals, kde=True, ax=ax2)
        ax2.set_title('Residual Distribution')
        ax2.set_xlabel('Residual')
        ax2.set_ylabel('Frequency')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
            
    def plot_feature_importance(self, test_data, model_name=None, save_path=None):
        """
        Plot feature importance for linear models.
        
        Args:
            test_data (pd.DataFrame): Test data (for feature names)
            model_name (str): Name of model to evaluate (uses first model if None)
            save_path (str): Path to save plot
        """
        if not self.models:
            print("No models loaded. Use load_models() first.")
            return
            
        # Use specified model or first one
        if model_name is None:
            model_name = list(self.models.keys())[0]
            
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return
            
        model = self.models[model_name]
        
        # Extract model from pipeline
        model_step = model.named_steps['model']
        
        # Check if model has coefficients
        if not hasattr(model_step, 'coef_'):
            print(f"Model {model_name} does not have feature coefficients.")
            return
            
        # Get feature names
        if 'Date' in test_data.columns:
            feature_names = test_data.drop(columns=[self.target_col, 'Date']).columns
        else:
            feature_names = test_data.drop(columns=[self.target_col]).columns
            
        # Get coefficients
        coefficients = model_step.coef_
        
        # Create DataFrame for plotting
        coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
        coef_df = coef_df.sort_values('Coefficient', ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(coef_df['Feature'], coef_df['Coefficient'])
        plt.title(f'Feature Importance: {model_name}')
        plt.xlabel('Coefficient')
        plt.ylabel('Feature')
        plt.grid(True, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def plot_predictions_over_time(self, df, rate_col='MORTGAGE30US', change_col='MORTGAGE30US_diff', 
                                 predictions=None, model_name=None, save_path=None):
        """
        Create a time series plot comparing actual and predicted mortgage rates.
        
        Args:
            df (pd.DataFrame): DataFrame containing dates, rates and rate changes
            rate_col (str): Name of column containing actual mortgage rates
            change_col (str): Name of column containing rate changes
            predictions (array-like, optional): Predicted rate changes. If None, will generate new predictions
            model_name (str): Name of model to evaluate (uses first model if None)
            save_path (str): Path to save plot
        """
        if not self.models and predictions is None:
            print("No models loaded and no predictions provided.")
            return
            
        if 'Date' not in df.columns or rate_col not in df.columns:
            print(f"DataFrame must include 'Date' and '{rate_col}' columns.")
            return
            
        dates = df['Date']
        actual_rates = df[rate_col]
        
        # Get predictions if not provided
        if predictions is None:
            if model_name is None:
                model_name = list(self.models.keys())[0]
                
            if model_name not in self.models:
                print(f"Model {model_name} not found.")
                return
                
            model = self.models[model_name]
            X = df.drop(columns=[change_col, 'Date'])
            predictions = model.predict(X)
        
        # Convert predicted changes to predicted rates
        # Shift actual rates forward by one week since predictions are for next week
        predicted_rates = actual_rates.shift(1) + predictions
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot actual and predicted rates
        plt.plot(dates, actual_rates, 'b-', label='Actual Rate', alpha=0.7)
        plt.plot(dates, predicted_rates, 'r--', label='Predicted Rate', alpha=0.7)
        
        title = '30-Year Fixed Mortgage Rates: Actual vs Predicted'
        if model_name:
            title += f'\n{model_name}'
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Mortgage Rate (%)')
        plt.legend()
        plt.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add text with error metrics
        valid_mask = ~np.isnan(predicted_rates)
        rmse = np.sqrt(mean_squared_error(actual_rates[valid_mask], predicted_rates[valid_mask]))
        mae = mean_absolute_error(actual_rates[valid_mask], predicted_rates[valid_mask])
        
        stats_text = (
            f'RMSE: {rmse:.4f}\n'
            f'MAE: {mae:.4f}'
        )
        plt.text(0.02, 0.98, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()