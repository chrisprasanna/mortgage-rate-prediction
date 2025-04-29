"""
Model training module for mortgage rate prediction.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class MortgageRateModelTrainer:
    """
    Class for training and evaluating mortgage rate prediction models.
    """
    def __init__(self, target_col='MORTGAGE30US_diff'):
        """
        Initialize the model trainer.
        
        Args:
            target_col (str): Target column for prediction
        """
        self.target_col = target_col
        self.models = {}
        self.results = {}
        
    def load_data(self, data_path):
        """
        Load data from CSV file.
        
        Args:
            data_path (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        df = pd.read_csv(data_path, parse_dates=['Date'])
        
        # Apply basic preprocessing
        df = self.preprocess_data(df)
        
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess data for modeling.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        # Drop target variable NaN values
        df.dropna(subset=[self.target_col], inplace=True)
        
        # Convert everything to float32
        for col in df.columns:
            if col == 'Date':
                continue
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype('float32')
                
        # Fill any remaining NaN values
        df.fillna(0, inplace=True)
        
        return df
    
    def create_pipeline(self, model):
        """
        Create a scikit-learn pipeline with preprocessing and model.
        
        Args:
            model: Scikit-learn model
            
        Returns:
            Pipeline: Scikit-learn pipeline
        """
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        return pipeline
    
    def train_and_evaluate(self, df, n_splits=5):
        """
        Train and evaluate models using K-Fold cross-validation.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            n_splits (int): Number of folds for cross-validation
            
        Returns:
            tuple: (models, results)
        """
        # Split data into features and target
        if 'Date' in df.columns:
            X = df.drop(columns=[self.target_col, 'Date'])
        else:
            X = df.drop(columns=[self.target_col])
            
        y = df[self.target_col]
        
        # Define models
        bayesian_model = self.create_pipeline(BayesianRidge())
        linear_model = self.create_pipeline(LinearRegression())
        
        models = {
            "Bayesian Ridge Regression": bayesian_model,
            "Linear Regression": linear_model
        }
        
        # Perform K-Fold Cross-Validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        results = {}
        
        for model_name, model in models.items():
            cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-cv_scores)
            results[model_name] = {
                "Mean RMSE": rmse_scores.mean(),
                "Std RMSE": rmse_scores.std()
            }
            print(f"{model_name} - Mean RMSE: {rmse_scores.mean():.4f}, Std RMSE: {rmse_scores.std():.4f}")
        
        # Train final models on the full dataset
        for model_name, model in models.items():
            model.fit(X, y)
            
        self.models = models
        self.results = results
        
        return models, results
    
    def save_models(self, output_dir='models'):
        """
        Save trained models to disk.
        
        Args:
            output_dir (str): Directory to save models
        """
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            file_name = f"{model_name.lower().replace(' ', '_')}_pipeline.pkl"
            file_path = os.path.join(output_dir, file_name)
            joblib.dump(model, file_path)
            print(f"Model saved to {file_path}")
        
        # Save results as well
        if self.results:
            results_df = pd.DataFrame({k: [v["Mean RMSE"], v["Std RMSE"]] for k, v in self.results.items()},
                                     index=["Mean RMSE", "Std RMSE"])
            results_df.to_csv(os.path.join(output_dir, "model_results.csv"))