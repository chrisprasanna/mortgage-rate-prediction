"""
End-to-end script for mortgage rate prediction pipeline.
"""
import os
import sys
import pandas as pd
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import project modules
from data_processor import MortgageDataProcessor
from model_trainer import MortgageRateModelTrainer
from model_evaluator import MortgageRateModelEvaluator
from utils import setup_directories
import config
from sklearn.model_selection import train_test_split

def main():
    # Create required directories
    setup_directories([config.DATA_DIR, config.MODELS_DIR, config.RESULTS_DIR, config.PLOTS_DIR])
    
    print("=" * 80)
    print("Starting Mortgage Rate Prediction Pipeline")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 1. Data Collection
    print("\n1. Collecting Data...")
    processor = MortgageDataProcessor(start_date=config.START_DATE)
    data = processor.fetch_and_process_data()
    processor.save_data(data, config.PROCESSED_DATA_PATH)
    print(f"Data shape: {data.shape}")
    
    # 2. Train-Test Split
    print("\n2. Splitting Data...")
    train_df, test_df = train_test_split(data, test_size=config.TEST_SIZE, random_state=config.SEED)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    # 3. Model Training
    print("\n3. Training Models...")
    trainer = MortgageRateModelTrainer(target_col=config.TARGET_COLUMN)
    models, results = trainer.train_and_evaluate(train_df, n_splits=config.CV_FOLDS)
    trainer.save_models(config.MODELS_DIR)
    
    # 4. Model Evaluation
    print("\n4. Evaluating Models...")
    evaluator = MortgageRateModelEvaluator(target_col=config.TARGET_COLUMN)
    evaluator.models = models  # Use models directly instead of loading from disk
    
    # Generate evaluation plots
    print("\nGenerating evaluation plots...")
    
    # Actual vs Predicted plot
    evaluator.plot_actual_vs_predicted(
        test_df,
        model_name="Bayesian Ridge Regression",
        save_path=os.path.join(config.PLOTS_DIR, "actual_vs_predicted.png")
    )
    
    # Residuals plot
    evaluator.plot_residuals(
        test_df,
        model_name="Bayesian Ridge Regression",
        save_path=os.path.join(config.PLOTS_DIR, "residuals.png")
    )
    
    # Feature importance plot
    evaluator.plot_feature_importance(
        test_df,
        model_name="Bayesian Ridge Regression",
        save_path=os.path.join(config.PLOTS_DIR, "feature_importance.png")
    )

    # Time series plot
    # Filter data for the most recent 10 years
    recent_data = data[data['Date'] >= (datetime.now() - pd.DateOffset(years=10)).strftime('%Y-%m-%d')]
    
    evaluator.plot_predictions_over_time(
        df=recent_data,
        rate_col='MORTGAGE30US',
        change_col='MORTGAGE30US_diff',
        model_name="Bayesian Ridge Regression",
        save_path=os.path.join(config.PLOTS_DIR, "predictions_over_last_10yrs.png")
    )

    # Generate next week's prediction
    latest_data = data.iloc[-1:]  # Get most recent data point
    next_week_pred = models["Bayesian Ridge Regression"].predict(
        latest_data.drop(columns=[config.TARGET_COLUMN, 'Date'])
    )[0]
    current_rate = latest_data['MORTGAGE30US'].values[0]
    
    # Save prediction results
    prediction_results = {
        'current_rate': current_rate,
        'predicted_change': next_week_pred,
        'predicted_rate': current_rate + next_week_pred,
        'prediction_date': (datetime.now() + pd.DateOffset(weeks=1)).strftime('%Y-%m-%d')
    }
    
    pd.DataFrame([prediction_results]).to_csv(
        os.path.join(config.RESULTS_DIR, "next_week_prediction.csv"),
        index=False
    )

    print(f"\nNext week's predicted mortgage rate: {prediction_results['predicted_rate']}")
    
    # Calculate and save evaluation metrics
    metrics_df = evaluator.evaluate_models(test_df)
    metrics_df.to_csv(os.path.join(config.RESULTS_DIR, "evaluation_metrics.csv"))
    
    print("\nPipeline completed successfully!")
    print(f"Results saved in {config.RESULTS_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()