"""
Data collection and processing module for mortgage rate prediction.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fredapi import Fred
import yfinance as yf
import time
import os

class MortgageDataProcessor:
    """
    Class for fetching, processing, and preparing mortgage rate data.
    """
    def __init__(self, fred_api_key=None, start_date="1971-04-02"):
        """
        Initialize the data processor.
        
        Args:
            fred_api_key (str): API key for FRED data
            start_date (str): Start date for data collection in YYYY-MM-DD format
        """
        self.start_date = start_date
        
        # Try to load API key from environment or file
        if fred_api_key is None:
            fred_api_key = os.environ.get('FRED_API_KEY')
            
            if fred_api_key is None and os.path.exists('fred_token.txt'):
                with open('fred_token.txt', 'r') as file:
                    fred_api_key = file.read().strip()
        
        if fred_api_key:
            self.fred = Fred(api_key=fred_api_key)
        else:
            self.fred = None
            print("WARNING: No FRED API key provided. FRED data collection will not work.")
            
        # Define FRED series IDs and their descriptions
        self.series_ids = {
            "MORTGAGE30US": "30-Year Fixed Mortgage Rate",
            "DGS10": "10-Year Treasury Yield",
            "FEDFUNDS": "Federal Funds Rate",
            "CPIAUCSL": "Consumer Price Index (CPI)",
            "UNRATE": "Unemployment Rate",
            "GDP": "Gross Domestic Product (GDP)",
            "HOUST": "New Residential Construction",
            "PERMIT": "Building Permits",
            'DFF': "Discount Rate",
            'M2SL': "Money Supply M2",
            'M1SL': "Money Supply M1",
            'M1V': "Money Velocity M1",
            'M2V': "Money Velocity M2",
            'PAYEMS': "All Employees: Total Nonfarm Payrolls",
            'CIVPART': "Civilian Labor Force Participation Rate",
            "PCEPI": "Personal Consumption Expenditures Price Index",
            "CPILFESL": "CPI for All Urban Consumers: Food",
            "PPIACO": "Producer Price Index for All Commodities",
            "MSPUS": "Median Sales Price of Houses Sold for the United States",
            "GDPC1": "Real Gross Domestic Product",
            "GNPCA": "Real Gross National Product",
            "A939RC0Q052SBEA": "Federal Government Current Expenditures",
            "EXPGS": "Exports of Goods and Services",
            "TOTRESNS": "Total Reserves of Depository Institutions",
            "BUSLOANS": "Commercial and Industrial Loans",
            "UMCSENT": "University of Michigan Consumer Sentiment",
            "USEPUINDXD": "Economic Policy Uncertainty Index for the United States",
        }
        
        # Yahoo Finance tickers
        self.yahoo_tickers = [
            '^GSPC',  # S&P 500
            '^VIX',   # VIX
            'FNMA',   # Fannie Mae
        ]

    def fetch_fred_data(self, series_id):
        """
        Fetch data from FRED for a given series ID.
        
        Args:
            series_id (str): FRED series ID
            
        Returns:
            pd.DataFrame: DataFrame with the series data
        """
        if self.fred is None:
            print(f"Cannot fetch {series_id} - No FRED API key provided")
            return None
        
        try:
            data = self.fred.get_series(series_id, observation_start=self.start_date)
            df = pd.DataFrame(data, columns=[series_id])
            df.index.name = "Date"
            return df
        except Exception as e:
            print(f"Error fetching data for {series_id}: {e}")
            return None

    def fetch_all_fred_data(self):
        """
        Fetch all FRED data series.
        
        Returns:
            dict: Dictionary of DataFrames with series data
        """
        return {name: self.fetch_fred_data(code) for code, name in self.series_ids.items()}

    def fetch_yahoo_data(self):
        """
        Fetch data from Yahoo Finance.
        
        Returns:
            dict: Dictionary with series data
        """
        try:
            yahoo_data = yf.download(self.yahoo_tickers, start=self.start_date, group_by='ticker')
            
            # Process and flatten data
            if isinstance(yahoo_data.columns, pd.MultiIndex):
                flat_data = yahoo_data.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
                flat_data = flat_data.pivot(index='Date', columns='Ticker')
                flat_data.columns = ['{}_{}'.format(ticker, col) for col, ticker in flat_data.columns]
                flat_data.index = pd.to_datetime(flat_data.index)
            else:
                flat_data = yahoo_data.copy()
            
            # Only keep Close prices
            flat_data = flat_data.filter(like='Close')
            # Rename columns for clarity
            flat_data.columns = [col.replace('Close_', '') for col in flat_data.columns]
            
            return flat_data.to_dict(orient='series')
        except Exception as e:
            print(f"Error fetching Yahoo Finance data: {e}")
            return {}

    def merge_all_data(self, data_frames):
        """
        Merge all data into a single DataFrame.
        
        Args:
            data_frames (dict): Dictionary of DataFrames to merge
            
        Returns:
            pd.DataFrame: Combined DataFrame
        """
        # Combine all dataframes
        combined = pd.concat([df for df in data_frames.values() if df is not None], axis=1)
        
        # Resample to weekly frequency
        combined = combined.resample("W").mean()
        
        # Add mortgage rate difference (one week into the future)
        if 'MORTGAGE30US' in combined.columns:
            combined['MORTGAGE30US_diff'] = combined['MORTGAGE30US'].shift(-1) - combined['MORTGAGE30US']
        
        return combined

    def preprocess_data(self, df, target_col='MORTGAGE30US_diff'):
        """
        Preprocess data for modeling.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_col (str): Target column name
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        # Fill missing values
        df.ffill(inplace=True)
        
        # Create lag features for mortgage rates
        if 'MORTGAGE30US' in df.columns:
            for lag in [1, 2, 4, 8]:
                df[f"MORTGAGE30US_{lag}"] = df["MORTGAGE30US"].shift(lag)
        
        # Create date features
        df = df.reset_index()
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['weekofyear'] = df['Date'].dt.isocalendar().week
        
        # Create rolling features
        if target_col in df.columns:
            df['rolling_mean_3'] = df[target_col].rolling(window=3).mean()
            df['rolling_std_3'] = df[target_col].rolling(window=3).std()
        
        # Fill NaN values from feature creation
        df.fillna(0, inplace=True)
        
        return df

    def fetch_and_process_data(self):
        """
        Fetch and process all data in one go.
        
        Returns:
            pd.DataFrame: Final processed DataFrame
        """
        # Fetch FRED data
        fred_data = self.fetch_all_fred_data()
        
        # Fetch Yahoo Finance data
        yahoo_data = self.fetch_yahoo_data()
        
        # Combine all data sources
        all_data = {**fred_data, **yahoo_data}
        
        # Merge all data into a single DataFrame
        combined_df = self.merge_all_data(all_data)
        
        # Preprocess data
        processed_df = self.preprocess_data(combined_df)
        
        return processed_df

    def save_data(self, df, output_path='data/full_mortgage_dataset.csv'):
        """
        Save processed data to CSV.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            output_path (str): Path to save the CSV file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")