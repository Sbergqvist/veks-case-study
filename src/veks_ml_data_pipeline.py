"""
VEKS ML Data Pipeline
Author: Sebastian Brydensholt
Created for VEKS Data Engineer Position

Dedicated data pipeline for machine learning models.
Pulls more historical data, cleans thoroughly, and prepares ML-ready datasets.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VEKSMLDataPipeline:
    """
    Dedicated data pipeline for VEKS machine learning models.
    
    Features:
    - Pulls 180 days of historical data for robust ML training
    - Thorough data cleaning and validation
    - ML-specific feature engineering
    - Separate storage from Power BI data
    """
    
    def __init__(self):
        self.base_url = "https://api.energidataservice.dk/dataset"
        self.output_dir = Path("data/veks_ml_data")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # VEKS regions
        self.veks_regions = ['DK1', 'DK2']
        
    def fetch_ml_data(self, days=180):
        """
        Fetch extended historical data for ML training.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Fetching {days} days of data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        url = f"{self.base_url}/elspotprices"
        params = {
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d"),
            "format": "json"
        }
        
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            # Filter for VEKS regions
            all_records = data.get('records', [])
            veks_records = [r for r in all_records if r.get('PriceArea') in self.veks_regions]
            
            logger.info(f"Retrieved {len(veks_records)} records for DK1/DK2")
            return veks_records
            
        except requests.RequestException as e:
            logger.error(f"API call failed: {e}")
            return []
    
    def clean_data_for_ml(self, raw_data):
        """
        Thorough data cleaning specifically for ML models.
        """
        if not raw_data:
            logger.warning("No data to clean")
            return pd.DataFrame()
        
        df = pd.DataFrame(raw_data)
        
        # Basic data cleaning
        df['HourDK'] = pd.to_datetime(df['HourDK'])
        df['SpotPriceDKK'] = pd.to_numeric(df['SpotPriceDKK'], errors='coerce')
        df['SpotPriceEUR'] = pd.to_numeric(df['SpotPriceEUR'], errors='coerce')
        
        # Remove rows with missing prices
        df = df.dropna(subset=['SpotPriceDKK', 'SpotPriceEUR'])
        
        # Handle extreme values (likely data errors)
        df = self._handle_extreme_values(df)
        
        # Add time-based features
        df = self._add_time_features(df)
        
        # Add lag and rolling features
        df = self._add_ml_features(df)
        
        # Final validation
        df = self._validate_ml_data(df)
        
        logger.info(f"Cleaned data: {len(df)} records, {len(df.columns)} features")
        return df
    
    def _handle_extreme_values(self, df):
        """
        Handle extreme price values that are likely data errors.
        """
        # Remove negative prices (except for rare valid cases)
        df = df[df['SpotPriceDKK'] >= -50]  # Allow small negative prices
        
        # Cap extremely high prices (likely data errors)
        for col in ['SpotPriceDKK', 'SpotPriceEUR']:
            q99 = df[col].quantile(0.99)
            q01 = df[col].quantile(0.01)
            
            # Cap at 99th percentile * 1.5
            df.loc[df[col] > q99 * 1.5, col] = q99 * 1.5
            
            # Floor at 1st percentile * 0.5
            df.loc[df[col] < q01 * 0.5, col] = q01 * 0.5
        
        return df
    
    def _add_time_features(self, df):
        """
        Add comprehensive time-based features.
        """
        # Basic time features
        df['Date'] = df['HourDK'].dt.date
        df['Hour'] = df['HourDK'].dt.hour
        df['DayOfWeek'] = df['HourDK'].dt.day_name()
        df['IsWeekend'] = df['HourDK'].dt.weekday >= 5
        df['Month'] = df['HourDK'].dt.month
        df['DayOfYear'] = df['HourDK'].dt.dayofyear
        
        # Cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['HourDK'].dt.dayofweek / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['HourDK'].dt.dayofweek / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        
        # Seasonal features
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                     9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
        df['Season'] = df['Month'].map(season_map)
        
        # Business logic features
        df['HeatingDemand'] = df['Hour'].apply(self._get_demand_category)
        df['CHPOptimal'] = df['Hour'].apply(self._is_chp_optimal_hour)
        
        return df
    
    def _add_ml_features(self, df):
        """
        Add lag and rolling features for ML models.
        """
        # Sort by time and region
        df = df.sort_values(['PriceArea', 'HourDK']).reset_index(drop=True)
        
        # Lag features
        for lag in [1, 2, 3, 6, 12, 24, 48, 72]:
            df[f'price_lag_{lag}'] = df.groupby('PriceArea')['SpotPriceDKK'].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12, 24, 48, 72]:
            df[f'price_rolling_mean_{window}'] = df.groupby('PriceArea')['SpotPriceDKK'].rolling(window, min_periods=1).mean().reset_index(0, drop=True)
            df[f'price_rolling_std_{window}'] = df.groupby('PriceArea')['SpotPriceDKK'].rolling(window, min_periods=1).std().reset_index(0, drop=True)
            df[f'price_rolling_min_{window}'] = df.groupby('PriceArea')['SpotPriceDKK'].rolling(window, min_periods=1).min().reset_index(0, drop=True)
            df[f'price_rolling_max_{window}'] = df.groupby('PriceArea')['SpotPriceDKK'].rolling(window, min_periods=1).max().reset_index(0, drop=True)
        
        # Price change features
        df['price_change'] = df.groupby('PriceArea')['SpotPriceDKK'].diff()
        df['price_change_pct'] = df.groupby('PriceArea')['SpotPriceDKK'].pct_change()
        
        # Volatility features
        df['price_volatility_24h'] = df.groupby('PriceArea')['SpotPriceDKK'].rolling(24, min_periods=1).std().reset_index(0, drop=True)
        df['price_range_24h'] = df.groupby('PriceArea')['SpotPriceDKK'].rolling(24, min_periods=1).max().reset_index(0, drop=True) - \
                               df.groupby('PriceArea')['SpotPriceDKK'].rolling(24, min_periods=1).min().reset_index(0, drop=True)
        
        # Regional features
        df['is_dk1'] = (df['PriceArea'] == 'DK1').astype(int)
        df['is_dk2'] = (df['PriceArea'] == 'DK2').astype(int)
        
        # Weekend and holiday features
        df['is_weekend'] = df['IsWeekend'].astype(int)
        
        return df
    
    def _validate_ml_data(self, df):
        """
        Final validation of ML-ready data.
        """
        # Remove any remaining infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Remove rows with too many missing values
        max_missing = len(df.columns) * 0.3  # Allow 30% missing values
        df = df.dropna(thresh=len(df.columns) - max_missing)
        
        # Fill remaining NaN values with appropriate methods
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                # Use forward fill then backward fill for time series
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                # If still NaN, use median
                df[col] = df[col].fillna(df[col].median())
        
        # Ensure no infinite values remain
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        return df
    
    def _get_demand_category(self, hour):
        """Categorize hours based on typical district heating demand."""
        if 6 <= hour <= 9 or 17 <= hour <= 22:
            return 'High Demand'
        elif 10 <= hour <= 16:
            return 'Medium Demand'
        else:
            return 'Low Demand'
    
    def _is_chp_optimal_hour(self, hour):
        """Identify when CHP production is most profitable."""
        if 7 <= hour <= 11 or 17 <= hour <= 20:
            return 'Peak - CHP Optimal'
        else:
            return 'Off-Peak'
    
    def save_ml_datasets(self, df):
        """
        Save ML-ready datasets with different splits.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save full dataset
        full_path = self.output_dir / f"veks_ml_full_dataset_{timestamp}.parquet"
        df.to_parquet(full_path, index=False)
        logger.info(f"Saved full ML dataset: {full_path}")
        
        # Save regional datasets
        for region in self.veks_regions:
            region_df = df[df['PriceArea'] == region].copy()
            region_path = self.output_dir / f"veks_ml_{region.lower()}_dataset_{timestamp}.parquet"
            region_df.to_parquet(region_path, index=False)
            logger.info(f"Saved {region} ML dataset: {region_path}")
        
        # Save recent data (last 30 days) for quick testing
        recent_cutoff = df['HourDK'].max() - timedelta(days=30)
        recent_df = df[df['HourDK'] >= recent_cutoff].copy()
        recent_path = self.output_dir / f"veks_ml_recent_dataset_{timestamp}.parquet"
        recent_df.to_parquet(recent_path, index=False)
        logger.info(f"Saved recent ML dataset: {recent_path}")
        
        # Create metadata file
        metadata = {
            'timestamp': timestamp,
            'total_records': len(df),
            'date_range': {
                'start': df['HourDK'].min().isoformat(),
                'end': df['HourDK'].max().isoformat()
            },
            'regions': df['PriceArea'].value_counts().to_dict(),
            'features': list(df.columns),
            'file_paths': {
                'full_dataset': str(full_path),
                'recent_dataset': str(recent_path),
                'regional_datasets': {
                    region: str(self.output_dir / f"veks_ml_{region.lower()}_dataset_{timestamp}.parquet")
                    for region in self.veks_regions
                }
            }
        }
        
        metadata_path = self.output_dir / f"veks_ml_metadata_{timestamp}.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved ML metadata: {metadata_path}")
        return metadata
    
    def run_ml_pipeline(self, days=180):
        """
        Run complete ML data pipeline.
        """
        logger.info("Starting VEKS ML data pipeline")
        
        # Fetch data
        logger.info("1. Fetching data from API...")
        raw_data = self.fetch_ml_data(days)
        
        if not raw_data:
            logger.error("No data fetched from API")
            return None
        
        # Clean and prepare data
        logger.info("2. Cleaning and preparing data for ML...")
        df = self.clean_data_for_ml(raw_data)
        
        if df.empty:
            logger.error("No data after cleaning")
            return None
        
        # Save datasets
        logger.info("3. Saving ML-ready datasets...")
        metadata = self.save_ml_datasets(df)
        
        logger.info("ML data pipeline completed successfully!")
        return metadata

def main():
    """
    Main function to run ML data pipeline.
    """
    pipeline = VEKSMLDataPipeline()
    
    try:
        # Run pipeline with 180 days of data
        metadata = pipeline.run_ml_pipeline(days=180)
        
        if metadata:
            print("\n" + "="*50)
            print("VEKS ML DATA PIPELINE SUMMARY")
            print("="*50)
            print(f"Total Records: {metadata['total_records']}")
            print(f"Date Range: {metadata['date_range']['start'][:10]} to {metadata['date_range']['end'][:10]}")
            print(f"Regions: {metadata['regions']}")
            print(f"Features: {len(metadata['features'])}")
            print(f"\nDatasets saved to: {pipeline.output_dir}")
            
    except Exception as e:
        logger.error(f"Error in ML data pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 