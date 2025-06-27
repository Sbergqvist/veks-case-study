"""
Energinet Data Pipeline
Case 1: Build a data pipeline using Energinet's API

This script fetches electricity spot prices from Energinet's API,
stores them in Parquet format, and creates visualizations.
"""

import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnerginetDataPipeline:
    """Main class for handling Energinet data pipeline operations"""
    
    def __init__(self):
        self.base_url = "https://api.energidataservice.dk/dataset"
        self.output_dir = Path("data")
        self.output_dir.mkdir(exist_ok=True)
        
    def fetch_electricity_prices(self, start_date=None, end_date=None):
        """
        Fetch electricity spot prices from Energinet API
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        # Dataset for electricity spot prices
        dataset = "elspotprices"
        
        url = f"{self.base_url}/{dataset}"
        params = {
            "start": start_date,
            "end": end_date,
            "format": "json"
        }
        
        try:
            logger.info(f"Fetching data from {start_date} to {end_date}")
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Successfully fetched {len(data.get('records', []))} records")
            
            return data.get('records', [])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data: {e}")
            return []
    
    def process_data(self, raw_data):
        """
        Process raw API data into a clean DataFrame
        
        Args:
            raw_data (list): Raw data from API
            
        Returns:
            pd.DataFrame: Processed data
        """
        if not raw_data:
            logger.warning("No data to process")
            return pd.DataFrame()
        
        df = pd.DataFrame(raw_data)
        
        # Convert timestamp to datetime
        df['HourUTC'] = pd.to_datetime(df['HourUTC'])
        df['HourDK'] = pd.to_datetime(df['HourDK'])
        
        # Convert price to numeric
        df['SpotPriceDKK'] = pd.to_numeric(df['SpotPriceDKK'], errors='coerce')
        df['SpotPriceEUR'] = pd.to_numeric(df['SpotPriceEUR'], errors='coerce')
        
        # Add date column for daily aggregation
        df['Date'] = df['HourDK'].dt.date
        
        logger.info(f"Processed {len(df)} records")
        return df
    
    def save_to_parquet(self, df, filename=None):
        """
        Save DataFrame to Parquet format
        
        Args:
            df (pd.DataFrame): Data to save
            filename (str): Output filename
        """
        if df.empty:
            logger.warning("No data to save")
            return
            
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"electricity_prices_{timestamp}.parquet"
        
        filepath = self.output_dir / filename
        df.to_parquet(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
    
    def create_dashboard(self, df):
        """
        Create Plotly dashboard with electricity price visualizations
        
        Args:
            df (pd.DataFrame): Processed electricity price data
        """
        if df.empty:
            logger.warning("No data for dashboard")
            return
        
        # Calculate daily average prices
        daily_avg = df.groupby('Date')['SpotPriceDKK'].mean().reset_index()
        daily_avg['Date'] = pd.to_datetime(daily_avg['Date'])
        
        # Create figure with subplots
        fig = go.Figure()
        
        # Add daily average line
        fig.add_trace(go.Scatter(
            x=daily_avg['Date'],
            y=daily_avg['SpotPriceDKK'],
            mode='lines+markers',
            name='Daily Average Price (DKK)',
            line=dict(color='blue', width=2)
        ))
        
        # Add hourly prices as scatter
        fig.add_trace(go.Scatter(
            x=df['HourDK'],
            y=df['SpotPriceDKK'],
            mode='markers',
            name='Hourly Price (DKK)',
            marker=dict(color='lightblue', size=4, opacity=0.6)
        ))
        
        fig.update_layout(
            title='Electricity Spot Prices - Daily Averages and Hourly Data',
            xaxis_title='Date',
            yaxis_title='Price (DKK/MWh)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Save dashboard
        dashboard_path = self.output_dir / "electricity_prices_dashboard.html"
        fig.write_html(dashboard_path)
        logger.info(f"Dashboard saved to {dashboard_path}")
        
        return fig
    
    def run_pipeline(self, start_date=None, end_date=None):
        """
        Run the complete data pipeline
        
        Args:
            start_date (str): Start date for data fetch
            end_date (str): End date for data fetch
        """
        logger.info("Starting Energinet Data Pipeline")
        
        # Step 1: Fetch data
        raw_data = self.fetch_electricity_prices(start_date, end_date)
        
        # Step 2: Process data
        df = self.process_data(raw_data)
        
        if df.empty:
            logger.error("Pipeline failed - no data processed")
            return
        
        # Step 3: Save to Parquet
        self.save_to_parquet(df)
        
        # Step 4: Create dashboard
        self.create_dashboard(df)
        
        logger.info("Pipeline completed successfully!")
        
        return df

def main():
    """Main function to run the pipeline"""
    pipeline = EnerginetDataPipeline()
    
    # Run pipeline for the last 7 days
    df = pipeline.run_pipeline()
    
    if not df.empty:
        print(f"\nPipeline Summary:")
        print(f"Records processed: {len(df)}")
        print(f"Date range: {df['HourDK'].min()} to {df['HourDK'].max()}")
        print(f"Average price: {df['SpotPriceDKK'].mean():.2f} DKK/MWh")
        print(f"Files saved in: {pipeline.output_dir}")

if __name__ == "__main__":
    main() 