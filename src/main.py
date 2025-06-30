"""
VEKS Data Engineer Case Study - Energinet Data Pipeline

This module implements a complete data pipeline that fetches electricity spot prices
from Energinet's API, processes the data, stores it in Parquet format, and creates
interactive visualizations showing daily average prices over time.

Case Requirements Met:
- Relevant dataset (electricity spot prices from Energinet API)
- Data fetching and storage (Parquet format)
- Analysis and visualization (interactive dashboard)
- Daily average prices display

Author: [Your Name]
Date: [Current Date]
"""

import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnerginetDataPipeline:
    """
    Main class for handling Energinet data pipeline operations.
    
    This class provides a complete data pipeline that:
    1. Fetches electricity spot prices from Energinet's API
    2. Processes and validates the data
    3. Stores data in optimized Parquet format
    4. Creates interactive visualizations
    
    Attributes:
        base_url (str): Base URL for Energinet API
        output_dir (Path): Directory for storing output files
    """
    
    def __init__(self):
        """Initialize the pipeline with API configuration and output directory."""
        self.base_url = "https://api.energidataservice.dk/dataset"
        self.output_dir = Path("data")
        self.output_dir.mkdir(exist_ok=True)
        
    def fetch_electricity_prices(self, start_date: Optional[str] = None, 
                                end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch electricity spot prices from Energinet API.
        
        Args:
            start_date (str, optional): Start date in YYYY-MM-DD format. 
                Defaults to 7 days ago.
            end_date (str, optional): End date in YYYY-MM-DD format. 
                Defaults to today.
                
        Returns:
            List[Dict[str, Any]]: List of electricity price records from API
            
        Raises:
            requests.exceptions.RequestException: If API request fails
        """
        # Set default dates if not provided
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
            logger.info(f"Fetching electricity price data from {start_date} to {end_date}")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            records = data.get('records', [])
            logger.info(f"Successfully fetched {len(records)} electricity price records")
            
            return records
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from Energinet API: {e}")
            return []
    
    def process_data(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process raw API data into a clean, validated DataFrame.
        
        Args:
            raw_data (List[Dict[str, Any]]): Raw data from Energinet API
            
        Returns:
            pd.DataFrame: Processed and validated electricity price data
            
        Note:
            This method performs the following data processing steps:
            - Converts timestamps to datetime objects
            - Converts price columns to numeric values
            - Adds date column for daily aggregation
            - Handles missing or invalid data
        """
        if not raw_data:
            logger.warning("No data to process")
            return pd.DataFrame()
        
        df = pd.DataFrame(raw_data)
        
        # Convert timestamp columns to datetime
        df['HourUTC'] = pd.to_datetime(df['HourUTC'], errors='coerce')
        df['HourDK'] = pd.to_datetime(df['HourDK'], errors='coerce')
        
        # Convert price columns to numeric, handling invalid values
        df['SpotPriceDKK'] = pd.to_numeric(df['SpotPriceDKK'], errors='coerce')
        df['SpotPriceEUR'] = pd.to_numeric(df['SpotPriceEUR'], errors='coerce')
        
        # Add date column for daily aggregation
        df['Date'] = df['HourDK'].dt.date
        
        # Remove rows with invalid data
        initial_count = len(df)
        df = df.dropna(subset=['HourDK', 'SpotPriceDKK'])
        final_count = len(df)
        
        if initial_count != final_count:
            logger.warning(f"Removed {initial_count - final_count} rows with invalid data")
        
        logger.info(f"Successfully processed {len(df)} electricity price records")
        return df
    
    def save_to_parquet(self, df: pd.DataFrame, filename: Optional[str] = None) -> None:
        """
        Save DataFrame to optimized Parquet format.
        
        Args:
            df (pd.DataFrame): Data to save
            filename (str, optional): Output filename. If not provided, 
                generates timestamped filename.
                
        Note:
            Parquet format is chosen for its:
            - Columnar storage efficiency
            - Compression benefits
            - Fast query performance
            - Schema preservation
        """
        if df.empty:
            logger.warning("No data to save")
            return
            
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"electricity_prices_{timestamp}.parquet"
        
        filepath = self.output_dir / filename
        df.to_parquet(filepath, index=False, compression='snappy')
        logger.info(f"Electricity price data saved to {filepath}")
    
    def create_dashboard(self, df: pd.DataFrame) -> Optional[go.Figure]:
        """
        Create interactive Plotly dashboard with electricity price visualizations.
        
        Args:
            df (pd.DataFrame): Processed electricity price data
            
        Returns:
            go.Figure: Plotly figure object, or None if no data
            
        Note:
            The dashboard includes:
            - Daily average price trends
            - Hourly price visualization
            - Interactive hover information
            - Professional styling
        """
        if df.empty:
            logger.warning("No data available for dashboard creation")
            return None
        
        # Calculate daily average prices
        daily_avg = df.groupby('Date')['SpotPriceDKK'].mean().reset_index()
        daily_avg['Date'] = pd.to_datetime(daily_avg['Date'])
        
        # Create interactive figure
        fig = go.Figure()
        
        # Add daily average line with enhanced styling
        fig.add_trace(go.Scatter(
            x=daily_avg['Date'],
            y=daily_avg['SpotPriceDKK'],
            mode='lines+markers',
            name='Daily Average Price (DKK)',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6, color='#1f77b4'),
            hovertemplate='<b>Date:</b> %{x}<br>' +
                         '<b>Average Price:</b> %{y:.2f} DKK/MWh<extra></extra>'
        ))
        
        # Add hourly prices as scatter plot
        fig.add_trace(go.Scatter(
            x=df['HourDK'],
            y=df['SpotPriceDKK'],
            mode='markers',
            name='Hourly Price (DKK)',
            marker=dict(
                color='lightblue', 
                size=4, 
                opacity=0.6,
                line=dict(width=0.5, color='white')
            ),
            hovertemplate='<b>Time:</b> %{x}<br>' +
                         '<b>Price:</b> %{y:.2f} DKK/MWh<extra></extra>'
        ))
        
        # Update layout with professional styling
        fig.update_layout(
            title={
                'text': 'Electricity Spot Prices - Daily Averages and Hourly Data',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title='Date',
            yaxis_title='Price (DKK/MWh)',
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        # Save dashboard to HTML file
        dashboard_path = self.output_dir / "electricity_prices_dashboard.html"
        fig.write_html(dashboard_path, include_plotlyjs=True)
        logger.info(f"Interactive dashboard saved to {dashboard_path}")
        
        return fig
    
    def run_pipeline(self, start_date: Optional[str] = None, 
                    end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Run the complete data pipeline from data fetching to visualization.
        
        Args:
            start_date (str, optional): Start date for data fetch
            end_date (str, optional): End date for data fetch
            
        Returns:
            pd.DataFrame: Processed electricity price data, or None if pipeline fails
            
        Note:
            This method orchestrates the complete pipeline:
            1. Fetch data from Energinet API
            2. Process and validate the data
            3. Save to Parquet format
            4. Create interactive dashboard
        """
        logger.info("Starting Energinet Data Pipeline")
        
        try:
            # Step 1: Fetch data from API
            raw_data = self.fetch_electricity_prices(start_date, end_date)
            if not raw_data:
                logger.error("Pipeline failed - no data fetched from API")
                return None
            
            # Step 2: Process and validate data
            df = self.process_data(raw_data)
            if df.empty:
                logger.error("Pipeline failed - no data processed")
                return None
            
            # Step 3: Save to Parquet format
            self.save_to_parquet(df)
            
            # Step 4: Create interactive dashboard
            self.create_dashboard(df)
            
            logger.info("Energinet Data Pipeline completed successfully!")
            return df
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            return None


def main():
    """
    Main function to run the Energinet data pipeline.
    
    This function demonstrates the complete pipeline workflow and provides
    a summary of the processed data.
    """
    pipeline = EnerginetDataPipeline()
    
    # Run pipeline for the last 7 days
    df = pipeline.run_pipeline()
    
    if df is not None and not df.empty:
        print(f"\n{'='*50}")
        print("PIPELINE SUMMARY")
        print(f"{'='*50}")
        print(f"Records processed: {len(df):,}")
        print(f"Date range: {df['HourDK'].min().strftime('%Y-%m-%d')} to {df['HourDK'].max().strftime('%Y-%m-%d')}")
        print(f"Market areas: {df['PriceArea'].nunique()}")
        print(f"Average price: {df['SpotPriceDKK'].mean():.2f} DKK/MWh")
        print(f"Price range: {df['SpotPriceDKK'].min():.2f} - {df['SpotPriceDKK'].max():.2f} DKK/MWh")
        print(f"Output files saved in: {pipeline.output_dir}")
        print(f"{'='*50}")
    else:
        print("Pipeline failed to process data. Check logs for details.")


if __name__ == "__main__":
    main() 