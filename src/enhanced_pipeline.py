"""
Enhanced Energinet Data Pipeline - Interview Ready Version
Optimized for Power BI dashboards and advanced analytics

This enhanced version includes:
- Multiple datasets from Energinet API
- Advanced data transformations
- Power BI optimized outputs
- Comprehensive analytics
"""

import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
from pathlib import Path
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedEnerginetPipeline:
    """Enhanced pipeline for comprehensive energy data analysis"""
    
    def __init__(self):
        self.base_url = "https://api.energidataservice.dk/dataset"
        self.output_dir = Path("data")
        self.output_dir.mkdir(exist_ok=True)
        
        # Available datasets for comprehensive analysis
        self.datasets = {
            'electricity_prices': 'elspotprices',
            'production_prognosis': 'productionprognosis', 
            'consumption_prognosis': 'consumptionprognosis',
            'co2_emissions': 'co2emis',
            'production_consumption': 'productionconsumptionsettlement'
        }
    
    def fetch_multiple_datasets(self, start_date=None, end_date=None, datasets=None):
        """
        Fetch multiple datasets for comprehensive analysis
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            datasets (list): List of dataset names to fetch
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if not datasets:
            datasets = ['electricity_prices', 'production_prognosis', 'consumption_prognosis']
        
        all_data = {}
        
        for dataset_name in datasets:
            if dataset_name in self.datasets:
                logger.info(f"Fetching {dataset_name} data")
                data = self._fetch_single_dataset(
                    self.datasets[dataset_name], 
                    start_date, 
                    end_date
                )
                if data:
                    all_data[dataset_name] = data
                    logger.info(f"Successfully fetched {len(data)} records for {dataset_name}")
                else:
                    logger.warning(f"No data retrieved for {dataset_name}")
        
        return all_data
    
    def _fetch_single_dataset(self, dataset, start_date, end_date):
        """Fetch a single dataset from the API"""
        url = f"{self.base_url}/{dataset}"
        params = {
            "start": start_date,
            "end": end_date,
            "format": "json"
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get('records', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {dataset}: {e}")
            return []
    
    def process_electricity_prices(self, raw_data):
        """Process electricity price data with advanced analytics"""
        if not raw_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(raw_data)
        
        # Data cleaning and transformation
        df['HourUTC'] = pd.to_datetime(df['HourUTC'])
        df['HourDK'] = pd.to_datetime(df['HourDK'])
        df['SpotPriceDKK'] = pd.to_numeric(df['SpotPriceDKK'], errors='coerce')
        df['SpotPriceEUR'] = pd.to_numeric(df['SpotPriceEUR'], errors='coerce')
        
        # Enhanced date/time features for Power BI
        df['Date'] = df['HourDK'].dt.date
        df['Year'] = df['HourDK'].dt.year
        df['Month'] = df['HourDK'].dt.month
        df['Day'] = df['HourDK'].dt.day
        df['Hour'] = df['HourDK'].dt.hour
        df['DayOfWeek'] = df['HourDK'].dt.day_name()
        df['IsWeekend'] = df['HourDK'].dt.weekday >= 5
        df['Quarter'] = df['HourDK'].dt.quarter
        
        # Price analytics
        df['PriceCategory'] = pd.cut(df['SpotPriceDKK'], 
                                   bins=[0, 200, 400, 600, float('inf')],
                                   labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Calculate daily statistics
        daily_stats = df.groupby(['Date', 'PriceArea']).agg({
            'SpotPriceDKK': ['mean', 'min', 'max', 'std'],
            'SpotPriceEUR': ['mean', 'min', 'max']
        }).reset_index()
        
        daily_stats.columns = ['Date', 'PriceArea', 'AvgPriceDKK', 'MinPriceDKK', 'MaxPriceDKK', 
                              'StdPriceDKK', 'AvgPriceEUR', 'MinPriceEUR', 'MaxPriceEUR']
        
        return df, daily_stats
    
    def process_production_data(self, raw_data):
        """Process production prognosis data"""
        if not raw_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(raw_data)
        
        # Clean and transform
        df['HourUTC'] = pd.to_datetime(df['HourUTC'])
        df['HourDK'] = pd.to_datetime(df['HourDK'])
        
        # Convert numeric columns
        numeric_cols = ['TotalProduction', 'OffshoreWindPower', 'OnshoreWindPower', 
                       'SolarPower', 'ThermalPower']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add time features
        df['Date'] = df['HourDK'].dt.date
        df['Hour'] = df['HourDK'].dt.hour
        df['DayOfWeek'] = df['HourDK'].dt.day_name()
        
        # Calculate renewable percentage
        renewable_cols = ['OffshoreWindPower', 'OnshoreWindPower', 'SolarPower']
        renewable_sum = df[renewable_cols].sum(axis=1)
        df['RenewablePercentage'] = (renewable_sum / df['TotalProduction']) * 100
        
        return df
    
    def create_comprehensive_analytics(self, electricity_df, daily_stats, production_df=None):
        """Create comprehensive analytics for Power BI"""
        analytics = {}
        
        # Price analytics
        analytics['price_summary'] = {
            'avg_price_dkk': electricity_df['SpotPriceDKK'].mean(),
            'max_price_dkk': electricity_df['SpotPriceDKK'].max(),
            'min_price_dkk': electricity_df['SpotPriceDKK'].min(),
            'price_volatility': electricity_df['SpotPriceDKK'].std(),
            'total_records': len(electricity_df)
        }
        
        # Market area comparison
        area_stats = electricity_df.groupby('PriceArea').agg({
            'SpotPriceDKK': ['mean', 'std', 'min', 'max'],
            'SpotPriceEUR': 'mean'
        }).round(2)
        
        analytics['area_comparison'] = area_stats
        
        # Time-based patterns
        hourly_patterns = electricity_df.groupby('Hour')['SpotPriceDKK'].mean()
        daily_patterns = electricity_df.groupby('DayOfWeek')['SpotPriceDKK'].mean()
        
        analytics['time_patterns'] = {
            'hourly': hourly_patterns,
            'daily': daily_patterns
        }
        
        # Price correlation matrix
        price_pivot = electricity_df.pivot_table(
            values='SpotPriceDKK',
            index='HourDK',
            columns='PriceArea',
            aggfunc='mean'
        )
        correlation_matrix = price_pivot.corr()
        analytics['price_correlations'] = correlation_matrix
        
        return analytics
    
    def export_for_powerbi(self, electricity_df, daily_stats, analytics, production_df=None):
        """Export data in Power BI optimized format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main electricity data - Power BI ready
        electricity_export = electricity_df.copy()
        electricity_export['Date'] = pd.to_datetime(electricity_export['Date'])
        electricity_file = self.output_dir / f"electricity_data_powerbi_{timestamp}.csv"
        electricity_export.to_csv(electricity_file, index=False)
        logger.info(f"Power BI electricity data exported to {electricity_file}")
        
        # Daily statistics
        daily_export = daily_stats.copy()
        daily_export['Date'] = pd.to_datetime(daily_export['Date'])
        daily_file = self.output_dir / f"daily_statistics_powerbi_{timestamp}.csv"
        daily_export.to_csv(daily_file, index=False)
        logger.info(f"Power BI daily stats exported to {daily_file}")
        
        # Market area summary for Power BI
        area_summary = electricity_df.groupby('PriceArea').agg({
            'SpotPriceDKK': ['mean', 'std', 'min', 'max', 'count'],
            'SpotPriceEUR': 'mean'
        }).round(2)
        area_summary.columns = ['AvgPrice_DKK', 'StdPrice_DKK', 'MinPrice_DKK', 
                               'MaxPrice_DKK', 'RecordCount', 'AvgPrice_EUR']
        area_summary = area_summary.reset_index()
        area_file = self.output_dir / f"market_areas_summary_{timestamp}.csv"
        area_summary.to_csv(area_file, index=False)
        logger.info(f"Power BI market areas summary exported to {area_file}")
        
        # Time patterns for Power BI
        hourly_data = electricity_df.groupby(['Hour', 'PriceArea'])['SpotPriceDKK'].mean().reset_index()
        hourly_data.columns = ['Hour', 'PriceArea', 'AvgPrice_DKK']
        hourly_file = self.output_dir / f"hourly_patterns_powerbi_{timestamp}.csv"
        hourly_data.to_csv(hourly_file, index=False)
        logger.info(f"Power BI hourly patterns exported to {hourly_file}")
        
        # Production data if available
        if production_df is not None and not production_df.empty:
            production_export = production_df.copy()
            production_export['Date'] = pd.to_datetime(production_export['Date'])
            production_file = self.output_dir / f"production_data_powerbi_{timestamp}.csv"
            production_export.to_csv(production_file, index=False)
            logger.info(f"Power BI production data exported to {production_file}")
        
        return {
            'electricity_file': electricity_file,
            'daily_stats_file': daily_file,
            'area_summary_file': area_file,
            'hourly_patterns_file': hourly_file,
            'production_file': production_file if production_df is not None else None
        }
    
    def create_advanced_dashboard(self, electricity_df, daily_stats, analytics):
        """Create an advanced Plotly dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Price Trends by Area', 'Daily Average Comparison', 
                          'Hourly Price Patterns', 'Price Distribution'),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # Price trends by area
        for area in electricity_df['PriceArea'].unique():
            area_data = electricity_df[electricity_df['PriceArea'] == area]
            daily_area = area_data.groupby('Date')['SpotPriceDKK'].mean()
            
            fig.add_trace(
                go.Scatter(x=daily_area.index, y=daily_area.values,
                          name=f'{area} - Daily Avg', mode='lines'),
                row=1, col=1
            )
        
        # Daily comparison bar chart
        area_means = electricity_df.groupby('PriceArea')['SpotPriceDKK'].mean()
        fig.add_trace(
            go.Bar(x=area_means.index, y=area_means.values,
                   name='Average Price by Area'),
            row=1, col=2
        )
        
        # Hourly patterns
        hourly_avg = electricity_df.groupby('Hour')['SpotPriceDKK'].mean()
        fig.add_trace(
            go.Scatter(x=hourly_avg.index, y=hourly_avg.values,
                      mode='lines+markers', name='Hourly Pattern'),
            row=2, col=1
        )
        
        # Price distribution
        fig.add_trace(
            go.Histogram(x=electricity_df['SpotPriceDKK'], name='Price Distribution'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Comprehensive Electricity Market Analysis")
        
        # Save dashboard
        dashboard_path = self.output_dir / "advanced_energy_dashboard.html"
        fig.write_html(dashboard_path)
        logger.info(f"Advanced dashboard saved to {dashboard_path}")
        
        return fig
    
    def run_enhanced_pipeline(self, start_date=None, end_date=None):
        """Run the enhanced pipeline with multiple datasets"""
        logger.info("Starting Enhanced Energinet Data Pipeline")
        
        # Fetch multiple datasets
        all_data = self.fetch_multiple_datasets(start_date, end_date)
        
        if not all_data:
            logger.error("No data fetched - pipeline failed")
            return None
        
        results = {}
        
        # Process electricity prices (main dataset)
        if 'electricity_prices' in all_data:
            electricity_df, daily_stats = self.process_electricity_prices(
                all_data['electricity_prices']
            )
            results['electricity_df'] = electricity_df
            results['daily_stats'] = daily_stats
            
            # Save main data
            self.save_to_parquet(electricity_df, "enhanced_electricity_data.parquet")
            self.save_to_parquet(daily_stats, "daily_statistics.parquet")
        
        # Process production data if available
        production_df = None
        if 'production_prognosis' in all_data:
            production_df = self.process_production_data(all_data['production_prognosis'])
            if not production_df.empty:
                results['production_df'] = production_df
                self.save_to_parquet(production_df, "production_data.parquet")
        
        # Create analytics
        analytics = self.create_comprehensive_analytics(
            electricity_df, daily_stats, production_df
        )
        results['analytics'] = analytics
        
        # Export for Power BI
        powerbi_files = self.export_for_powerbi(
            electricity_df, daily_stats, analytics, production_df
        )
        results['powerbi_files'] = powerbi_files
        
        # Create advanced dashboard
        self.create_advanced_dashboard(electricity_df, daily_stats, analytics)
        
        logger.info("Enhanced pipeline completed successfully!")
        
        return results
    
    def save_to_parquet(self, df, filename):
        """Save DataFrame to Parquet format"""
        if df.empty:
            logger.warning(f"No data to save for {filename}")
            return
        
        filepath = self.output_dir / filename
        df.to_parquet(filepath, index=False)
        logger.info(f"Data saved to {filepath}")

def main():
    """Run the enhanced pipeline"""
    pipeline = EnhancedEnerginetPipeline()
    
    # Run pipeline for the last 30 days for more comprehensive data
    results = pipeline.run_enhanced_pipeline()
    
    if results:
        print(f"\n=== ENHANCED PIPELINE SUMMARY ===")
        print(f"Electricity records: {len(results['electricity_df'])}")
        print(f"Date range: {results['electricity_df']['HourDK'].min()} to {results['electricity_df']['HourDK'].max()}")
        print(f"Market areas: {results['electricity_df']['PriceArea'].unique()}")
        print(f"Average price: {results['analytics']['price_summary']['avg_price_dkk']:.2f} DKK/MWh")
        print(f"Price volatility: {results['analytics']['price_summary']['price_volatility']:.2f}")
        
        if 'production_df' in results:
            print(f"Production records: {len(results['production_df'])}")
        
        print(f"\n=== POWER BI FILES CREATED ===")
        for file_type, filepath in results['powerbi_files'].items():
            if filepath:
                print(f"{file_type}: {filepath}")
        
        print(f"\n✅ Ready for Power BI import and dashboard creation!")

if __name__ == "__main__":
    main() 