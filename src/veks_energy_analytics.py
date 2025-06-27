"""
VEKS Energy Market Analytics
Author: Sebastian Brydensholt
Created for VEKS Data Engineer Position

Analyzes electricity market data to support district heating operations.
Focuses on CHP optimization and market intelligence for DK1/DK2 regions.
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VEKSEnergyAnalytics:
    """
    Energy market analytics tailored for VEKS operations.
    
    VEKS operates in DK1 and DK2 price areas, so we focus our analysis there.
    Main use cases:
    - CHP production timing optimization
    - Market price intelligence for planning
    - Regional price spread analysis
    """
    
    def __init__(self):
        self.base_url = "https://api.energidataservice.dk/dataset"
        self.output_dir = Path("data/veks_analytics")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # VEKS primarily operates in these Danish price areas
        self.veks_regions = ['DK1', 'DK2']
        
    def fetch_market_data(self, start_date=None, end_date=None):
        """
        Get electricity spot prices from Energinet API.
        Default to last 30 days if no dates specified.
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Fetching data from {start_date} to {end_date}")
        
        url = f"{self.base_url}/elspotprices"
        params = {
            "start": start_date,
            "end": end_date,
            "format": "json"
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Filter for our regions of interest
            all_records = data.get('records', [])
            veks_records = [r for r in all_records if r.get('PriceArea') in self.veks_regions]
            
            logger.info(f"Retrieved {len(veks_records)} records for DK1/DK2")
            return veks_records
            
        except requests.RequestException as e:
            logger.error(f"API call failed: {e}")
            return []
    
    def process_market_data(self, raw_data):
        """
        Clean and enrich the electricity market data.
        Add features relevant for district heating operations.
        """
        if not raw_data:
            logger.warning("No data to process")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        df = pd.DataFrame(raw_data)
        
        # Basic data cleaning
        df['HourDK'] = pd.to_datetime(df['HourDK'])
        df['SpotPriceDKK'] = pd.to_numeric(df['SpotPriceDKK'], errors='coerce')
        df['SpotPriceEUR'] = pd.to_numeric(df['SpotPriceEUR'], errors='coerce')
        
        # Add time-based features for analysis
        df['Date'] = df['HourDK'].dt.date
        df['Hour'] = df['HourDK'].dt.hour
        df['DayOfWeek'] = df['HourDK'].dt.day_name()
        df['IsWeekend'] = df['HourDK'].dt.weekday >= 5
        df['Month'] = df['HourDK'].dt.month
        
        # Seasonal mapping - important for heating demand
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                     9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
        df['Season'] = df['Month'].map(season_map)
        
        # District heating specific features
        df['HeatingDemand'] = df['Hour'].apply(self._get_demand_category)
        df['CHPOptimal'] = df['Hour'].apply(self._is_chp_optimal_hour)
        
        # Price level categorization for decision making
        df['PriceLevel'] = pd.cut(df['SpotPriceDKK'], 
                                bins=[0, 200, 400, 600, 1000, float('inf')],
                                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # Calculate regional price differences
        price_spread = self._calculate_regional_spread(df)
        
        # Daily summary statistics
        daily_summary = self._create_daily_summary(df)
        
        return df, daily_summary, price_spread
    
    def _get_demand_category(self, hour):
        """Categorize hours based on typical district heating demand."""
        # Morning and evening peaks when people are home
        if 6 <= hour <= 9 or 17 <= hour <= 22:
            return 'High Demand'
        elif 10 <= hour <= 16:
            return 'Medium Demand'
        else:
            return 'Low Demand'
    
    def _is_chp_optimal_hour(self, hour):
        """
        Identify when CHP production is most profitable.
        Based on typical electricity price patterns.
        """
        # Peak price hours when CHP gives best economics
        if 7 <= hour <= 11 or 17 <= hour <= 20:
            return 'Peak - CHP Optimal'
        else:
            return 'Off-Peak'
    
    def _calculate_regional_spread(self, df):
        """Calculate price differences between DK1 and DK2."""
        if not all(area in df['PriceArea'].values for area in ['DK1', 'DK2']):
            return pd.DataFrame()
        
        # Pivot to get DK1 and DK2 prices side by side
        dk1_data = df[df['PriceArea'] == 'DK1'][['HourDK', 'SpotPriceDKK']].set_index('HourDK')
        dk2_data = df[df['PriceArea'] == 'DK2'][['HourDK', 'SpotPriceDKK']].set_index('HourDK')
        
        spread_df = pd.merge(dk1_data, dk2_data, left_index=True, right_index=True, 
                           suffixes=('_DK1', '_DK2'))
        
        # Calculate spread metrics
        spread_df['PriceSpread'] = spread_df['SpotPriceDKK_DK1'] - spread_df['SpotPriceDKK_DK2']
        avg_price = spread_df[['SpotPriceDKK_DK1', 'SpotPriceDKK_DK2']].mean(axis=1)
        spread_df['SpreadPercent'] = (spread_df['PriceSpread'] / avg_price) * 100
        
        return spread_df.reset_index()
    
    def _create_daily_summary(self, df):
        """Create daily aggregated statistics for reporting."""
        daily_stats = df.groupby(['Date', 'PriceArea']).agg({
            'SpotPriceDKK': ['mean', 'min', 'max', 'std'],
            'Hour': 'count'
        }).reset_index()
        
        # Flatten column names
        daily_stats.columns = ['Date', 'PriceArea', 'AvgPrice', 'MinPrice', 
                              'MaxPrice', 'Volatility', 'Records']
        
        # Add derived metrics
        daily_stats['PriceRange'] = daily_stats['MaxPrice'] - daily_stats['MinPrice']
        
        # Volatility categories for risk assessment
        daily_stats['VolatilityLevel'] = pd.cut(daily_stats['Volatility'],
                                               bins=[0, 50, 100, 200, float('inf')],
                                               labels=['Low', 'Medium', 'High', 'Very High'])
        
        return daily_stats
    
    def create_dashboard(self, df, daily_summary, price_spread):
        """Create visualization dashboard for VEKS operations."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'DK1 vs DK2 Daily Prices',
                'CHP Optimal Hours Analysis', 
                'Heating Demand vs Price Correlation',
                'Weekly Price Patterns'
            )
        )
        
        # 1. Regional price comparison
        for region in self.veks_regions:
            if region in df['PriceArea'].values:
                region_data = df[df['PriceArea'] == region]
                daily_prices = region_data.groupby('Date')['SpotPriceDKK'].mean()
                
                fig.add_trace(
                    go.Scatter(x=daily_prices.index, y=daily_prices.values,
                             name=f'{region} Average', mode='lines+markers'),
                    row=1, col=1
                )
        
        # 2. CHP optimization analysis
        chp_data = df.groupby(['Hour', 'CHPOptimal'])['SpotPriceDKK'].mean().reset_index()
        peak_hours = chp_data[chp_data['CHPOptimal'] == 'Peak - CHP Optimal']
        
        if not peak_hours.empty:
            fig.add_trace(
                go.Bar(x=peak_hours['Hour'], y=peak_hours['SpotPriceDKK'],
                      name='CHP Optimal Hours', marker_color='green'),
                row=1, col=2
            )
        
        # 3. Demand vs price correlation
        demand_prices = df.groupby('HeatingDemand')['SpotPriceDKK'].mean()
        
        fig.add_trace(
            go.Scatter(x=demand_prices.index, y=demand_prices.values,
                      mode='markers', marker_size=20,
                      name='Demand Period Prices'),
            row=2, col=1
        )
        
        # 4. Weekly patterns
        weekly_prices = df.groupby('DayOfWeek')['SpotPriceDKK'].mean()
        # Reorder days properly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                    'Friday', 'Saturday', 'Sunday']
        weekly_prices = weekly_prices.reindex(day_order)
        
        fig.add_trace(
            go.Bar(x=weekly_prices.index, y=weekly_prices.values,
                  name='Weekly Pattern', marker_color='lightblue'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="VEKS Energy Market Dashboard",
            showlegend=True
        )
        
        # Save the dashboard
        output_file = self.output_dir / "veks_energy_analytics_dashboard.html"
        fig.write_html(output_file)
        logger.info(f"Dashboard saved: {output_file}")
        
        return fig
    
    def export_for_powerbi(self, df, daily_summary, price_spread):
        """Export data in formats optimized for Power BI import."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main dataset
        main_df = df.copy()
        main_df['Date'] = pd.to_datetime(main_df['Date'])
        
        main_file = self.output_dir / f"veks_electricity_market_data_{timestamp}.csv"
        main_df.to_csv(main_file, index=False)
        
        # Daily summary
        daily_file = self.output_dir / f"veks_daily_analytics_{timestamp}.csv"
        daily_summary['Date'] = pd.to_datetime(daily_summary['Date'])
        daily_summary.to_csv(daily_file, index=False)
        
        # CHP recommendations
        chp_data = df.groupby(['Hour', 'CHPOptimal', 'HeatingDemand']).agg({
            'SpotPriceDKK': 'mean'
        }).reset_index()
        
        chp_file = self.output_dir / f"veks_chp_optimization_{timestamp}.csv"
        chp_data.to_csv(chp_file, index=False)
        
        files_created = {
            'main_data': main_file,
            'daily_summary': daily_file,
            'chp_optimization': chp_file
        }
        
        # Regional spread analysis (if available)
        if not price_spread.empty:
            spread_file = self.output_dir / f"veks_dk1_dk2_spread_{timestamp}.csv"
            price_spread.to_csv(spread_file, index=False)
            files_created['price_spread'] = spread_file
        
        for file_type, filepath in files_created.items():
            logger.info(f"Exported {file_type}: {filepath}")
            
        return files_created
    
    def analyze_market_insights(self, df, daily_summary):
        """Generate key insights for VEKS operations."""
        insights = {}
        
        # Market overview
        insights['market_summary'] = {
            'total_records': len(df),
            'date_range': f"{df['HourDK'].min()} to {df['HourDK'].max()}",
            'regions_covered': df['PriceArea'].unique().tolist(),
            'average_price': df['SpotPriceDKK'].mean(),
            'price_volatility': df['SpotPriceDKK'].std()
        }
        
        # Regional price analysis
        for region in self.veks_regions:
            if region in df['PriceArea'].values:
                region_avg = df[df['PriceArea'] == region]['SpotPriceDKK'].mean()
                insights['market_summary'][f'{region.lower()}_avg_price'] = region_avg
        
        # CHP optimization opportunities
        chp_optimal = df[df['CHPOptimal'] == 'Peak - CHP Optimal']
        insights['chp_analysis'] = {
            'optimal_hours_count': len(chp_optimal),
            'avg_peak_price': chp_optimal['SpotPriceDKK'].mean() if not chp_optimal.empty else 0,
            'peak_hours_by_time': chp_optimal.groupby('Hour')['SpotPriceDKK'].mean().to_dict() if not chp_optimal.empty else {}
        }
        
        # Heating demand insights
        demand_analysis = df.groupby('HeatingDemand')['SpotPriceDKK'].agg(['mean', 'count'])
        insights['demand_correlation'] = demand_analysis.to_dict()
        
        return insights
    
    def run_analysis(self, start_date=None, end_date=None):
        """
        Main function to run the complete analysis pipeline.
        Returns processed data and insights for VEKS operations.
        """
        logger.info("Starting VEKS energy market analysis")
        
        # Fetch data from API
        raw_data = self.fetch_market_data(start_date, end_date)
        if not raw_data:
            logger.error("No data retrieved - analysis failed")
            return None
        
        # Process and enrich the data
        df, daily_summary, price_spread = self.process_market_data(raw_data)
        if df.empty:
            logger.error("Data processing failed")
            return None
        
        # Generate insights
        insights = self.analyze_market_insights(df, daily_summary)
        
        # Create visualizations
        self.create_dashboard(df, daily_summary, price_spread)
        
        # Export for Power BI
        exported_files = self.export_for_powerbi(df, daily_summary, price_spread)
        
        logger.info("Analysis completed successfully!")
        
        return {
            'data': df,
            'daily_summary': daily_summary,
            'price_spread': price_spread,
            'insights': insights,
            'exported_files': exported_files
        }

def main():
    """Run the VEKS energy market analysis."""
    analyzer = VEKSEnergyAnalytics()
    
    # Run analysis for last 30 days
    results = analyzer.run_analysis()
    
    if results:
        insights = results['insights']
        
        print(f"\n{'='*50}")
        print(f"VEKS ENERGY MARKET ANALYSIS RESULTS")
        print(f"{'='*50}")
        
        print(f"\nMarket Data Overview:")
        print(f"  Records processed: {insights['market_summary']['total_records']}")
        print(f"  Date range: {insights['market_summary']['date_range']}")
        print(f"  Regions: {', '.join(insights['market_summary']['regions_covered'])}")
        print(f"  Average price: {insights['market_summary']['average_price']:.2f} DKK/MWh")
        print(f"  Volatility: {insights['market_summary']['price_volatility']:.2f}")
        
        print(f"\nCHP Optimization Opportunities:")
        print(f"  Optimal production hours: {insights['chp_analysis']['optimal_hours_count']}")
        print(f"  Average peak price: {insights['chp_analysis']['avg_peak_price']:.2f} DKK/MWh")
        
        print(f"\nFiles Created:")
        for file_type, filepath in results['exported_files'].items():
            print(f"  {file_type}: {filepath}")
        
        print(f"\n{'='*50}")
        print(f"Analysis ready for VEKS presentation!")
        print(f"{'='*50}")

if __name__ == "__main__":
    main() 