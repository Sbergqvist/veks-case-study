"""
VEKS Geographic Analytics
Author: Sebastian Brydensholt
Created for VEKS Data Engineer Position

Integrates geographic municipality data with VEKS energy analytics
for enhanced Power BI mapping and regional analysis.
"""

import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VEKSGeographicAnalytics:
    """
    Geographic analytics for VEKS operations.
    
    Integrates municipality-level data with energy market analytics
    to provide regional insights for district heating operations.
    """
    
    def __init__(self):
        self.base_url = "https://api.energidataservice.dk/dataset"
        self.output_dir = Path("data/veks_analytics")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # VEKS operates in these municipalities (approximate list)
        self.veks_municipalities = [
            'Hvidovre', 'Rødovre', 'Glostrup', 'Brøndby', 'Ishøj', 'Vallensbæk',
            'Albertslund', 'Høje-Taastrup', 'Egedal', 'Furesø', 'Herlev', 'Gladsaxe'
        ]
        
        # Municipality number to name mapping for VEKS operational areas
        self.municipality_mapping = {
            '101': 'København',
            '102': 'Frederiksberg', 
            '103': 'Ballerup',
            '104': 'Brøndby',
            '105': 'Dragør',
            '106': 'Gentofte',
            '107': 'Gladsaxe',
            '108': 'Glostrup',
            '109': 'Herlev',
            '110': 'Albertslund',
            '111': 'Hvidovre',
            '112': 'Høje-Taastrup',
            '113': 'Lyngby-Taarbæk',
            '114': 'Rødovre',
            '115': 'Tårnby',
            '116': 'Vallensbæk',
            '117': 'Furesø',
            '118': 'Allerød',
            '119': 'Egedal',
            '120': 'Fredensborg',
            '121': 'Helsingør',
            '122': 'Hillerød',
            '123': 'Hørsholm',
            '124': 'Rudersdal',
            '125': 'Ishøj',
            '126': 'Greve',
            '127': 'Køge',
            '128': 'Lejre',
            '129': 'Roskilde',
            '130': 'Solrød',
            '131': 'Gribskov',
            '132': 'Odsherred',
            '133': 'Holbæk',
            '134': 'Faxe',
            '135': 'Kalundborg',
            '136': 'Ringsted',
            '137': 'Slagelse',
            '138': 'Stevns',
            '139': 'Sorø',
            '140': 'Haslev',
            '141': 'Fuglebjerg',
            '142': 'Næstved',
            '143': 'Vordingborg',
            '144': 'Guldborgsund',
            '145': 'Lolland',
            '146': 'Møn',
            '147': 'Bornholm'
        }
        
    def fetch_municipality_capacity(self):
        """
        Fetch capacity and production data per municipality.
        This provides geographic structure for VEKS operations.
        """
        url = f"{self.base_url}/CapacityPerMunicipality"
        
        try:
            logger.info("Fetching municipality capacity data...")
            # Add timeout and limit for efficiency
            response = requests.get(url, timeout=15, params={'limit': 100})
            response.raise_for_status()
            data = response.json()
            
            records = data.get('records', [])
            logger.info(f"Retrieved {len(records)} municipality capacity records")
            
            return records
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch municipality data: {e}")
            return []
    
    def fetch_community_production(self):
        """
        Fetch monthly electricity production per municipality.
        Provides production type breakdown for regional analysis.
        """
        url = f"{self.base_url}/CommunityProduction"
        
        try:
            logger.info("Fetching community production data...")
            # Add timeout and limit for efficiency
            response = requests.get(url, timeout=15, params={'limit': 50})
            response.raise_for_status()
            data = response.json()
            
            records = data.get('records', [])
            logger.info(f"Retrieved {len(records)} community production records")
            
            return records
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch community production: {e}")
            return []
    
    def process_municipality_data(self, capacity_data, production_data):
        """
        Process and combine municipality-level data.
        """
        if not capacity_data:
            logger.warning("No capacity data to process")
            return pd.DataFrame()
        
        # Process capacity data
        capacity_df = pd.DataFrame(capacity_data)
        
        # Check what columns are actually available
        logger.info(f"Capacity data columns: {list(capacity_df.columns)}")
        
        # Clean and structure the data - use MunicipalityNo instead of MunicipalityName
        if 'MunicipalityNo' in capacity_df.columns:
            capacity_df['MunicipalityNo'] = capacity_df['MunicipalityNo'].astype(str)
            # Add municipality names for readability
            capacity_df['MunicipalityName'] = capacity_df['MunicipalityNo'].map(self.municipality_mapping)
        
        # Add VEKS operational flag based on municipality numbers
        # VEKS operates in municipalities around Copenhagen area
        veks_municipality_numbers = ['101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112']
        capacity_df['VEKS_Operational'] = capacity_df['MunicipalityNo'].isin(veks_municipality_numbers)
        
        # Process production data if available
        production_df = pd.DataFrame()
        if production_data:
            production_df = pd.DataFrame(production_data)
            logger.info(f"Production data columns: {list(production_df.columns)}")
            
            # Convert date columns
            if 'Month' in production_df.columns:
                production_df['Month'] = pd.to_datetime(production_df['Month'])
            
            # Add municipality names for readability
            if 'MunicipalityNo' in production_df.columns:
                production_df['MunicipalityNo'] = production_df['MunicipalityNo'].astype(str)
                production_df['MunicipalityName'] = production_df['MunicipalityNo'].map(self.municipality_mapping)
            
            # Convert numeric columns
            numeric_cols = ['OnshoreWindPower', 'OffshoreWindPower', 'SolarPower', 'CentralPower', 'DecentralPower']
            for col in numeric_cols:
                if col in production_df.columns:
                    production_df[col] = pd.to_numeric(production_df[col], errors='coerce')
        
        return capacity_df, production_df
    
    def create_geographic_insights(self, capacity_df, production_df, veks_price_data):
        """
        Create geographic insights combining municipality data with VEKS analytics.
        """
        insights = {}
        
        # Municipality capacity analysis
        if not capacity_df.empty:
            insights['municipality_capacity'] = {
                'total_municipalities': len(capacity_df),
                'veks_operational_municipalities': capacity_df['VEKS_Operational'].sum(),
                'total_capacity': capacity_df.get('Capacity', pd.Series([0])).sum(),
                'production_units': capacity_df.get('ProductionUnits', pd.Series([0])).sum()
            }
        
        # Production analysis by municipality
        if not production_df.empty:
            recent_production = production_df[production_df['Month'] >= '2024-01-01']
            
            insights['municipality_production'] = {
                'total_production_mwh': recent_production.get('OnshoreWindPower', pd.Series([0])).sum() + 
                                      recent_production.get('OffshoreWindPower', pd.Series([0])).sum() +
                                      recent_production.get('SolarPower', pd.Series([0])).sum() +
                                      recent_production.get('CentralPower', pd.Series([0])).sum() +
                                      recent_production.get('DecentralPower', pd.Series([0])).sum(),
                'renewable_percentage': self._calculate_renewable_percentage(recent_production),
                'municipalities_with_production': len(recent_production['MunicipalityNo'].unique())
            }
        
        # Regional price correlation (if VEKS price data available)
        if veks_price_data is not None and not veks_price_data.empty:
            insights['regional_price_analysis'] = {
                'dk1_avg_price': veks_price_data[veks_price_data['PriceArea'] == 'DK1']['SpotPriceDKK'].mean(),
                'dk2_avg_price': veks_price_data[veks_price_data['PriceArea'] == 'DK2']['SpotPriceDKK'].mean(),
                'price_spread': veks_price_data[veks_price_data['PriceArea'] == 'DK1']['SpotPriceDKK'].mean() - 
                               veks_price_data[veks_price_data['PriceArea'] == 'DK2']['SpotPriceDKK'].mean()
            }
        
        return insights
    
    def _calculate_renewable_percentage(self, production_df):
        """Calculate renewable energy percentage from production data."""
        if production_df.empty:
            return 0
        
        renewable_cols = ['OnshoreWindPower', 'OffshoreWindPower', 'SolarPower']
        total_renewable = sum(production_df.get(col, pd.Series([0])).sum() for col in renewable_cols)
        total_production = sum(production_df.get(col, pd.Series([0])).sum() for col in 
                             ['OnshoreWindPower', 'OffshoreWindPower', 'SolarPower', 'CentralPower', 'DecentralPower'])
        
        return (total_renewable / total_production * 100) if total_production > 0 else 0
    
    def export_geographic_data(self, capacity_df, production_df, insights):
        """
        Export geographic data in Power BI optimized format.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Municipality capacity data
        if not capacity_df.empty:
            capacity_file = self.output_dir / f"veks_municipality_capacity_{timestamp}.csv"
            capacity_df.to_csv(capacity_file, index=False)
            logger.info(f"Municipality capacity data exported to {capacity_file}")
        
        # Community production data
        if not production_df.empty:
            production_file = self.output_dir / f"veks_community_production_{timestamp}.csv"
            production_df.to_csv(production_file, index=False)
            logger.info(f"Community production data exported to {production_file}")
        
        # Geographic insights summary
        insights_file = self.output_dir / f"veks_geographic_insights_{timestamp}.json"
        with open(insights_file, 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        logger.info(f"Geographic insights exported to {insights_file}")
        
        return {
            'capacity_file': capacity_file if not capacity_df.empty else None,
            'production_file': production_file if not production_df.empty else None,
            'insights_file': insights_file
        }
    
    def create_geographic_dashboard(self, capacity_df, production_df):
        """
        Create geographic visualization dashboard.
        """
        if capacity_df.empty and production_df.empty:
            logger.warning("No geographic data for dashboard")
            return None
        
        fig = go.Figure()
        
        # Municipality capacity visualization
        if not capacity_df.empty:
            # Filter for VEKS operational municipalities
            veks_capacity = capacity_df[capacity_df['VEKS_Operational'] == True]
            
            if not veks_capacity.empty:
                fig.add_trace(go.Bar(
                    x=veks_capacity['MunicipalityName'],
                    y=veks_capacity.get('CapacityLt100MW', [0] * len(veks_capacity)),
                    name='VEKS Municipalities Capacity',
                    marker_color='green'
                ))
        
        # Production visualization
        if not production_df.empty:
            recent_production = production_df[production_df['Month'] >= '2024-01-01']
            
            if not recent_production.empty:
                # Aggregate by municipality
                municipality_production = recent_production.groupby('MunicipalityName').agg({
                    'OnshoreWindPower': 'sum',
                    'OffshoreWindPower': 'sum',
                    'SolarPower': 'sum',
                    'CentralPower': 'sum',
                    'DecentralPower': 'sum'
                }).reset_index()
                
                # Add renewable vs thermal comparison
                municipality_production['Renewable'] = municipality_production['OnshoreWindPower'] + municipality_production['OffshoreWindPower'] + municipality_production['SolarPower']
                municipality_production['Thermal'] = municipality_production['CentralPower'] + municipality_production['DecentralPower']
                
                fig.add_trace(go.Scatter(
                    x=municipality_production['MunicipalityName'],
                    y=municipality_production['Renewable'],
                    mode='markers',
                    name='Renewable Production',
                    marker=dict(size=10, color='blue')
                ))
        
        fig.update_layout(
            title='VEKS Geographic Analysis - Municipality Capacity and Production',
            xaxis_title='Municipality',
            yaxis_title='Capacity/Production (MW/MWh)',
            template='plotly_white'
        )
        
        # Save dashboard
        dashboard_path = self.output_dir / "veks_geographic_dashboard.html"
        fig.write_html(dashboard_path)
        logger.info(f"Geographic dashboard saved to {dashboard_path}")
        
        return fig
    
    def run_geographic_pipeline(self, veks_price_data=None):
        """
        Run the complete geographic analytics pipeline.
        """
        logger.info("Starting VEKS Geographic Analytics Pipeline")
        
        # Fetch geographic data
        capacity_data = self.fetch_municipality_capacity()
        production_data = self.fetch_community_production()
        
        # Process data
        capacity_df, production_df = self.process_municipality_data(capacity_data, production_data)
        
        # Create insights
        insights = self.create_geographic_insights(capacity_df, production_df, veks_price_data)
        
        # Export data
        export_files = self.export_geographic_data(capacity_df, production_df, insights)
        
        # Create dashboard
        self.create_geographic_dashboard(capacity_df, production_df)
        
        logger.info("Geographic analytics pipeline completed successfully!")
        
        return {
            'capacity_df': capacity_df,
            'production_df': production_df,
            'insights': insights,
            'export_files': export_files
        }

def main():
    """Run the geographic analytics pipeline."""
    geographic_analytics = VEKSGeographicAnalytics()
    
    # Run pipeline
    results = geographic_analytics.run_geographic_pipeline()
    
    if results:
        print(f"\n=== VEKS GEOGRAPHIC ANALYTICS SUMMARY ===")
        print(f"Municipalities analyzed: {results['insights']['municipality_capacity']['total_municipalities']}")
        print(f"VEKS operational municipalities: {results['insights']['municipality_capacity']['veks_operational_municipalities']}")
        print(f"Total capacity: {results['insights']['municipality_capacity']['total_capacity']:.2f} MW")
        print(f"Production units: {results['insights']['municipality_capacity']['production_units']}")
        
        if 'municipality_production' in results['insights']:
            print(f"Renewable percentage: {results['insights']['municipality_production']['renewable_percentage']:.1f}%")
        
        print(f"\n=== POWER BI FILES CREATED ===")
        for file_type, filepath in results['export_files'].items():
            if filepath:
                print(f"{file_type}: {filepath}")
        
        print(f"\n✅ Geographic data ready for Power BI mapping!")

if __name__ == "__main__":
    main() 