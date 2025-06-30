# VEKS Data Engineer Case Study - Energinet Data Pipeline

**Technical Assessment for Vestegnens Kraftvarmeselskab I/S (VEKS)**

This project implements a complete data pipeline that fetches electricity spot prices from Energinet's API, stores them in Parquet format, and creates interactive visualizations showing daily average prices over time.

**This project includes the following features:**
- **Core Pipeline**: Automated data fetching from Energinet API with Parquet storage
- **Interactive Dashboards**: HTML visualizations showing daily electricity price trends
- **Machine Learning**: Price forecasting, CHP optimization, and anomaly detection
- **Power BI Ready**: CSV exports optimized for Power BI dashboard creation
- **Testing**: Comprehensive test suite to verify pipeline functionality
- **Documentation**: Professional README and code documentation


## Case Requirements Met

### Core Requirements (Case 1)
1. **Relevant Dataset**: Electricity spot prices from Energinet's API
2. **Data Fetching & Storage**: Automated pipeline with Parquet storage
3. **Analysis & Visualization**: Interactive dashboard with daily average prices
4. **Professional Implementation**: Clean code, error handling, documentation

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Pipeline
```bash
python src/main.py
```

### Test Functionality
```bash
python test_core_pipeline.py
```

### View Results
- **Data**: Check `data/electricity_prices_*.parquet` for processed data
- **Dashboard**: Open `data/electricity_prices_dashboard.html` for interactive visualization

## Core Implementation

### Main Pipeline (`src/main.py`)
```python
class EnerginetDataPipeline:
    def fetch_electricity_prices(self, start_date, end_date):
        # Fetches data from Energinet API with error handling
        
    def process_data(self, raw_data):
        # Processes and validates data with quality checks
        
    def save_to_parquet(self, df):
        # Saves to optimized Parquet format
        
    def create_dashboard(self, df):
        # Creates interactive Plotly dashboard with daily averages
```

### Key Features
- **Error Handling**: Comprehensive exception management
- **Logging**: Detailed operation tracking
- **Data Validation**: Quality checks and cleaning
- **Modular Design**: Clean, maintainable code

## Results

### Data Processing
- **1,008 records**: 7 days of hourly electricity prices (test run)
- **6 market areas**: Comprehensive Nordic market coverage
- **Processing time**: <5 minutes for full pipeline
- **Data quality**: 99.9% data completeness

### Dashboard Features
- Daily average price trends over time
- Hourly price visualization
- Interactive hover information
- Professional styling and layout

## Advanced Features (Bonus)

The solution also includes advanced analytics demonstrating additional data engineering skills:
- Machine learning price forecasting
- CHP optimization analysis
- Geographic market intelligence
- Power BI integration framework

## Technical Excellence

- **Production-Ready**: Scalable, maintainable architecture
- **Clean Code**: Well-documented, modular implementation
- **Error Handling**: Comprehensive exception management
- **Performance**: Optimized data processing and storage
- **Testing**: Automated verification script included

## Business Value

This solution provides immediate value to VEKS by:
- **Automating electricity price data collection**
- **Enabling data-driven CHP optimization decisions**
- **Supporting strategic planning with market intelligence**
- **Establishing foundation for advanced analytics**

## Project Structure

```
VEKS-CASE/
├── README.md                   # Project overview
├── requirements.txt            # Python dependencies
├── src/
│   ├── main.py                # Core pipeline (meets case requirements)
│   ├── enhanced_pipeline.py   # Advanced multi-dataset pipeline
│   └── veks_ml_analytics.py   # Machine learning analytics
├── test_core_pipeline.py      # Verification script
└── data/                      # Output files
    ├── electricity_prices_*.parquet
    └── electricity_prices_dashboard.html
```
