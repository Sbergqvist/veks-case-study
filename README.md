# VEKS Energy Analytics Pipeline - Data Engineer Case Study

**Technical Assessment for Vestegnens Kraftvarmeselskab I/S (VEKS)**

This project implements a comprehensive data pipeline that analyzes electricity market data from Energinet's API to support district heating operations and strategic decision-making.

## Case Study Overview

**Technical Challenge:** Build a data pipeline using Energinet's API (Case 1)
**Solution:** Complete data pipeline with analysis, visualization, and advanced analytics

### Core Requirements Met

1. **Relevant Dataset**: Electricity spot prices from Energinet's API
2. **Data Fetching & Storage**: Automated pipeline with Parquet storage
3. **Analysis & Visualization**: Interactive dashboards and comprehensive analytics
4. **Daily Average Prices**: Dashboard showing price trends over time

### Advanced Features (Bonus Work)

- **Machine Learning Analytics**: Price forecasting and anomaly detection
- **CHP Optimization**: Identifies profitable production hours
- **Backtesting Framework**: Historical performance validation
- **Geographic Intelligence**: Regional market analysis
- **Power BI Integration**: Professional dashboard framework

## Quick Start

### Installation
```bash
git clone <repository-url>
cd VEKS-CASE
pip install -r requirements.txt
```

### Run Core Pipeline
```bash
# Basic pipeline (meets case requirements)
python src/main.py

# Enhanced pipeline with additional features
python src/enhanced_pipeline.py
```

## Core Pipeline Features

### Data Ingestion
- **API Integration**: Robust connection to Energinet's data service
- **Error Handling**: Comprehensive retry logic and validation
- **Data Quality**: Automated validation and cleaning processes

### Data Processing
- **ETL Pipeline**: Efficient data transformation and enrichment
- **Storage**: Optimized Parquet format for analytics
- **Scalability**: Designed for production deployment

### Analytics & Visualization
- **Interactive Dashboards**: Plotly-based visualizations
- **Daily Statistics**: Price trends and market analysis
- **Regional Intelligence**: DK1/DK2 market comparison

## Data Results

### Core Pipeline Outputs
- **4,320 records**: 30 days of hourly electricity prices
- **6 market areas**: Comprehensive Nordic market coverage
- **Interactive dashboards**: Real-time price visualization
- **Daily statistics**: Market trend analysis

### Advanced Analytics Results
- **Price forecasting**: MAE 10-13 DKK/MWh, RMSE 22-39 DKK/MWh
- **Anomaly detection**: 425-427 anomalies identified (10% of data)
- **CHP optimization**: 540 profitable production hours identified

## Technical Implementation

### Architecture Design
```
Data Ingestion → Processing → Analytics → Visualization
     ↓              ↓           ↓           ↓
  API Client → ETL Pipeline → ML Models → Dashboards
```

### Key Technologies
- **Python**: Core development language
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning models
- **Parquet**: Efficient data storage format

## Business Value

### Operational Benefits
- **CHP Optimization**: Revenue maximization through price intelligence
- **Risk Management**: Market volatility assessment and hedging strategies
- **Strategic Planning**: Data-driven decision support for 170,000 households

### Technical Benefits
- **Scalable Architecture**: Production-ready pipeline design
- **Maintainable Code**: Clean, documented, and modular implementation
- **Extensible Framework**: Easy integration of additional data sources

## Usage Examples

### Basic Pipeline
```python
from src.main import EnerginetDataPipeline

pipeline = EnerginetDataPipeline()
df = pipeline.run_pipeline()
print(f"Processed {len(df)} records")
```

## License

This project is created as part of a technical assessment for VEKS' Data Engineer position. 