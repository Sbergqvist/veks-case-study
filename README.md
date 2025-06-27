# Energinet Data Pipeline - Case 1 Solution

This project implements a data pipeline that fetches electricity spot prices from Energinet's API, processes the data, stores it in Parquet format, and creates interactive visualizations.

## Project Overview

**Case 1: Energinet Data Pipeline**
- **API Source:** https://api.energidataservice.dk/index.html
- **Dataset:** Electricity spot prices (elspotprices)
- **Output:** Parquet files + Interactive Plotly dashboard

## Features

- **Data Fetching:** Automated API calls to Energinet's data service
- **Data Processing:** Clean and transform raw API data
- **Storage:** Save data in efficient Parquet format
- **Visualization:** Interactive Plotly dashboard with daily averages
- **Logging:** Comprehensive logging for monitoring
- **Error Handling:** Robust error handling and validation

## Sample Results

The pipeline successfully processed:
- **1,008 records** of electricity price data
- **Date range:** 7 days of historical data
- **Average price:** 348.33 DKK/MWh
- **Output files:** Parquet data + HTML dashboard

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd VEKS-CASE
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the pipeline:**
   ```bash
   python src/main.py
   ```

## Project Structure

```
VEKS CASE/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── extract_pdf.py           # PDF text extraction utility
├── src/
│   └── main.py              # Main pipeline implementation
└── data/                    # Output directory (created automatically)
    ├── electricity_prices_*.parquet    # Processed data files
    └── electricity_prices_dashboard.html # Interactive dashboard
```

## Usage

### Basic Usage
```python
from src.main import EnerginetDataPipeline

# Create pipeline instance
pipeline = EnerginetDataPipeline()

# Run complete pipeline (last 7 days)
df = pipeline.run_pipeline()
```

### Custom Date Range
```python
# Fetch specific date range
df = pipeline.run_pipeline(
    start_date="2025-06-01",
    end_date="2025-06-15"
)
```

### Individual Components
```python
# Fetch raw data only
raw_data = pipeline.fetch_electricity_prices()

# Process data
df = pipeline.process_data(raw_data)

# Save to Parquet
pipeline.save_to_parquet(df)

# Create dashboard
pipeline.create_dashboard(df)
```

## Data Schema

The processed data includes:
- **HourUTC:** UTC timestamp
- **HourDK:** Danish timestamp
- **SpotPriceDKK:** Price in Danish Krone
- **SpotPriceEUR:** Price in Euro
- **Date:** Date for daily aggregation

## Dashboard Features

The interactive dashboard includes:
- **Daily average prices** (line chart)
- **Hourly price points** (scatter plot)
- **Interactive hover information**
- **Responsive design**
- **Export capabilities**

## API Details

- **Base URL:** https://api.energidataservice.dk/dataset
- **Dataset:** elspotprices
- **Format:** JSON
- **Rate Limits:** Standard API limits apply

## Testing

To test the pipeline:
```bash
# Run the main pipeline
python src/main.py

# Check output files
ls data/
```

## Logging

The pipeline includes comprehensive logging:
- INFO level for successful operations
- WARNING for non-critical issues
- ERROR for failures
- Timestamps for all operations

## Future Enhancements

Potential improvements:
- Add data validation and quality checks
- Implement incremental data loading
- Add more visualization types
- Create automated scheduling
- Add unit tests
- Implement data versioning

## License

This project is created as part of a technical assessment.

## Author

Created for the VEKS data engineer role application.

---

**Note:** This solution demonstrates data engineering best practices including data fetching, processing, storage, and visualization using modern Python tools and libraries. 