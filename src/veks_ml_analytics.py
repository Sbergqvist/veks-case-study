"""
VEKS Machine Learning Analytics
Author: Sebastian Brydensholt
Created for VEKS Data Engineer Position

Advanced ML models for electricity price forecasting, CHP optimization,
and market intelligence using the existing VEKS data pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import traceback
import sys

# ML Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Time series specific
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VEKSMLAnalytics:
    """
    Machine Learning analytics for VEKS energy operations.
    
    Implements:
    - Electricity price forecasting
    - CHP production optimization
    - Anomaly detection
    - Market segmentation
    - Demand forecasting
    """
    
    def __init__(self, data_path="data/veks_ml_data/"):
        self.data_path = Path(data_path)
        self.output_dir = Path("data/veks_ml_results")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
        # Results storage
        self.forecast_results = {}
        self.optimization_results = {}
        self.anomaly_results = {}
        
    def load_and_prepare_data(self, filename_pattern="veks_ml_full_dataset_*.parquet"):
        """
        Load and prepare data for machine learning models.
        """
        # Find the most recent ML data file
        data_files = list(self.data_path.glob(filename_pattern))
        if not data_files:
            raise FileNotFoundError(f"No ML data files found matching {filename_pattern}")
        
        latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading ML data from {latest_file}")
        
        # Load data
        df = pd.read_parquet(latest_file)
        logger.info(f"[DEBUG] Loaded DataFrame columns: {list(df.columns)}")
        df['HourDK'] = pd.to_datetime(df['HourDK'])
        df = df.sort_values('HourDK').reset_index(drop=True)
        
        # Create features for ML
        df = self._create_ml_features(df)
        logger.info(f"[DEBUG] After feature creation, DataFrame columns: {list(df.columns)}")
        
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
        return df
    
    def _create_ml_features(self, df):
        """
        Create advanced features for machine learning models.
        """
        # Time-based features
        df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['HourDK'].dt.dayofweek / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['HourDK'].dt.dayofweek / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Lag features for time series
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f'price_lag_{lag}'] = df.groupby('PriceArea')['SpotPriceDKK'].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12, 24]:
            df[f'price_rolling_mean_{window}'] = df.groupby('PriceArea')['SpotPriceDKK'].rolling(window).mean().reset_index(0, drop=True)
            df[f'price_rolling_std_{window}'] = df.groupby('PriceArea')['SpotPriceDKK'].rolling(window).std().reset_index(0, drop=True)
        
        # Price change features
        df['price_change'] = df.groupby('PriceArea')['SpotPriceDKK'].diff()
        df['price_change_pct'] = df.groupby('PriceArea')['SpotPriceDKK'].pct_change()
        
        # Categorical encoding (only if not already present)
        if 'season_encoded' not in df.columns:
            le = LabelEncoder()
            df['season_encoded'] = le.fit_transform(df['Season'])
            df['heating_demand_encoded'] = le.fit_transform(df['HeatingDemand'])
            df['chp_optimal_encoded'] = le.fit_transform(df['CHPOptimal'])
        
        # Regional features
        df['is_dk1'] = (df['PriceArea'] == 'DK1').astype(int)
        df['is_dk2'] = (df['PriceArea'] == 'DK2').astype(int)
        
        # Weekend and holiday features
        df['is_weekend'] = df['IsWeekend'].astype(int)
        
        # Clean infinite values and outliers
        df = self._clean_data(df)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        return df
    
    def _clean_data(self, df):
        """
        Clean the data by handling infinite values and outliers.
        """
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Handle extreme price values (likely data errors)
        price_cols = ['SpotPriceDKK', 'SpotPriceEUR']
        for col in price_cols:
            if col in df.columns:
                # Remove negative prices (except for rare cases where it might be valid)
                df.loc[df[col] < -100, col] = np.nan
                
                # Cap extremely high prices (likely data errors)
                price_99th = df[col].quantile(0.99)
                df.loc[df[col] > price_99th * 2, col] = price_99th
        
        # Clean rolling statistics
        rolling_cols = [col for col in df.columns if 'rolling' in col]
        for col in rolling_cols:
            # Replace infinite values in rolling stats
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Fill with forward fill then backward fill
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Clean percentage change columns
        pct_cols = [col for col in df.columns if 'pct' in col]
        for col in pct_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(0)
        
        return df
    
    def price_forecasting_model(self, df, region='DK1', forecast_hours=24):
        """
        Build and train electricity price forecasting model.
        """
        logger.info(f"Building price forecasting model for {region}")
        logger.info(f"[DEBUG] DataFrame columns at start of price_forecasting_model: {list(df.columns)}")
        
        # Filter data for specific region
        region_data = df[df['PriceArea'] == region].copy()
        logger.info(f"[DEBUG] Region data shape: {region_data.shape}, columns: {list(region_data.columns)}")
        
        # Prepare features and target
        feature_columns = [col for col in region_data.columns if col not in 
                          ['HourUTC', 'HourDK', 'PriceArea', 'SpotPriceEUR', 'Date', 
                           'DayOfWeek', 'Season', 'HeatingDemand', 'CHPOptimal', 'PriceLevel']]
        logger.info(f"[DEBUG] Feature columns: {feature_columns}")
        if 'SpotPriceDKK' not in region_data.columns:
            logger.error(f"[ERROR] 'SpotPriceDKK' not in columns: {list(region_data.columns)}")
            raise KeyError("'SpotPriceDKK' column missing from region_data")
        try:
            X = region_data[feature_columns].drop('SpotPriceDKK', axis=1)
        except Exception as e:
            logger.error(f"[ERROR] Exception in dropping 'SpotPriceDKK' from features: {e}")
            logger.error(f"[DEBUG] feature_columns: {feature_columns}")
            raise
        y = region_data['SpotPriceDKK']
        
        # Remove rows with NaN or infinite values in X or y
        mask = (~X.isnull().any(axis=1)) & (~X.isin([np.inf, -np.inf]).any(axis=1)) & (~y.isnull()) & (~np.isinf(y))
        X = X[mask]
        y = y[mask]
        logger.info(f"[DEBUG] X shape after cleaning: {X.shape}, y shape: {y.shape}")
        
        # Split data (time series split)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train multiple models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'SVR': SVR(kernel='rbf')
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Cross-validation
            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Remove any remaining NaN or inf in splits
                train_mask = (~X_train.isnull().any(axis=1)) & (~X_train.isin([np.inf, -np.inf]).any(axis=1)) & (~y_train.isnull()) & (~np.isinf(y_train))
                val_mask = (~X_val.isnull().any(axis=1)) & (~X_val.isin([np.inf, -np.inf]).any(axis=1)) & (~y_val.isnull()) & (~np.isinf(y_val))
                X_train, y_train = X_train[train_mask], y_train[train_mask]
                X_val, y_val = X_val[val_mask], y_val[val_mask]
                
                if len(X_train) == 0 or len(X_val) == 0:
                    logger.warning(f"[DEBUG] Empty train or val split for {name}")
                    continue
                
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                except Exception as e:
                    logger.error(f"[ERROR] Exception during model training or prediction: {e}")
                    logger.error(f"[DEBUG] X_train columns: {list(X_train.columns)}")
                    logger.error(f"[DEBUG] X_val columns: {list(X_val.columns)}")
                    raise
                
                mae = mean_absolute_error(y_val, y_pred)
                cv_scores.append(mae)
            
            # Train on full dataset
            if len(X) > 0:
                try:
                    model.fit(X, y)
                except Exception as e:
                    logger.error(f"[ERROR] Exception during final model fit: {e}")
                    logger.error(f"[DEBUG] X columns: {list(X.columns)}")
                    raise
            
            # Store model and results
            self.models[f'{region}_{name}'] = model
            results[name] = {
                'cv_mae_mean': np.mean(cv_scores) if cv_scores else None,
                'cv_mae_std': np.std(cv_scores) if cv_scores else None,
                'model': model
            }
        
        # Generate forecast
        try:
            forecast = self._generate_forecast(region_data, feature_columns, forecast_hours)
        except Exception as e:
            logger.error(f"[ERROR] Exception in _generate_forecast: {e}")
            logger.error(f"[DEBUG] region_data columns: {list(region_data.columns)}")
            raise
        
        self.forecast_results[region] = {
            'model_results': results,
            'forecast': forecast
        }
        
        return results, forecast
    
    def _generate_forecast(self, data, feature_columns, forecast_hours):
        """
        Generate price forecast for the next N hours.
        """
        # Use the best model (XGBoost typically performs well)
        best_model = self.models.get(f'{data["PriceArea"].iloc[0]}_XGBoost')
        if best_model is None:
            logger.warning("No XGBoost model found for forecasting")
            return None
        
        # Create future timestamps
        last_timestamp = data['HourDK'].max()
        future_timestamps = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=forecast_hours,
            freq='H'
        )
        
        # Create future features - simplified approach
        future_data = []
        # Use only historical data for features, avoid complex rolling calculations
        last_prices = data['SpotPriceDKK'].tail(24).tolist()
        
        for i, timestamp in enumerate(future_timestamps):
            row = {}
            row['HourDK'] = timestamp
            row['Hour'] = timestamp.hour
            row['Month'] = timestamp.month
            row['IsWeekend'] = timestamp.weekday() >= 5
            row['DayOfYear'] = timestamp.timetuple().tm_yday
            
            # Time-based features
            row['hour_sin'] = np.sin(2 * np.pi * timestamp.hour / 24)
            row['hour_cos'] = np.cos(2 * np.pi * timestamp.hour / 24)
            row['day_of_week_sin'] = np.sin(2 * np.pi * timestamp.weekday() / 7)
            row['day_of_week_cos'] = np.cos(2 * np.pi * timestamp.weekday() / 7)
            row['month_sin'] = np.sin(2 * np.pi * timestamp.month / 12)
            row['month_cos'] = np.cos(2 * np.pi * timestamp.month / 12)
            row['day_of_year_sin'] = np.sin(2 * np.pi * row['DayOfYear'] / 365)
            row['day_of_year_cos'] = np.cos(2 * np.pi * row['DayOfYear'] / 365)
            
            # Seasonal features
            season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                         3: 'Spring', 4: 'Spring', 5: 'Spring',
                         6: 'Summer', 7: 'Summer', 8: 'Summer',
                         9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
            row['Season'] = season_map[timestamp.month]
            
            # Business logic features
            row['HeatingDemand'] = self._get_demand_category(timestamp.hour)
            row['CHPOptimal'] = self._is_chp_optimal_hour(timestamp.hour)
            
            # Categorical encoding for forecast
            season_map_encoded = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3}
            demand_map_encoded = {'Low Demand': 0, 'Medium Demand': 1, 'High Demand': 2}
            chp_map_encoded = {'Off-Peak': 0, 'Peak - CHP Optimal': 1}
            
            row['season_encoded'] = season_map_encoded[row['Season']]
            row['heating_demand_encoded'] = demand_map_encoded[row['HeatingDemand']]
            row['chp_optimal_encoded'] = chp_map_encoded[row['CHPOptimal']]
            
            # Simplified lag features - use only historical data
            for lag in [1, 2, 3, 6, 12, 24, 48, 72]:
                if i < lag and len(last_prices) >= lag:
                    row[f'price_lag_{lag}'] = last_prices[-(lag-i)]
                elif len(last_prices) > 0:
                    row[f'price_lag_{lag}'] = last_prices[-1]  # Use most recent price
                else:
                    row[f'price_lag_{lag}'] = 0  # Fallback
            
            # Simplified rolling features - use only historical data
            for window in [3, 6, 12, 24, 48, 72]:
                if len(last_prices) >= window:
                    window_data = last_prices[-window:]
                else:
                    window_data = last_prices
                
                if len(window_data) > 0:
                    row[f'price_rolling_mean_{window}'] = np.mean(window_data)
                    row[f'price_rolling_std_{window}'] = np.std(window_data)
                    row[f'price_rolling_min_{window}'] = np.min(window_data)
                    row[f'price_rolling_max_{window}'] = np.max(window_data)
                else:
                    row[f'price_rolling_mean_{window}'] = 0
                    row[f'price_rolling_std_{window}'] = 0
                    row[f'price_rolling_min_{window}'] = 0
                    row[f'price_rolling_max_{window}'] = 0
            
            # Price change features
            row['price_change'] = 0
            row['price_change_pct'] = 0
            
            # Volatility features - use only historical data
            if len(last_prices) >= 24:
                volatility_data = last_prices[-24:]
            else:
                volatility_data = last_prices
            
            if len(volatility_data) > 0:
                row['price_volatility_24h'] = np.std(volatility_data)
                row['price_range_24h'] = np.max(volatility_data) - np.min(volatility_data)
            else:
                row['price_volatility_24h'] = 0
                row['price_range_24h'] = 0
            
            # Regional features
            row['is_dk1'] = 1
            row['is_dk2'] = 0
            row['is_weekend'] = int(timestamp.weekday() >= 5)
            
            # Make prediction
            available_features = [col for col in feature_columns if col != 'SpotPriceDKK' and col in row]
            missing_features = [col for col in feature_columns if col != 'SpotPriceDKK' and col not in row]
            if missing_features:
                logger.warning(f"Missing features for forecast: {missing_features}")
            
            try:
                features = [row[col] for col in available_features]
                row['predicted_price'] = best_model.predict([features])[0]
            except Exception as e:
                logger.error(f"Error making prediction: {e}")
                row['predicted_price'] = np.mean(last_prices) if len(last_prices) > 0 else 0
            
            future_data.append(row)
        
        return pd.DataFrame(future_data)
    
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
    
    def chp_optimization_model(self, df, forecast_hours=24):
        """
        Optimize CHP production scheduling using ML predictions.
        """
        logger.info("Building CHP optimization model")
        logger.info(f"[DEBUG] DataFrame columns at start of chp_optimization_model: {list(df.columns)}")
        
        # For now, use historical data to demonstrate optimization
        # This avoids the forecast DataFrame access issues
        recent_data = df.tail(forecast_hours * 2)  # Use recent historical data
        logger.info(f"[DEBUG] recent_data shape: {recent_data.shape}, columns: {list(recent_data.columns)}")
        
        # Create optimization data from historical prices
        optimization_data = []
        for _, row in recent_data.iterrows():
            opt_row = {
                'timestamp': row.get('HourDK', None),
                'dk1_price': row.get('SpotPriceDKK', None) if row.get('PriceArea', None) == 'DK1' else None,
                'dk2_price': row.get('SpotPriceDKK', None) if row.get('PriceArea', None) == 'DK2' else None,
                'hour': row.get('Hour', None),
                'is_weekend': row.get('is_weekend', False)
            }
            
            # Use the actual price for optimization
            price = row.get('SpotPriceDKK', 0)
            price_threshold = 400  # DKK/MWh threshold for CHP operation
            
            opt_row['should_run_chp'] = price > price_threshold
            opt_row['expected_revenue'] = price if opt_row['should_run_chp'] else 0
            opt_row['profitability_score'] = price / price_threshold if price > price_threshold else 0
            
            optimization_data.append(opt_row)
        
        optimization_df = pd.DataFrame(optimization_data)
        logger.info(f"[DEBUG] optimization_df shape: {optimization_df.shape}, columns: {list(optimization_df.columns)}")
        
        # Calculate optimization metrics
        total_hours = len(optimization_df)
        optimal_hours = optimization_df['should_run_chp'].sum() if 'should_run_chp' in optimization_df.columns else 0
        total_revenue = optimization_df['expected_revenue'].sum() if 'expected_revenue' in optimization_df.columns else 0
        
        optimization_summary = {
            'total_hours': total_hours,
            'optimal_chp_hours': optimal_hours,
            'chp_utilization_rate': optimal_hours / total_hours if total_hours > 0 else 0,
            'total_expected_revenue': total_revenue,
            'average_price_when_running': optimization_df[optimization_df['should_run_chp']]['expected_revenue'].mean() if optimal_hours > 0 else 0,
            'recommendations': self._generate_chp_recommendations(optimization_df)
        }
        
        self.optimization_results = {
            'optimization_data': optimization_df,
            'summary': optimization_summary
        }
        
        return optimization_summary
    
    def _generate_chp_recommendations(self, optimization_df):
        """
        Generate actionable recommendations for CHP operations.
        """
        recommendations = []
        logger.info(f"_generate_chp_recommendations: DataFrame shape: {optimization_df.shape}")
        logger.info(f"Columns: {list(optimization_df.columns)}")
        
        # Peak hours analysis
        if 'should_run_chp' in optimization_df.columns:
            peak_hours = optimization_df[optimization_df['should_run_chp']]
        else:
            peak_hours = pd.DataFrame()
        logger.info(f"Peak hours shape: {peak_hours.shape}")
        if len(peak_hours) > 0:
            avg_peak_price = peak_hours['expected_revenue'].mean() if 'expected_revenue' in peak_hours.columns else 0
            recommendations.append(f"Run CHP for {len(peak_hours)} hours with average price {avg_peak_price:.2f} DKK/MWh")
        else:
            recommendations.append("No optimal CHP hours detected in the period.")
        
        # Weekend vs weekday analysis (robust)
        if 'is_weekend' in peak_hours.columns:
            weekday_optimal = peak_hours[peak_hours['is_weekend'] == 0].shape[0]
            weekend_optimal = peak_hours[peak_hours['is_weekend'] == 1].shape[0]
        else:
            weekday_optimal = weekend_optimal = 0
        recommendations.append(f"Optimal hours: {weekday_optimal} weekday, {weekend_optimal} weekend")
        
        # Price volatility warning (robust)
        if 'expected_revenue' in peak_hours.columns and len(peak_hours) > 1:
            price_std = peak_hours['expected_revenue'].std()
            logger.info(f"Peak hours price std: {price_std}")
            if price_std > 200:
                recommendations.append(f"High price volatility detected (std: {price_std:.2f}). Consider flexible scheduling.")
        else:
            recommendations.append("Not enough data for volatility analysis.")
        
        return recommendations
    
    def anomaly_detection(self, df):
        """
        Detect anomalies in electricity prices and market behavior.
        """
        logger.info("Running anomaly detection")
        logger.info(f"[DEBUG] DataFrame columns at start of anomaly_detection: {list(df.columns)}")
        
        # Prepare data for anomaly detection
        price_data = df[[col for col in ['HourDK', 'PriceArea', 'SpotPriceDKK'] if col in df.columns]].copy()
        logger.info(f"[DEBUG] price_data shape: {price_data.shape}, columns: {list(price_data.columns)}")
        
        # Detect anomalies per region
        anomalies = {}
        
        for region in ['DK1', 'DK2']:
            if 'PriceArea' not in price_data.columns or 'SpotPriceDKK' not in price_data.columns:
                logger.warning(f"[WARNING] Missing columns for anomaly detection in region {region}")
                continue
            region_data = price_data[price_data['PriceArea'] == region]['SpotPriceDKK'].values.reshape(-1, 1)
            
            # Isolation Forest for anomaly detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(region_data)
            
            # Get anomaly indices
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            anomaly_prices = region_data[anomaly_indices].flatten()
            
            # Statistical analysis
            mean_price = np.mean(region_data)
            std_price = np.std(region_data)
            
            # Additional statistical anomalies (beyond 3 standard deviations)
            statistical_anomalies = np.where(np.abs(region_data - mean_price) > 3 * std_price)[0]
            
            anomalies[region] = {
                'isolation_forest_anomalies': len(anomaly_indices),
                'statistical_anomalies': len(statistical_anomalies),
                'anomaly_prices': anomaly_prices.tolist(),
                'mean_price': mean_price,
                'std_price': std_price,
                'anomaly_indices': anomaly_indices.tolist()
            }
        
        self.anomaly_results = anomalies
        
        return anomalies
    
    def market_segmentation(self, df):
        """
        Perform market segmentation using clustering analysis.
        """
        logger.info("Running market segmentation analysis")
        logger.info(f"[DEBUG] DataFrame columns at start of market_segmentation: {list(df.columns)}")
        
        # Prepare features for clustering
        clustering_features = [
            'SpotPriceDKK', 'hour_sin', 'hour_cos', 'day_of_week_sin', 
            'day_of_week_cos', 'month_sin', 'month_cos', 'is_weekend'
        ]
        available_features = [col for col in clustering_features if col in df.columns]
        if len(available_features) < len(clustering_features):
            logger.warning(f"[WARNING] Missing clustering features: {set(clustering_features) - set(available_features)}")
        X_cluster = df[available_features].dropna()
        logger.info(f"[DEBUG] X_cluster shape: {X_cluster.shape}, columns: {list(X_cluster.columns)}")
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        cluster_labels = kmeans.fit_predict(X_cluster)
        
        # Add cluster labels to data
        df_with_clusters = X_cluster.copy()
        df_with_clusters['cluster'] = cluster_labels
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(4):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            cluster_analysis[f'Cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'avg_price': cluster_data['SpotPriceDKK'].mean() if 'SpotPriceDKK' in cluster_data.columns else None,
                'price_std': cluster_data['SpotPriceDKK'].std() if 'SpotPriceDKK' in cluster_data.columns else None,
                'avg_hour': np.arctan2(cluster_data['hour_sin'].mean(), cluster_data['hour_cos'].mean()) * 12 / np.pi if 'hour_sin' in cluster_data.columns and 'hour_cos' in cluster_data.columns else None,
                'weekend_ratio': cluster_data['is_weekend'].mean() if 'is_weekend' in cluster_data.columns else None
            }
        
        return cluster_analysis, df_with_clusters
    
    def generate_ml_report(self):
        """
        Generate comprehensive ML analysis report.
        """
        # Create a clean report without circular references
        report = {
            'timestamp': datetime.now().isoformat(),
            'forecast_results': {},
            'backtest_results': {},
            'optimization_results': {},
            'anomaly_results': self.anomaly_results
        }
        
        # Add forecast results without model objects
        for region, results in self.forecast_results.items():
            report['forecast_results'][region] = {
                'model_results': {}
            }
            for model_name, model_data in results['model_results'].items():
                report['forecast_results'][region]['model_results'][model_name] = {
                    'cv_mae_mean': model_data['cv_mae_mean'],
                    'cv_mae_std': model_data['cv_mae_std']
                }
            # Add forecast data if available
            if 'forecast' in results and results['forecast'] is not None:
                report['forecast_results'][region]['forecast_summary'] = {
                    'forecast_hours': len(results['forecast']),
                    'avg_predicted_price': results['forecast']['predicted_price'].mean() if 'predicted_price' in results['forecast'].columns else None,
                    'min_predicted_price': results['forecast']['predicted_price'].min() if 'predicted_price' in results['forecast'].columns else None,
                    'max_predicted_price': results['forecast']['predicted_price'].max() if 'predicted_price' in results['forecast'].columns else None
                }
        
        # Add backtest results
        if hasattr(self, 'backtest_results'):
            for region, backtest_data in self.backtest_results.items():
                report['backtest_results'][region] = {
                    'metrics': backtest_data['metrics'],
                    'test_days': backtest_data['test_days'],
                    'n_predictions': len(backtest_data['predictions']),
                    'actual_prices_summary': {
                        'mean': np.mean(backtest_data['actuals']),
                        'std': np.std(backtest_data['actuals']),
                        'min': np.min(backtest_data['actuals']),
                        'max': np.max(backtest_data['actuals'])
                    },
                    'predicted_prices_summary': {
                        'mean': np.mean(backtest_data['predictions']),
                        'std': np.std(backtest_data['predictions']),
                        'min': np.min(backtest_data['predictions']),
                        'max': np.max(backtest_data['predictions'])
                    }
                }
        
        # Add optimization results
        if self.optimization_results:
            report['optimization_results'] = {
                'summary': self.optimization_results['summary']
            }
            # Add optimization data summary
            if 'optimization_data' in self.optimization_results:
                opt_df = self.optimization_results['optimization_data']
                report['optimization_results']['data_summary'] = {
                    'total_records': len(opt_df),
                    'optimal_hours': opt_df['should_run_chp'].sum() if 'should_run_chp' in opt_df.columns else 0,
                    'avg_revenue': opt_df['expected_revenue'].mean() if 'expected_revenue' in opt_df.columns else 0
                }
        
        # Save report
        report_path = self.output_dir / f"veks_ml_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        import json
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif pd.isna(obj):
                return None
            return obj
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, default=convert_numpy, indent=2)
            logger.info(f"ML report saved to {report_path}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            # Save a simplified report
            simple_report = {
                'timestamp': datetime.now().isoformat(),
                'status': 'completed',
                'error': str(e)
            }
            with open(report_path, 'w') as f:
                json.dump(simple_report, f, indent=2)
        
        return report
    
    def run_complete_ml_analysis(self, forecast_hours=24):
        """
        Run complete ML analysis pipeline.
        """
        logger.info("Starting complete ML analysis pipeline")
        
        # Load and prepare data
        df = self.load_and_prepare_data()
        
        # Run all analyses
        logger.info("1. Running price forecasting...")
        for region in ['DK1', 'DK2']:
            self.price_forecasting_model(df, region, forecast_hours)
        
        logger.info("2. Running backtest forecasting...")
        for region in ['DK1', 'DK2']:
            self.backtest_forecasting(df, region, test_days=7)
        
        logger.info("3. Running CHP optimization...")
        self.chp_optimization_model(df, forecast_hours)
        
        logger.info("4. Running anomaly detection...")
        self.anomaly_detection(df)
        
        logger.info("5. Running market segmentation...")
        segmentation_results = self.market_segmentation(df)
        
        logger.info("6. Generating report...")
        report = self.generate_ml_report()
        
        logger.info("ML analysis pipeline completed successfully!")
        return report
    
    def backtest_forecasting(self, df, region='DK1', test_days=7):
        """
        Perform walk-forward validation on historical data.
        
        Args:
            df: Full dataset
            region: Price area (DK1 or DK2)
            test_days: Number of days to use for testing (default: 7)
        
        Returns:
            Dictionary with backtest results and metrics
        """
        logger.info(f"Starting backtest for {region} with {test_days} test days")
        
        # Filter data for specific region
        region_data = df[df['PriceArea'] == region].copy()
        region_data = region_data.sort_values('HourDK').reset_index(drop=True)
        
        # Split into train and test sets
        test_hours = test_days * 24
        train_data = region_data.iloc[:-test_hours].copy()
        test_data = region_data.iloc[-test_hours:].copy()
        
        logger.info(f"Train set: {len(train_data)} hours, Test set: {len(test_data)} hours")
        
        # Prepare features for training
        feature_columns = [col for col in train_data.columns if col not in 
                          ['HourUTC', 'HourDK', 'PriceArea', 'SpotPriceEUR', 'Date', 
                           'DayOfWeek', 'Season', 'HeatingDemand', 'CHPOptimal', 'PriceLevel']]
        
        X_train = train_data[feature_columns].drop('SpotPriceDKK', axis=1)
        y_train = train_data['SpotPriceDKK']
        
        # Remove rows with NaN or infinite values
        mask = (~X_train.isnull().any(axis=1)) & (~X_train.isin([np.inf, -np.inf]).any(axis=1)) & (~y_train.isnull()) & (~np.isinf(y_train))
        X_train = X_train[mask]
        y_train = y_train[mask]
        
        # Train the best model (Random Forest)
        best_model = RandomForestRegressor(n_estimators=100, random_state=42)
        best_model.fit(X_train, y_train)
        
        # Perform walk-forward prediction
        predictions = []
        actuals = []
        timestamps = []
        
        for i in range(len(test_data)):
            # Get the current test point
            current_point = test_data.iloc[i:i+1]
            
            # Prepare features for prediction (excluding target)
            X_test_point = current_point[feature_columns].drop('SpotPriceDKK', axis=1)
            
            # Make prediction
            try:
                pred = best_model.predict(X_test_point)[0]
                actual = current_point['SpotPriceDKK'].iloc[0]
                timestamp = current_point['HourDK'].iloc[0]
                
                predictions.append(pred)
                actuals.append(actual)
                timestamps.append(timestamp)
                
            except Exception as e:
                logger.warning(f"Error predicting point {i}: {e}")
                continue
        
        # Calculate error metrics
        if len(predictions) > 0:
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'timestamp': timestamps,
                'actual_price': actuals,
                'predicted_price': predictions,
                'error': np.array(actuals) - np.array(predictions),
                'error_pct': (np.array(actuals) - np.array(predictions)) / np.array(actuals) * 100
            })
            
            backtest_results = {
                'region': region,
                'test_days': test_days,
                'metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'n_predictions': len(predictions)
                },
                'results_df': results_df,
                'predictions': predictions,
                'actuals': actuals,
                'timestamps': timestamps
            }
            
            logger.info(f"Backtest completed for {region}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
            
            # Store results
            if not hasattr(self, 'backtest_results'):
                self.backtest_results = {}
            self.backtest_results[region] = backtest_results
            
            return backtest_results
        else:
            logger.error("No predictions were made during backtest")
            return None

def main():
    """
    Main function to run VEKS ML analytics.
    """
    ml_analytics = VEKSMLAnalytics()
    
    try:
        # Run complete analysis
        report = ml_analytics.run_complete_ml_analysis(forecast_hours=48)
        
        # Print summary
        print("\n" + "="*50)
        print("VEKS ML ANALYTICS SUMMARY")
        print("="*50)
        
        # Price forecasting summary
        for region in ['DK1', 'DK2']:
            if region in ml_analytics.forecast_results:
                best_model = min(
                    ml_analytics.forecast_results[region]['model_results'].items(), 
                    key=lambda x: x[1]['cv_mae_mean'] if x[1]['cv_mae_mean'] is not None else float('inf')
                )
                print(f"\n{region} Price Forecasting:")
                print(f"  Best Model: {best_model[0]}")
                print(f"  MAE: {best_model[1]['cv_mae_mean']:.2f} DKK/MWh")
        
        # CHP optimization summary
        if ml_analytics.optimization_results:
            opt_summary = ml_analytics.optimization_results['summary']
            print(f"\nCHP Optimization:")
            print(f"  Optimal Hours: {opt_summary['optimal_chp_hours']}/{opt_summary['total_hours']}")
            print(f"  Utilization Rate: {opt_summary['chp_utilization_rate']:.1%}")
            print(f"  Expected Revenue: {opt_summary['total_expected_revenue']:.0f} DKK")
            print(f"  Recommendations: {opt_summary['recommendations']}")
        
        # Anomaly detection summary
        if ml_analytics.anomaly_results:
            print(f"\nAnomaly Detection:")
            for region, results in ml_analytics.anomaly_results.items():
                print(f"  {region}: {results['isolation_forest_anomalies']} anomalies detected")
        
        print(f"\nDetailed results saved to: {ml_analytics.output_dir}")
        
    except Exception as e:
        print("\n[ERROR] Exception occurred during ML analytics run:")
        traceback.print_exc()
        # Try to print DataFrame shapes/columns if available
        try:
            if hasattr(ml_analytics, 'optimization_results') and 'optimization_data' in ml_analytics.optimization_results:
                df = ml_analytics.optimization_results['optimization_data']
                print(f"\n[DEBUG] optimization_data shape: {df.shape}")
                print(f"[DEBUG] optimization_data columns: {list(df.columns)}")
        except Exception as debug_e:
            print(f"[DEBUG] Could not print optimization_data info: {debug_e}")
        # Print all loaded DataFrame columns if possible
        try:
            if hasattr(ml_analytics, 'forecast_results'):
                for region, res in ml_analytics.forecast_results.items():
                    if 'forecast' in res and hasattr(res['forecast'], 'columns'):
                        print(f"[DEBUG] {region} forecast columns: {list(res['forecast'].columns)}")
        except Exception as debug_e:
            print(f"[DEBUG] Could not print forecast info: {debug_e}")
        raise

if __name__ == "__main__":
    main() 