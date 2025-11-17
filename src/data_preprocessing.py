"""
Data Preprocessing Module for AQI Anomaly Detection System

This module handles:
- Data loading and exploration
- Missing value imputation
- Feature engineering (temporal, lag features)
- Data normalization and scaling
- Train-test split preparation

Author: TY Sem 5 AIML Student
Date: November 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import warnings
from pathlib import Path
import sys

warnings.filterwarnings('ignore')

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


class AQIDataPreprocessor:
    """Comprehensive data preprocessing pipeline for AQI anomaly detection"""
    
    def __init__(self, data_path=None):
        """
        Initialize the preprocessor
        
        Args:
            data_path (str): Path to the raw dataset CSV file
        """
        if data_path is None:
            data_path = DATA_DIR / 'dataset.csv'
        
        self.data_path = data_path
        self.df = None
        self.df_clean = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load and perform initial exploration of the dataset"""
        print("=" * 80)
        print("📊 LOADING AQI DATASET")
        print("=" * 80)
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded successfully!")
            print(f"   Shape: {self.df.shape}")
            print(f"   Columns: {list(self.df.columns)}")
            print(f"   Date Range: {self.df['Date'].min()} to {self.df['Date'].max()}")
            print(f"   Cities: {self.df['City'].nunique()} unique cities")
            return True
        except FileNotFoundError:
            print(f"❌ Error: Dataset not found at {self.data_path}")
            print(f"   Please ensure dataset.csv is in the {DATA_DIR} directory")
            return False
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            return False
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "=" * 80)
        print("🔍 EXPLORATORY DATA ANALYSIS")
        print("=" * 80)
        
        # Basic statistics
        print("\n📈 Dataset Overview:")
        print(f"   Total Records: {len(self.df):,}")
        print(f"   Features: {len(self.df.columns)}")
        print(f"   Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values analysis
        print("\n🔍 Missing Values Analysis:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing': missing.values,
            'Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
        
        if len(missing_df) > 0:
            print(missing_df.to_string(index=False))
            print(f"\n   Total Missing Values: {missing.sum():,}")
            print(f"   Percentage: {(missing.sum() / (len(self.df) * len(self.df.columns)) * 100):.2f}%")
        else:
            print("   ✅ No missing values found!")
        
        # AQI statistics
        if 'AQI' in self.df.columns:
            print("\n📊 AQI Statistics:")
            print(f"   Mean: {self.df['AQI'].mean():.2f}")
            print(f"   Median: {self.df['AQI'].median():.2f}")
            print(f"   Std Dev: {self.df['AQI'].std():.2f}")
            print(f"   Min: {self.df['AQI'].min():.2f}")
            print(f"   Max: {self.df['AQI'].max():.2f}")
        
        # AQI Bucket distribution
        if 'AQI_Bucket' in self.df.columns:
            print("\n🏷️  AQI Bucket Distribution:")
            bucket_counts = self.df['AQI_Bucket'].value_counts()
            for bucket, count in bucket_counts.items():
                pct = (count / len(self.df)) * 100
                print(f"   {bucket:15s}: {count:6,} ({pct:5.2f}%)")
        
        # City distribution
        print("\n🌆 Top 10 Cities by Record Count:")
        city_counts = self.df['City'].value_counts().head(10)
        for city, count in city_counts.items():
            print(f"   {city:20s}: {count:6,}")
    
    def handle_missing_values(self):
        """Handle missing values using intelligent imputation strategies"""
        print("\n" + "=" * 80)
        print("🧹 HANDLING MISSING VALUES")
        print("=" * 80)
        
        # Create a copy
        self.df_clean = self.df.copy()
        
        # Define pollutant columns
        pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 
                         'O3', 'Benzene', 'Toluene', 'Xylene']
        
        # Strategy 1: Drop rows where AQI is missing (target variable)
        initial_count = len(self.df_clean)
        self.df_clean = self.df_clean.dropna(subset=['AQI'])
        dropped = initial_count - len(self.df_clean)
        print(f"✅ Dropped {dropped:,} rows with missing AQI values")
        
        # Strategy 2: Impute pollutant values with city-wise median
        print("\n📊 Imputing pollutant values...")
        for col in pollutant_cols:
            if col in self.df_clean.columns:
                # City-wise median imputation
                city_medians = self.df_clean.groupby('City')[col].transform('median')
                self.df_clean[col].fillna(city_medians, inplace=True)
                
                # Global median for remaining missing values
                global_median = self.df_clean[col].median()
                self.df_clean[col].fillna(global_median, inplace=True)
                
                # If still missing (all values were NaN), fill with 0
                self.df_clean[col].fillna(0, inplace=True)
        
        # Verify no missing values remain
        remaining_missing = self.df_clean.isnull().sum().sum()
        print(f"✅ Missing values after imputation: {remaining_missing}")
        print(f"✅ Clean dataset shape: {self.df_clean.shape}")
    
    def feature_engineering(self):
        """Create additional features for anomaly detection"""
        print("\n" + "=" * 80)
        print("🔧 FEATURE ENGINEERING")
        print("=" * 80)
        
        # Convert Date to datetime
        self.df_clean['Date'] = pd.to_datetime(self.df_clean['Date'])
        
        # Extract temporal features
        print("\n📅 Extracting temporal features...")
        self.df_clean['Year'] = self.df_clean['Date'].dt.year
        self.df_clean['Month'] = self.df_clean['Date'].dt.month
        self.df_clean['DayOfWeek'] = self.df_clean['Date'].dt.dayofweek
        self.df_clean['DayOfYear'] = self.df_clean['Date'].dt.dayofyear
        self.df_clean['Quarter'] = self.df_clean['Date'].dt.quarter
        
        # Create season feature
        self.df_clean['Season'] = self.df_clean['Month'].apply(self._get_season)
        
        # Weekend flag
        self.df_clean['IsWeekend'] = (self.df_clean['DayOfWeek'] >= 5).astype(int)
        
        print("   ✅ Temporal features created: Year, Month, DayOfWeek, DayOfYear, Quarter, Season, IsWeekend")
        
        # Create lagged features (previous day values)
        print("\n📊 Creating lagged features...")
        self.df_clean = self.df_clean.sort_values(['City', 'Date'])
        
        # AQI lag features
        self.df_clean['AQI_lag1'] = self.df_clean.groupby('City')['AQI'].shift(1)
        self.df_clean['AQI_lag7'] = self.df_clean.groupby('City')['AQI'].shift(7)
        
        # PM2.5 lag features
        self.df_clean['PM2.5_lag1'] = self.df_clean.groupby('City')['PM2.5'].shift(1)
        
        # Fill NaN in lag features with current values
        self.df_clean['AQI_lag1'].fillna(self.df_clean['AQI'], inplace=True)
        self.df_clean['AQI_lag7'].fillna(self.df_clean['AQI'], inplace=True)
        self.df_clean['PM2.5_lag1'].fillna(self.df_clean['PM2.5'], inplace=True)
        
        print("   ✅ Lagged features created: AQI_lag1, AQI_lag7, PM2.5_lag1")
        
        # Rolling statistics
        print("\n📈 Creating rolling statistics...")
        self.df_clean['AQI_rolling_mean_7'] = self.df_clean.groupby('City')['AQI'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        self.df_clean['AQI_rolling_std_7'] = self.df_clean.groupby('City')['AQI'].transform(
            lambda x: x.rolling(window=7, min_periods=1).std()
        ).fillna(0)
        
        print("   ✅ Rolling features created: AQI_rolling_mean_7, AQI_rolling_std_7")
        
        # Pollutant ratios (capture relationships)
        print("\n🔬 Creating pollutant ratio features...")
        self.df_clean['PM_ratio'] = self.df_clean['PM2.5'] / (self.df_clean['PM10'] + 1e-6)
        self.df_clean['NOx_NO2_ratio'] = self.df_clean['NOx'] / (self.df_clean['NO2'] + 1e-6)
        
        print("   ✅ Ratio features created: PM_ratio, NOx_NO2_ratio")
        
        print(f"\n✅ Feature engineering completed!")
        print(f"   Total features: {len(self.df_clean.columns)}")
    
    def _get_season(self, month):
        """Helper function to determine season from month"""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall/Autumn
    
    def prepare_features(self):
        """Prepare final feature set for modeling"""
        print("\n" + "=" * 80)
        print("🎯 PREPARING FEATURES FOR MODELING")
        print("=" * 80)
        
        # Select features for modeling
        feature_cols = [
            # Pollutants
            'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 
            'O3', 'Benzene', 'Toluene', 'Xylene',
            # Temporal features
            'Month', 'DayOfWeek', 'DayOfYear', 'Quarter', 'Season', 'IsWeekend',
            # Lagged features
            'AQI_lag1', 'AQI_lag7', 'PM2.5_lag1',
            # Rolling features
            'AQI_rolling_mean_7', 'AQI_rolling_std_7',
            # Ratio features
            'PM_ratio', 'NOx_NO2_ratio'
        ]
        
        # Add city encoding (one-hot or label encoding)
        print("\n🏙️  Encoding city feature...")
        self.df_clean['City_Encoded'] = self.label_encoder.fit_transform(self.df_clean['City'])
        feature_cols.append('City_Encoded')
        
        # Target variable
        target_col = 'AQI'
        
        # Create feature matrix
        X = self.df_clean[feature_cols].copy()
        y = self.df_clean[target_col].copy()
        
        print(f"\n✅ Feature matrix prepared:")
        print(f"   Features (X): {X.shape}")
        print(f"   Target (y): {y.shape}")
        print(f"   Feature list: {feature_cols}")
        
        return X, y, feature_cols
    
    def normalize_features(self, X, fit=True):
        """
        Normalize features using StandardScaler
        
        Args:
            X: Feature matrix
            fit: Whether to fit the scaler (True for training data)
        
        Returns:
            Normalized feature matrix
        """
        print("\n" + "=" * 80)
        print("📏 NORMALIZING FEATURES")
        print("=" * 80)
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            print("✅ Scaler fitted and features normalized")
        else:
            X_scaled = self.scaler.transform(X)
            print("✅ Features normalized using existing scaler")
        
        return X_scaled
    
    def save_processed_data(self, X, y, feature_cols):
        """Save processed data and artifacts"""
        print("\n" + "=" * 80)
        print("💾 SAVING PROCESSED DATA")
        print("=" * 80)
        
        # Save clean dataframe
        processed_data_path = DATA_DIR / 'processed_data.csv'
        self.df_clean.to_csv(processed_data_path, index=False)
        print(f"✅ Processed data saved: {processed_data_path}")
        
        # Save feature matrix and target
        features_path = DATA_DIR / 'features.csv'
        target_path = DATA_DIR / 'target.csv'
        pd.DataFrame(X, columns=feature_cols).to_csv(features_path, index=False)
        pd.Series(y, name='AQI').to_csv(target_path, index=False)
        print(f"✅ Features saved: {features_path}")
        print(f"✅ Target saved: {target_path}")
        
        # Save scaler
        scaler_path = MODELS_DIR / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Scaler saved: {scaler_path}")
        
        # Save label encoder
        encoder_path = MODELS_DIR / 'label_encoder.pkl'
        joblib.dump(self.label_encoder, encoder_path)
        print(f"✅ Label encoder saved: {encoder_path}")
        
        # Save feature columns
        feature_cols_path = MODELS_DIR / 'feature_columns.pkl'
        joblib.dump(feature_cols, feature_cols_path)
        print(f"✅ Feature columns saved: {feature_cols_path}")
        
        # Save metadata
        metadata = {
            'total_records': len(self.df_clean),
            'num_features': len(feature_cols),
            'feature_names': feature_cols,
            'cities': self.df_clean['City'].unique().tolist(),
            'date_range': {
                'start': str(self.df_clean['Date'].min()),
                'end': str(self.df_clean['Date'].max())
            },
            'aqi_stats': {
                'mean': float(y.mean()),
                'std': float(y.std()),
                'min': float(y.min()),
                'max': float(y.max())
            }
        }
        
        import json
        metadata_path = DATA_DIR / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"✅ Metadata saved: {metadata_path}")
    
    def run_full_pipeline(self):
        """Execute the complete preprocessing pipeline"""
        print("\n" + "🚀" * 40)
        print("STARTING COMPLETE DATA PREPROCESSING PIPELINE")
        print("🚀" * 40)
        
        # Step 1: Load data
        if not self.load_data():
            print("\n❌ Pipeline failed: Could not load data")
            return False
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Handle missing values
        self.handle_missing_values()
        
        # Step 4: Feature engineering
        self.feature_engineering()
        
        # Step 5: Prepare features
        X, y, feature_cols = self.prepare_features()
        
        # Step 6: Normalize features
        X_scaled = self.normalize_features(X, fit=True)
        
        # Step 7: Save processed data
        self.save_processed_data(X, y, feature_cols)
        
        print("\n" + "=" * 80)
        print("✅ DATA PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\n📊 Summary:")
        print(f"   Records processed: {len(self.df_clean):,}")
        print(f"   Features created: {len(feature_cols)}")
        print(f"   Data saved to: {DATA_DIR}")
        print(f"   Models saved to: {MODELS_DIR}")
        
        return True


def main():
    """Main execution function"""
    # Initialize preprocessor
    preprocessor = AQIDataPreprocessor()
    
    # Run full pipeline
    success = preprocessor.run_full_pipeline()
    
    if success:
        print("\n✨ Ready for anomaly detection modeling!")
        print("📝 Next step: Run src/anomaly_detectors.py")
    else:
        print("\n❌ Preprocessing failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
