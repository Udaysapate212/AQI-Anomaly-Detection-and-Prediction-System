"""
Feature Engineering Utility Module
Provides consistent feature engineering across all dashboard pages
"""

import pandas as pd
import numpy as np
from datetime import datetime


def engineer_features(df):
    """
    Apply consistent feature engineering to a DataFrame
    
    Args:
        df (DataFrame): Input dataframe with pollutant data
        
    Returns:
        DataFrame: DataFrame with engineered features
    """
    df = df.copy()
    
    # Ensure Date column is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        # If no Date column, use current date
        df['Date'] = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    
    # Extract temporal features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Quarter'] = df['Date'].dt.quarter
    
    # Create season feature
    df['Season'] = df['Month'].apply(_get_season)
    
    # Weekend flag
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    # Sort by City and Date for lag features
    df = df.sort_values(['City', 'Date']) if 'City' in df.columns else df.sort_values('Date')
    
    # Create lag features
    if 'City' in df.columns:
        df['AQI_lag1'] = df.groupby('City')['AQI'].shift(1)
        df['AQI_lag7'] = df.groupby('City')['AQI'].shift(7)
        df['PM2.5_lag1'] = df.groupby('City')['PM2.5'].shift(1) if 'PM2.5' in df.columns else 0
        
        # Rolling statistics
        df['AQI_rolling_mean_7'] = df.groupby('City')['AQI'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        df['AQI_rolling_std_7'] = df.groupby('City')['AQI'].transform(
            lambda x: x.rolling(window=7, min_periods=1).std()
        ).fillna(0)
    else:
        # Single city case
        df['AQI_lag1'] = df['AQI'].shift(1)
        df['AQI_lag7'] = df['AQI'].shift(7)
        df['PM2.5_lag1'] = df['PM2.5'].shift(1) if 'PM2.5' in df.columns else 0
        df['AQI_rolling_mean_7'] = df['AQI'].rolling(window=7, min_periods=1).mean()
        df['AQI_rolling_std_7'] = df['AQI'].rolling(window=7, min_periods=1).std().fillna(0)
    
    # Fill NaN in lag features with current values or median
    if 'AQI_lag1' in df.columns:
        df['AQI_lag1'].fillna(df['AQI'].median(), inplace=True)
    if 'AQI_lag7' in df.columns:
        df['AQI_lag7'].fillna(df['AQI'].median(), inplace=True)
    if 'PM2.5_lag1' in df.columns:
        df['PM2.5_lag1'].fillna(df['PM2.5'].median() if 'PM2.5' in df.columns else 0, inplace=True)
    
    # Pollutant ratios
    if 'PM2.5' in df.columns and 'PM10' in df.columns:
        df['PM_ratio'] = df['PM2.5'] / (df['PM10'] + 1e-6)
    else:
        df['PM_ratio'] = 0
        
    if 'NOx' in df.columns and 'NO2' in df.columns:
        df['NOx_NO2_ratio'] = df['NOx'] / (df['NO2'] + 1e-6)
    else:
        df['NOx_NO2_ratio'] = 0
    
    # Encode City if present
    if 'City' in df.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['City_Encoded'] = le.fit_transform(df['City'])
    else:
        df['City_Encoded'] = 0
    
    return df


def _get_season(month):
    """Helper function to determine season from month"""
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Fall/Autumn


def prepare_single_prediction_features(pollutants, city='Delhi', date=None, use_onehot_city=True, simple_features=False):
    """
    Prepare features for a single prediction
    
    Args:
        pollutants (dict): Dictionary of pollutant values
        city (str): City name
        date (datetime): Date for prediction
        use_onehot_city (bool): If True, use one-hot encoding for city; if False, use label encoding
        simple_features (bool): If True, only create basic features matching trained model (Month, DayOfWeek, AQI_lag1, PM2.5_lag1)
        
    Returns:
        DataFrame: Single row with all required features
    """
    if date is None:
        date = datetime.now()
    
    # Create base dataframe with pollutants
    data = {
        'PM2.5': [float(pollutants.get('PM2.5', 50))],
        'PM10': [float(pollutants.get('PM10', 75))],
        'NO': [float(pollutants.get('NO', 10))],
        'NO2': [float(pollutants.get('NO2', 40))],
        'NOx': [float(pollutants.get('NOx', 50))],
        'NH3': [float(pollutants.get('NH3', 15))],
        'CO': [float(pollutants.get('CO', 1.0))],
        'SO2': [float(pollutants.get('SO2', 20))],
        'O3': [float(pollutants.get('O3', 50))],
        'Benzene': [float(pollutants.get('Benzene', 1.0))],
        'Toluene': [float(pollutants.get('Toluene', 5.0))],
        'Xylene': [float(pollutants.get('Xylene', 2.0))]
    }
    
    df = pd.DataFrame(data)
    
    if simple_features:
        # Match exactly what the trained model expects
        # Add temporal features (only Month and DayOfWeek)
        df['Month'] = date.month
        df['DayOfWeek'] = date.weekday()
        
        # Add lag features (use provided AQI or estimate from pollutants)
        df['AQI_lag1'] = float(pollutants.get('AQI', pollutants.get('PM2.5', 50) * 2))
        df['PM2.5_lag1'] = float(pollutants.get('PM2.5', 50))
        
        # Handle city encoding
        if use_onehot_city:
            # One-hot encode city - MUST match training data exactly
            all_cities = [
                'Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru', 'Bhopal',
                'Chandigarh', 'Chennai', 'Coimbatore', 'Delhi', 'Ernakulam',
                'Gurugram', 'Guwahati', 'Hyderabad', 'Jaipur', 'Jorapokhar', 'Kochi',
                'Kolkata', 'Lucknow', 'Mumbai', 'Patna', 'Shillong', 'Talcher',
                'Thiruvananthapuram', 'Visakhapatnam'
            ]
            
            # Create one-hot columns
            for c in all_cities:
                df[f'City_{c}'] = 1 if c == city else 0
    else:
        # Full feature engineering
        df['Date'] = date
        df['City'] = city
        df['AQI'] = pollutants.get('AQI', 100)
        
        df = engineer_features(df)
        
        if use_onehot_city:
            all_cities = [
                'Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru', 'Bhopal',
                'Chandigarh', 'Chennai', 'Coimbatore', 'Delhi', 'Ernakulam',
                'Gurugram', 'Guwahati', 'Hyderabad', 'Jaipur', 'Jorapokhar', 'Kochi',
                'Kolkata', 'Lucknow', 'Mumbai', 'Patna', 'Shillong', 'Talcher',
                'Thiruvananthapuram', 'Visakhapatnam'
            ]
            
            for c in all_cities:
                df[f'City_{c}'] = 1 if c == city else 0
            
            if 'City_Encoded' in df.columns:
                df = df.drop('City_Encoded', axis=1)
    
    return df


def get_required_feature_columns(include_city_onehot=True):
    """
    Get the list of EXACT feature columns the trained model expects
    
    Args:
        include_city_onehot (bool): If True, include one-hot encoded city columns (RECOMMENDED - matches training)
        
    Returns:
        list: List of EXACTLY 41 feature column names as used in training
    """
    # EXACTLY 41 features as trained
    base_features = [
        # Pollutants (12 features)
        'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 
        'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene',
        # Temporal features (2 features ONLY - NOT 6!)
        'DayOfWeek', 'Month',
        # Lag features (2 features ONLY - NOT 3!)
        'AQI_lag1', 'PM2.5_lag1'
    ]
    
    if include_city_onehot:
        # Add one-hot encoded city columns (25 features)
        all_cities = [
            'Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru', 'Bhopal',
            'Chandigarh', 'Chennai', 'Coimbatore', 'Delhi', 'Ernakulam',
            'Gurugram', 'Guwahati', 'Hyderabad', 'Jaipur', 'Jorapokhar', 'Kochi',
            'Kolkata', 'Lucknow', 'Mumbai', 'Patna', 'Shillong', 'Talcher',
            'Thiruvananthapuram', 'Visakhapatnam'
        ]
        city_features = [f'City_{c}' for c in all_cities]
        return base_features + city_features
    else:
        # Use label encoding (1 feature)
        return base_features + ['City_Encoded']
