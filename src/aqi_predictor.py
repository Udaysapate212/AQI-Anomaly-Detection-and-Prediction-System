"""
Comprehensive AQI Prediction System
Implements multiple ML models for AQI prediction and classification

Features:
- Multiple regression models (Random Forest, Gradient Boosting, KNN, Decision Tree, Linear Regression)
- Multiple classification models (Random Forest, Gradient Boosting, KNN, Decision Tree, Logistic Regression, Naive Bayes)
- AdaBoost ensemble methods
- Clustering analysis (K-Means, DBSCAN, Hierarchical)
- Model comparison and evaluation
- Feature importance analysis
- Cross-validation
"""

import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, silhouette_score
)

# Regression Models
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor

# Classification Models
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Clustering Models
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AQIPredictorSystem:
    """
    Comprehensive AQI Prediction System with Multiple ML Models
    """
    
    def __init__(self, models_dir='models'):
        """
        Initialize the prediction system
        
        Args:
            models_dir (str): Directory to save/load models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize model dictionaries
        self.regression_models = {}
        self.classification_models = {}
        self.clustering_models = {}
        
        # Model performance storage
        self.regression_results = {}
        self.classification_results = {}
        self.clustering_results = {}
        
        # Best models
        self.best_regressor = None
        self.best_classifier = None
        
        # Data preprocessing
        self.scaler = None
        self.feature_columns = None
        self.aqi_bucket_mapping = None

        # Metadata for inference-time preprocessing
        self.regressor_uses_scaled = False
        self.classifier_uses_scaled = False
        self.metadata_path = self.models_dir / 'model_metadata.json'
        
        logger.info("AQI Prediction System initialized")
    
    def _define_regression_models(self):
        """Define all regression models"""
        return {
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, 
                random_state=42
            ),
            'AdaBoost': AdaBoostRegressor(
                n_estimators=100,
                random_state=42
            ),
            'Decision Tree': DecisionTreeRegressor(
                max_depth=10, 
                random_state=42
            ),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(
                alpha=1.0, 
                random_state=42
            ),
            'KNN Regressor': KNeighborsRegressor(
                n_neighbors=5
            )
        }
    
    def _define_classification_models(self):
        """Define all classification models"""
        return {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, 
                random_state=42
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100,
                random_state=42
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10, 
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=42
            ),
            'KNN Classifier': KNeighborsClassifier(
                n_neighbors=5
            ),
            'Naive Bayes': GaussianNB()
        }
    
    def _define_clustering_models(self):
        """Define clustering models"""
        return {
            'K-Means': KMeans(
                n_clusters=6,  # 6 AQI categories
                random_state=42,
                n_init=10
            ),
            'DBSCAN': DBSCAN(
                eps=50,
                min_samples=5
            ),
            'Hierarchical': AgglomerativeClustering(
                n_clusters=6
            )
        }
    
    def prepare_features(self, df, target_col='AQI', bucket_col='AQI_Bucket'):
        """
        Prepare features for model training
        
        Args:
            df (DataFrame): Input data
            target_col (str): Regression target column
            bucket_col (str): Classification target column
            
        Returns:
            tuple: X, y_reg, y_cls
        """
        logger.info("Preparing features...")
        
        # Convert Date to datetime if exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['Month'] = df['Date'].dt.month
        
        # Sort by City and Date for lag features
        if 'City' in df.columns and 'Date' in df.columns:
            df = df.sort_values(['City', 'Date']).reset_index(drop=True)
        
        # Create lag features
        if 'City' in df.columns:
            df['AQI_lag1'] = df.groupby('City')[target_col].shift(1)
            df['PM2.5_lag1'] = df.groupby('City')['PM2.5'].shift(1)
        else:
            df['AQI_lag1'] = df[target_col].shift(1)
            df['PM2.5_lag1'] = df['PM2.5'].shift(1)
        
        # Drop rows with NaN in lag features
        df = df.dropna(subset=['AQI_lag1', 'PM2.5_lag1'])
        
        # One-hot encode City if exists
        if 'City' in df.columns:
            city_dummies = pd.get_dummies(df['City'], prefix='City')
            df = pd.concat([df, city_dummies], axis=1)
        
        # Define feature columns
        pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 
                         'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
        
        time_cols = ['DayOfWeek', 'Month'] if 'DayOfWeek' in df.columns else []
        lag_cols = ['AQI_lag1', 'PM2.5_lag1']
        
        city_cols = [col for col in df.columns if col.startswith('City_')]
        
        self.feature_columns = pollutant_cols + time_cols + lag_cols + city_cols
        
        # Prepare X and y
        X = df[self.feature_columns]
        y_reg = df[target_col]
        
        # Encode AQI buckets
        if bucket_col in df.columns:
            self.aqi_bucket_mapping = {
                'Good': 0,
                'Satisfactory': 1,
                'Moderate': 2,
                'Poor': 3,
                'Very Poor': 4,
                'Severe': 5
            }
            y_cls = df[bucket_col].map(self.aqi_bucket_mapping)
        else:
            # Create buckets from AQI values
            y_cls = pd.cut(
                y_reg,
                bins=[0, 50, 100, 200, 300, 400, 500],
                labels=[0, 1, 2, 3, 4, 5]
            )
        
        # Drop any remaining NaN values in features and targets
        valid_mask = ~(X.isna().any(axis=1) | y_reg.isna() | y_cls.isna())
        X = X[valid_mask]
        y_reg = y_reg[valid_mask]
        y_cls = y_cls[valid_mask]
        df = df[valid_mask]
        
        # Fill any remaining NaN with 0 (shouldn't happen but safety measure)
        X = X.fillna(0)
        
        logger.info(f"Features prepared: {X.shape}")
        return X, y_reg, y_cls, df
    
    def train_all_models(self, X_train, X_test, y_reg_train, y_reg_test, 
                        y_cls_train, y_cls_test):
        """
        Train all regression and classification models
        
        Args:
            X_train, X_test: Feature matrices
            y_reg_train, y_reg_test: Regression targets
            y_cls_train, y_cls_test: Classification targets
        """
        logger.info("="*80)
        logger.info("Training All Models")
        logger.info("="*80)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train regression models
        logger.info("\n[Regression Models]")
        self.regression_models = self._define_regression_models()
        
        scaled_regression_models = {'Linear Regression', 'Ridge Regression', 'KNN Regressor'}

        for name, model in self.regression_models.items():
            logger.info(f"\nTraining {name}...")

            use_scaled = name in scaled_regression_models
            
            # Use scaled data for some models
            if use_scaled:
                model.fit(X_train_scaled, y_reg_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_reg_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_reg_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_reg_test, y_pred)
            r2 = r2_score(y_reg_test, y_pred)
            mape = np.mean(np.abs((y_reg_test - y_pred) / y_reg_test)) * 100
            
            self.regression_results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'uses_scaled': use_scaled
            }
            
            logger.info(f"  R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
        
        # Train classification models
        logger.info("\n[Classification Models]")
        self.classification_models = self._define_classification_models()
        
        scaled_classification_models = {'Logistic Regression', 'KNN Classifier', 'Naive Bayes'}

        for name, model in self.classification_models.items():
            logger.info(f"\nTraining {name}...")

            use_scaled = name in scaled_classification_models
            
            # Use scaled data for some models
            if use_scaled:
                model.fit(X_train_scaled, y_cls_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_cls_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_cls_test, y_pred)
            precision = precision_score(y_cls_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_cls_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_cls_test, y_pred, average='weighted', zero_division=0)
            
            self.classification_results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'uses_scaled': use_scaled
            }
            
            logger.info(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Select best models
        self.best_regressor = max(
            self.regression_results.items(), 
            key=lambda x: x[1]['r2']
        )
        self.regressor_uses_scaled = self.best_regressor[1].get('uses_scaled', False)
        self.best_classifier = max(
            self.classification_results.items(), 
            key=lambda x: x[1]['accuracy']
        )
        self.classifier_uses_scaled = self.best_classifier[1].get('uses_scaled', False)
        
        logger.info(f"\n✅ Best Regressor: {self.best_regressor[0]} (R²={self.best_regressor[1]['r2']:.4f})")
        logger.info(f"✅ Best Classifier: {self.best_classifier[0]} (Accuracy={self.best_classifier[1]['accuracy']:.4f})")
    
    def train_clustering_models(self, X):
        """
        Train clustering models for pattern analysis
        
        Args:
            X: Feature matrix
        """
        logger.info("\n[Clustering Models]")
        
        # Scale features for clustering
        X_scaled = self.scaler.transform(X) if self.scaler else StandardScaler().fit_transform(X)
        
        self.clustering_models = self._define_clustering_models()
        
        for name, model in self.clustering_models.items():
            logger.info(f"\nTraining {name}...")
            
            try:
                labels = model.fit_predict(X_scaled)
                
                # Calculate silhouette score
                if len(set(labels)) > 1:  # Need at least 2 clusters
                    silhouette = silhouette_score(X_scaled, labels)
                else:
                    silhouette = -1
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                self.clustering_results[name] = {
                    'model': model,
                    'labels': labels,
                    'n_clusters': n_clusters,
                    'silhouette_score': silhouette
                }
                
                logger.info(f"  Clusters: {n_clusters}, Silhouette Score: {silhouette:.4f}")
            except Exception as e:
                logger.error(f"  Error training {name}: {e}")
    
    def save_models(self):
        """Save all trained models"""
        logger.info("\nSaving models...")
        
        # Save regression models
        for name, results in self.regression_results.items():
            filename = self.models_dir / f"{name.lower().replace(' ', '_')}_regressor.joblib"
            joblib.dump(results['model'], filename)
        
        # Save classification models
        for name, results in self.classification_results.items():
            filename = self.models_dir / f"{name.lower().replace(' ', '_')}_classifier.joblib"
            joblib.dump(results['model'], filename)
        
        # Save best models
        joblib.dump(self.best_regressor[1]['model'], self.models_dir / 'best_regressor.joblib')
        joblib.dump(self.best_classifier[1]['model'], self.models_dir / 'best_classifier.joblib')
        
        # Save preprocessing objects
        joblib.dump(self.scaler, self.models_dir / 'scaler.joblib')
        joblib.dump(self.feature_columns, self.models_dir / 'feature_columns.joblib')
        
        if self.aqi_bucket_mapping:
            joblib.dump(self.aqi_bucket_mapping, self.models_dir / 'aqi_bucket_mapping.joblib')
        
        # Save comparison results
        reg_df = pd.DataFrame(self.regression_results).T
        reg_df.to_csv(self.models_dir / 'regression_comparison.csv')
        
        cls_df = pd.DataFrame(self.classification_results).T
        cls_df.to_csv(self.models_dir / 'classification_comparison.csv')
        
        # Persist metadata for inference
        metadata = {
            'best_regressor_name': self.best_regressor[0] if self.best_regressor else None,
            'best_classifier_name': self.best_classifier[0] if self.best_classifier else None,
            'regressor_uses_scaled': self.regressor_uses_scaled,
            'classifier_uses_scaled': self.classifier_uses_scaled
        }
        try:
            with open(self.metadata_path, 'w', encoding='utf-8') as meta_file:
                json.dump(metadata, meta_file, indent=2)
        except Exception as exc:
            logger.error(f"Failed to write model metadata: {exc}")
        
        logger.info(f"✅ All models saved to {self.models_dir}")
    
    def load_models(self):
        """Load pre-trained models"""
        logger.info("Loading models...")
        
        try:
            self.best_regressor = ('Best', {
                'model': joblib.load(self.models_dir / 'best_regressor.joblib')
            })
            self.best_classifier = ('Best', {
                'model': joblib.load(self.models_dir / 'best_classifier.joblib')
            })
            self.scaler = joblib.load(self.models_dir / 'scaler.joblib')
            self.feature_columns = joblib.load(self.models_dir / 'feature_columns.joblib')
            self.aqi_bucket_mapping = joblib.load(self.models_dir / 'aqi_bucket_mapping.joblib')

            # Load metadata if available, otherwise infer
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r', encoding='utf-8') as meta_file:
                    metadata = json.load(meta_file)
                self.regressor_uses_scaled = metadata.get('regressor_uses_scaled', None)
                self.classifier_uses_scaled = metadata.get('classifier_uses_scaled', None)
            else:
                self.regressor_uses_scaled = None
                self.classifier_uses_scaled = None

            if self.regressor_uses_scaled is None:
                self.regressor_uses_scaled = self._infer_scaler_requirement(
                    self.best_regressor[1]['model'], task='regression'
                )
            if self.classifier_uses_scaled is None:
                self.classifier_uses_scaled = self._infer_scaler_requirement(
                    self.best_classifier[1]['model'], task='classification'
                )
            
            logger.info("✅ Models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict(self, features):
        """
        Make predictions using best models
        
        Args:
            features (DataFrame or array): Input features
            
        Returns:
            dict: Predictions from regression and classification
        """
        if self.best_regressor is None or self.best_classifier is None:
            raise ValueError("Models not trained. Call train_all_models() or load_models() first.")
        
        # Ensure features are in correct order
        if isinstance(features, pd.DataFrame):
            features = features[self.feature_columns]
        else:
            features = pd.DataFrame(features, columns=self.feature_columns)

        # Prepare scaled copy only when needed to avoid mismatched preprocessing
        features_scaled = None
        if self.scaler:
            features_scaled = self.scaler.transform(features)

        reg_model = self.best_regressor[1]['model']
        cls_model = self.best_classifier[1]['model']
        
        reg_input = features_scaled if self.regressor_uses_scaled else features
        cls_input = features_scaled if self.classifier_uses_scaled else features
        
        # If scaling required but scaler missing, raise helpful error
        if self.regressor_uses_scaled and features_scaled is None:
            raise ValueError("Scaler not available for regression model that requires scaled features.")
        if self.classifier_uses_scaled and features_scaled is None:
            raise ValueError("Scaler not available for classification model that requires scaled features.")
        
        # Make predictions
        aqi_pred = reg_model.predict(reg_input)[0]
        bucket_pred = cls_model.predict(cls_input)[0]
        
        # Get bucket name
        reverse_mapping = {v: k for k, v in self.aqi_bucket_mapping.items()}
        bucket_name = reverse_mapping.get(bucket_pred, 'Unknown')
        
        return {
            'predicted_aqi': aqi_pred,
            'predicted_bucket_code': bucket_pred,
            'predicted_bucket_name': bucket_name
        }

    def _infer_scaler_requirement(self, model, task='regression'):
        """Infer if a model expects scaled features based on its class."""
        if model is None:
            return False
        if task == 'regression':
            scaled_types = (LinearRegression, Ridge, KNeighborsRegressor)
        else:
            scaled_types = (LogisticRegression, KNeighborsClassifier, GaussianNB)
        return isinstance(model, scaled_types)


# Example usage
if __name__ == "__main__":
    # Initialize system - use correct path relative to src/
    predictor = AQIPredictorSystem(models_dir='models')
    
    # Load dataset
    data_path = Path('data/City_Day.csv')
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        print(f"Loaded dataset: {df.shape}")
        
        # Prepare features
        X, y_reg, y_cls, df_processed = predictor.prepare_features(df)
        
        # Split data
        X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
            X, y_reg, y_cls, test_size=0.2, random_state=42, stratify=y_cls
        )
        
        # Train all models
        predictor.train_all_models(
            X_train, X_test, 
            y_reg_train, y_reg_test,
            y_cls_train, y_cls_test
        )
        
        # Train clustering
        predictor.train_clustering_models(X)
        
        # Save models
        predictor.save_models()
        
        print("\n✅ All models trained and saved!")
    else:
        print(f"❌ Dataset not found at {data_path}")
