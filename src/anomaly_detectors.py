"""
Anomaly Detection Module for AQI Monitoring System

Implements three state-of-the-art anomaly detection algorithms:
1. Isolation Forest - Ensemble-based method
2. Local Outlier Factor (LOF) - Density-based method
3. Autoencoder Neural Network - Deep learning method

Author: TY Sem 5 AIML Student
Date: November 2025
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# Deep Learning - Lazy import to avoid mutex lock issues
TENSORFLOW_AVAILABLE = False
tf = None
keras = None
layers = None
Model = None
EarlyStopping = None

def _import_tensorflow():
    """Lazy import TensorFlow only when needed"""
    global TENSORFLOW_AVAILABLE, tf, keras, layers, Model, EarlyStopping
    
    if TENSORFLOW_AVAILABLE:
        return True
    
    try:
        import os
        # Fix TensorFlow threading issues on macOS
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        
        import tensorflow as tensorflow_module
        tf = tensorflow_module
        
        # Disable GPU/Metal on macOS to prevent mutex locks
        tf.config.set_visible_devices([], 'GPU')
        # Set threading to avoid conflicts
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        
        from tensorflow import keras as keras_module
        from tensorflow.keras import layers as layers_module, Model as Model_class
        from tensorflow.keras.callbacks import EarlyStopping as EarlyStopping_class
        
        keras = keras_module
        layers = layers_module
        Model = Model_class
        EarlyStopping = EarlyStopping_class
        
        TENSORFLOW_AVAILABLE = True
        return True
    except ImportError:
        print("⚠️  TensorFlow not available. Autoencoder will be skipped.")
        return False
    except Exception as e:
        print(f"⚠️  TensorFlow initialization error: {e}. Autoencoder will be skipped.")
        return False

# warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


class IsolationForestDetector:
    """Isolation Forest anomaly detector"""
    
    def __init__(self, contamination=0.1, random_state=42):
        """
        Initialize Isolation Forest detector
        
        Args:
            contamination (float): Expected proportion of anomalies (default: 0.1 = 10%)
            random_state (int): Random seed for reproducibility
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto',
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1,
            verbose=0
        )
        self.contamination = contamination
        self.name = "Isolation Forest"
    
    def fit(self, X):
        """Train the model"""
        print(f"\n🌲 Training {self.name}...")
        self.model.fit(X)
        print(f"   ✅ {self.name} trained successfully!")
        return self
    
    def predict(self, X):
        """
        Predict anomalies
        
        Returns:
            predictions: 1 for normal, -1 for anomaly
            scores: Anomaly scores (lower = more anomalous)
        """
        predictions = self.model.predict(X)
        scores = self.model.score_samples(X)
        
        # Convert to binary: 1 for anomaly, 0 for normal
        binary_predictions = (predictions == -1).astype(int)
        
        return binary_predictions, scores
    
    def save(self, path):
        """Save model to disk"""
        joblib.dump(self.model, path)
        print(f"   💾 {self.name} saved to {path}")


class LOFDetector:
    """Local Outlier Factor anomaly detector"""
    
    def __init__(self, contamination=0.1, n_neighbors=20):
        """
        Initialize LOF detector
        
        Args:
            contamination (float): Expected proportion of anomalies
            n_neighbors (int): Number of neighbors to consider
        """
        self.model = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=n_neighbors,
            algorithm='auto',
            leaf_size=30,
            metric='minkowski',
            p=2,
            n_jobs=-1,
            novelty=True  # Allow predict on new data
        )
        self.contamination = contamination
        self.name = "Local Outlier Factor (LOF)"
    
    def fit(self, X):
        """Train the model"""
        print(f"\n🔍 Training {self.name}...")
        self.model.fit(X)
        print(f"   ✅ {self.name} trained successfully!")
        return self
    
    def predict(self, X):
        """Predict anomalies"""
        predictions = self.model.predict(X)
        scores = self.model.score_samples(X)
        
        # Convert to binary
        binary_predictions = (predictions == -1).astype(int)
        
        return binary_predictions, scores
    
    def save(self, path):
        """Save model to disk"""
        joblib.dump(self.model, path)
        print(f"   💾 {self.name} saved to {path}")


class AutoencoderDetector:
    """Autoencoder Neural Network anomaly detector"""
    
    def __init__(self, contamination=0.1, encoding_dim=10, random_state=42):
        """
        Initialize Autoencoder detector
        
        Args:
            contamination (float): Expected proportion of anomalies
            encoding_dim (int): Dimension of encoded representation
            random_state (int): Random seed
        """
        # Lazy import TensorFlow
        if not _import_tensorflow():
            raise ImportError("TensorFlow is required for Autoencoder")
        
        self.contamination = contamination
        self.encoding_dim = encoding_dim
        self.random_state = random_state
        self.model = None
        self.threshold = None
        self.name = "Autoencoder Neural Network"
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def build_model(self, input_dim):
        """Build autoencoder architecture"""
        # Clear any previous sessions
        tf.keras.backend.clear_session()
        
        # Encoder
        encoder_input = layers.Input(shape=(input_dim,), name='input')
        encoded = layers.Dense(64, activation='relu', name='encoder_layer1')(encoder_input)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(32, activation='relu', name='encoder_layer2')(encoded)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='bottleneck')(encoded)
        
        # Decoder
        decoded = layers.Dense(32, activation='relu', name='decoder_layer1')(encoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dense(64, activation='relu', name='decoder_layer2')(decoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(input_dim, activation='linear', name='output')(decoded)
        
        # Autoencoder model
        autoencoder = Model(inputs=encoder_input, outputs=decoded, name='autoencoder')
        
        # Compile
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return autoencoder
    
    def fit(self, X, epochs=50, batch_size=256, validation_split=0.2):
        """Train the autoencoder"""
        print(f"\n🧠 Training {self.name}...")
        
        # Build model
        self.model = self.build_model(X.shape[1])
        
        # Print architecture
        print(f"   Model architecture:")
        print(f"   Input: {X.shape[1]} features")
        print(f"   Encoder: 64 → 32 → {self.encoding_dim}")
        print(f"   Decoder: 32 → 64 → {X.shape[1]}")
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train
        history = self.model.fit(
            X, X,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Calculate reconstruction errors on training data
        reconstructions = self.model.predict(X, verbose=0)
        reconstruction_errors = np.mean(np.square(X - reconstructions), axis=1)
        
        # Set threshold at the specified percentile
        threshold_percentile = 100 * (1 - self.contamination)
        self.threshold = np.percentile(reconstruction_errors, threshold_percentile)
        
        print(f"   ✅ {self.name} trained successfully!")
        print(f"   Final training loss: {history.history['loss'][-1]:.6f}")
        print(f"   Final validation loss: {history.history['val_loss'][-1]:.6f}")
        print(f"   Anomaly threshold: {self.threshold:.6f}")
        
        return self
    
    def predict(self, X):
        """Predict anomalies based on reconstruction error"""
        # Reconstruct
        reconstructions = self.model.predict(X, verbose=0)
        
        # Calculate reconstruction errors
        scores = np.mean(np.square(X - reconstructions), axis=1)
        
        # Predict anomalies
        binary_predictions = (scores > self.threshold).astype(int)
        
        # Return negative scores for consistency (lower = more anomalous)
        return binary_predictions, -scores
    
    def save(self, path):
        """Save model to disk"""
        # Save Keras model
        model_path = str(path).replace('.pkl', '.h5')
        self.model.save(model_path)
        
        # Save threshold
        metadata = {
            'threshold': float(self.threshold),
            'encoding_dim': self.encoding_dim,
            'contamination': self.contamination
        }
        metadata_path = str(path).replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        print(f"   💾 {self.name} saved to {model_path}")


class AnomalyDetectionSystem:
    """Unified system for training and comparing multiple anomaly detectors"""
    
    def __init__(self, contamination=0.1):
        """
        Initialize the detection system
        
        Args:
            contamination (float): Expected proportion of anomalies
        """
        self.contamination = contamination
        self.detectors = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.feature_names = None
    
    def load_data(self):
        """Load preprocessed data"""
        print("=" * 80)
        print("📊 LOADING PREPROCESSED DATA")
        print("=" * 80)
        
        try:
            # Load features
            features_path = DATA_DIR / 'features.csv'
            X = pd.read_csv(features_path).values
            
            # Load feature names
            feature_cols_path = MODELS_DIR / 'feature_columns.pkl'
            self.feature_names = joblib.load(feature_cols_path)
            
            print(f"✅ Data loaded successfully!")
            print(f"   Shape: {X.shape}")
            print(f"   Features: {len(self.feature_names)}")
            
            # Split into train/test
            self.X_train, self.X_test = train_test_split(
                X, test_size=0.2, random_state=42
            )
            
            print(f"   Train set: {self.X_train.shape}")
            print(f"   Test set: {self.X_test.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            print(f"   Please run data_preprocessing.py first")
            return False
    
    def initialize_detectors(self):
        """Initialize all anomaly detection models"""
        print("\n" + "=" * 80)
        print("🤖 INITIALIZING ANOMALY DETECTORS")
        print("=" * 80)
        
        # Isolation Forest
        self.detectors['isolation_forest'] = IsolationForestDetector(
            contamination=self.contamination
        )
        print(f"✅ {self.detectors['isolation_forest'].name} initialized")
        
        # LOF
        self.detectors['lof'] = LOFDetector(
            contamination=self.contamination,
            n_neighbors=20
        )
        print(f"✅ {self.detectors['lof'].name} initialized")
        
        # Autoencoder - DISABLED due to TensorFlow mutex lock issues on macOS
        # Uncomment below to enable (requires fixing TensorFlow installation)
        # try:
        #     self.detectors['autoencoder'] = AutoencoderDetector(
        #         contamination=self.contamination,
        #         encoding_dim=10
        #     )
        #     print(f"✅ {self.detectors['autoencoder'].name} initialized")
        # except (ImportError, Exception) as e:
        #     print(f"⚠️  Autoencoder skipped: {str(e)}")
        print(f"⚠️  Autoencoder disabled (TensorFlow has mutex lock issues on this system)")
    
    def train_all_models(self):
        """Train all detectors"""
        print("\n" + "=" * 80)
        print("🚀 TRAINING ALL ANOMALY DETECTORS")
        print("=" * 80)
        
        for name, detector in self.detectors.items():
            try:
                if name == 'autoencoder':
                    detector.fit(self.X_train, epochs=50, batch_size=256)
                else:
                    detector.fit(self.X_train)
            except Exception as e:
                print(f"❌ Error training {detector.name}: {str(e)}")
    
    def evaluate_detectors(self):
        """Evaluate all detectors on test set"""
        print("\n" + "=" * 80)
        print("📊 EVALUATING ANOMALY DETECTORS")
        print("=" * 80)
        
        for name, detector in self.detectors.items():
            print(f"\n{detector.name}:")
            print("-" * 60)
            
            try:
                # Predict on test set
                predictions, scores = detector.predict(self.X_test)
                
                # Calculate metrics
                num_anomalies = predictions.sum()
                anomaly_rate = (num_anomalies / len(predictions)) * 100
                
                print(f"   Anomalies detected: {num_anomalies:,} ({anomaly_rate:.2f}%)")
                print(f"   Normal instances: {len(predictions) - num_anomalies:,}")
                
                # Store results
                self.results[name] = {
                    'predictions': predictions,
                    'scores': scores,
                    'num_anomalies': int(num_anomalies),
                    'anomaly_rate': float(anomaly_rate),
                    'model': detector
                }
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
    
    def compare_detectors(self):
        """Compare all detectors and find consensus"""
        print("\n" + "=" * 80)
        print("🔍 COMPARING DETECTORS")
        print("=" * 80)
        
        # Create comparison dataframe
        comparison_data = []
        
        for name, result in self.results.items():
            comparison_data.append({
                'Detector': self.detectors[name].name,
                'Anomalies Detected': result['num_anomalies'],
                'Anomaly Rate (%)': f"{result['anomaly_rate']:.2f}",
                'Normal Instances': len(result['predictions']) - result['num_anomalies']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + comparison_df.to_string(index=False))
        
        # Consensus anomalies (detected by at least 2 models)
        if len(self.results) >= 2:
            print("\n" + "-" * 80)
            print("🤝 CONSENSUS ANALYSIS")
            print("-" * 80)
            
            predictions_array = np.array([r['predictions'] for r in self.results.values()])
            consensus = np.sum(predictions_array, axis=0)
            
            unanimous = (consensus == len(self.results)).sum()
            majority = (consensus >= 2).sum()
            print(f"   Unanimous anomalies (all models agree): {unanimous:,}")
            print(f"   Majority anomalies (2+ models agree): {majority:,}")
            print(f"   Percentage of unanimous: {(unanimous/len(consensus)*100):.2f}%")
    
    def save_all_models(self):
        """Save all trained models"""
        print("\n" + "=" * 80)
        print("💾 SAVING MODELS")
        print("=" * 80)
        
        for name, detector in self.detectors.items():
            model_path = MODELS_DIR / f'{name}_model.pkl'
            detector.save(model_path)
        
        # Save results summary
        results_summary = {
            'contamination': self.contamination,
            'train_size': len(self.X_train),
            'test_size': len(self.X_test),
            'detectors': {}
        }
        
        for name, result in self.results.items():
            results_summary['detectors'][name] = {
                'num_anomalies': result['num_anomalies'],
                'anomaly_rate': result['anomaly_rate']
            }
        
        summary_path = RESULTS_DIR / 'detection_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=4)
        
        print(f"\n✅ Results summary saved to {summary_path}")
    
    def visualize_results(self):
        """Create visualizations of anomaly detection results"""
        print("\n" + "=" * 80)
        print("📊 GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Anomaly Detection Results Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Anomaly counts
        ax1 = axes[0, 0]
        names = [self.detectors[name].name for name in self.results.keys()]
        counts = [self.results[name]['num_anomalies'] for name in self.results.keys()]
        
        bars = ax1.bar(range(len(names)), counts, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax1.set_xlabel('Detector', fontsize=12)
        ax1.set_ylabel('Number of Anomalies', fontsize=12)
        ax1.set_title('Anomalies Detected by Each Model', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=15, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Anomaly rates
        ax2 = axes[0, 1]
        rates = [self.results[name]['anomaly_rate'] for name in self.results.keys()]
        
        bars2 = ax2.bar(range(len(names)), rates, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax2.set_xlabel('Detector', fontsize=12)
        ax2.set_ylabel('Anomaly Rate (%)', fontsize=12)
        ax2.set_title('Anomaly Detection Rate', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=15, ha='right')
        ax2.axhline(y=self.contamination * 100, color='red', linestyle='--', 
                   label=f'Expected Rate ({self.contamination*100:.1f}%)')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontsize=10)
        
        # Plot 3: Score distribution (first detector)
        ax3 = axes[1, 0]
        first_detector = list(self.results.keys())[0]
        scores = self.results[first_detector]['scores']
        predictions = self.results[first_detector]['predictions']
        
        ax3.hist(scores[predictions == 0], bins=50, alpha=0.7, label='Normal', color='green')
        ax3.hist(scores[predictions == 1], bins=50, alpha=0.7, label='Anomaly', color='red')
        ax3.set_xlabel('Anomaly Score', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title(f'Score Distribution - {self.detectors[first_detector].name}', 
                     fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Plot 4: Consensus heatmap
        ax4 = axes[1, 1]
        if len(self.results) >= 2:
            predictions_array = np.array([r['predictions'] for r in self.results.values()])
            consensus = np.sum(predictions_array, axis=0)
            
            consensus_counts = np.bincount(consensus, minlength=len(self.results)+1)
            labels = [f'{i} models' for i in range(len(self.results)+1)]
            
            bars4 = ax4.bar(range(len(consensus_counts)), consensus_counts, 
                           color=['green', 'yellow', 'orange', 'red'][:len(consensus_counts)])
            ax4.set_xlabel('Number of Models Detecting Anomaly', fontsize=12)
            ax4.set_ylabel('Number of Instances', fontsize=12)
            ax4.set_title('Consensus Analysis', fontsize=14, fontweight='bold')
            ax4.set_xticks(range(len(labels)))
            ax4.set_xticklabels(labels, rotation=15, ha='right')
            ax4.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars4:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        viz_path = RESULTS_DIR / 'anomaly_detection_comparison.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to {viz_path}")
        
        plt.close()
    
    def run_full_pipeline(self):
        """Execute complete anomaly detection pipeline"""
        print("\n" + "🚀" * 40)
        print("STARTING ANOMALY DETECTION PIPELINE")
        print("🚀" * 40)
        
        # Load data
        if not self.load_data():
            return False
        
        # Initialize detectors
        self.initialize_detectors()
        
        # Train models
        self.train_all_models()
        
        # Evaluate
        self.evaluate_detectors()
        
        # Compare
        self.compare_detectors()
        
        # Save
        self.save_all_models()
        
        # Visualize
        self.visualize_results()
        
        print("\n" + "=" * 80)
        print("✅ ANOMALY DETECTION PIPELINE COMPLETED!")
        print("=" * 80)
        
        return True


def main():
    """Main execution function"""
    import argparse
    print("Welcome to anomaly detector")
    
    parser = argparse.ArgumentParser(description='AQI Anomaly Detection System')
    parser.add_argument('--contamination', type=float, default=0.1,
                       help='Expected proportion of anomalies (default: 0.1)')
    parser.add_argument('--train-all', action='store_true',
                       help='Train all models')
    
    args = parser.parse_args()
    
    # Initialize system
    system = AnomalyDetectionSystem(contamination=args.contamination)
    
    # Run pipeline
    success = system.run_full_pipeline()
    
    if success:
        print("\n✨ Anomaly detection models ready for deployment!")
        print("📝 Next step: Run src/explainable_ai.py for SHAP/LIME analysis")
    else:
        print("\n❌ Pipeline failed. Please check errors above.")


if __name__ == "__main__":
    main()
