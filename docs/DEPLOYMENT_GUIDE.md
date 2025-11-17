# 🚀 Deployment Guide: AQI Anomaly Detection System

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation Methods](#installation-methods)
3. [Configuration](#configuration)
4. [Running the System](#running-the-system)
5. [Troubleshooting](#troubleshooting)
6. [Production Deployment](#production-deployment)
7. [API Reference](#api-reference)
8. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Prerequisites

### System Requirements

**Minimum:**
- OS: macOS 10.14+, Ubuntu 18.04+, Windows 10
- RAM: 4 GB
- Storage: 2 GB free space
- CPU: Dual-core processor

**Recommended:**
- OS: macOS 12+, Ubuntu 20.04+, Windows 11
- RAM: 8 GB or more
- Storage: 5 GB free space
- CPU: Quad-core processor
- GPU: Optional (for faster Autoencoder training)

### Software Dependencies

**Required:**
- Python 3.8 or higher
- pip 21.0+
- git (for cloning repository)

**Optional:**
- Conda/Miniconda (for environment management)
- Docker (for containerized deployment)
- CUDA 11.2+ (for GPU acceleration)

---

## Installation Methods

### Method 1: Automated Setup (Recommended)

#### Step 1: Clone or Extract Project
```bash
# If using git
git clone <repository-url>
cd Project

# If using downloaded folder
cd /path/to/Project
```

#### Step 2: Run Setup Script
```bash
# Make script executable
chmod +x setup.sh

# Run automated setup
./setup.sh
```

**What the script does:**
1. ✅ Checks Python version (3.8+)
2. ✅ Creates virtual environment
3. ✅ Installs all dependencies
4. ✅ Runs data preprocessing
5. ✅ Trains all models
6. ✅ Generates explanations
7. ✅ Launches dashboard

**Total time:** 10-15 minutes (depending on system)

---

### Method 2: Manual Installation

#### Step 1: Create Virtual Environment
```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# Using conda
conda create -n aqi-anomaly python=3.8
conda activate aqi-anomaly
```

#### Step 2: Install Dependencies
```bash
# Install all packages
pip install -r requirements.txt

# Verify installation
pip list | grep -E 'scikit-learn|tensorflow|shap|lime|streamlit'
```

#### Step 3: Prepare Dataset
```bash
# Ensure dataset.csv is in data/ folder
ls data/dataset.csv

# If missing, add your AQI dataset as data/dataset.csv
```

#### Step 4: Run Preprocessing
```bash
python src/data_preprocessing.py
```

**Expected output:**
```
✅ Data loaded: 29531 records
✅ Missing values handled
✅ Features engineered: 28 total
✅ Data normalized
✅ Saved to data/processed_data.csv
```

#### Step 5: Train Models
```bash
python src/anomaly_detectors.py
```

**Expected output:**
```
Training Isolation Forest... ✅
Training LOF... ✅
Training Autoencoder... ✅ (may take 5-10 minutes)
Models saved to models/
```

#### Step 6: Generate Explanations
```bash
python src/explainable_ai.py
```

**Expected output:**
```
Computing SHAP values... ✅
Generating LIME explanations... ✅
Visualizations saved to results/
```

#### Step 7: Launch Dashboard
```bash
streamlit run dashboard/streamlit_app.py
```

**Access dashboard at:** http://localhost:8501

---

## Configuration

### 1. Data Configuration

Edit `src/data_preprocessing.py` to customize:

```python
# Line 30-35: Paths
DATA_PATH = 'data/dataset.csv'
OUTPUT_DIR = 'data/'

# Line 150-155: Imputation strategy
def handle_missing_values(self):
    strategy = 'median'  # Options: 'median', 'mean', 'ffill'
```

### 2. Model Hyperparameters

Edit `src/anomaly_detectors.py`:

```python
# Line 40-45: Isolation Forest
IsolationForest(
    contamination=0.1,      # Change to expected anomaly rate
    n_estimators=100,       # More trees = better but slower
    max_samples='auto'
)

# Line 120-125: LOF
LocalOutlierFactor(
    contamination=0.1,
    n_neighbors=20          # Increase for smoother density
)

# Line 220-240: Autoencoder
encoder_dims = [64, 32, 10]  # Adjust architecture
decoder_dims = [32, 64]
epochs = 50                  # More epochs = better fit
batch_size = 256
```

### 3. Explainability Settings

Edit `src/explainable_ai.py`:

```python
# Line 50-55: SHAP background samples
n_background_samples = 100  # More = accurate but slower

# Line 150-155: LIME perturbations
n_samples = 5000           # More = stable explanations
```

### 4. Dashboard Customization

Edit `dashboard/streamlit_app.py`:

```python
# Line 20-30: Severity thresholds
def get_severity_level(score, p90):
    if score < p90 * 2:
        return "🔴 Critical"  # Adjust multipliers
    elif score < p90 * 1.5:
        return "🟠 High"
    # ...

# Line 50-60: Page layout
st.set_page_config(
    page_title="AQI Anomaly Detection",
    page_icon="🌍",
    layout="wide"  # Options: "centered", "wide"
)
```

---

## Running the System

### Development Mode

**Quick Start:**
```bash
# Activate environment
source venv/bin/activate  # or conda activate aqi-anomaly

# Launch dashboard
streamlit run dashboard/streamlit_app.py
```

**Access URLs:**
- Local: http://localhost:8501
- Network: http://192.168.x.x:8501 (for same-network devices)

### Running Individual Components

**1. Only Preprocessing:**
```bash
python src/data_preprocessing.py
```

**2. Only Model Training:**
```bash
python src/anomaly_detectors.py --train-all
```

**3. Only Explainability:**
```bash
python src/explainable_ai.py --model isolation_forest
```

**4. Only Dashboard (no retraining):**
```bash
streamlit run dashboard/streamlit_app.py
```

---

## Troubleshooting

### Issue 1: "Command not found: python3"

**Solution:**
```bash
# Check Python installation
which python3
python3 --version

# If not installed
# macOS
brew install python@3.8

# Ubuntu
sudo apt-get update
sudo apt-get install python3.8

# Windows: Download from python.org
```

---

### Issue 2: "ModuleNotFoundError: No module named 'streamlit'"

**Cause:** Dependencies not installed

**Solution:**
```bash
# Ensure virtual environment is active
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Verify
python -c "import streamlit; print(streamlit.__version__)"
```

---

### Issue 3: "FileNotFoundError: dataset.csv"

**Cause:** Dataset missing or incorrect path

**Solution:**
```bash
# Check if file exists
ls data/dataset.csv

# If missing, copy your dataset
cp /path/to/your/aqi_data.csv data/dataset.csv

# Verify format (must have columns: City, Date, PM2.5, PM10, etc.)
head -5 data/dataset.csv
```

---

### Issue 4: "TensorFlow not available" warning

**Cause:** TensorFlow installation issue (optional dependency)

**Solution:**
```bash
# Install TensorFlow
pip install tensorflow==2.12.0

# For macOS with M1/M2 chip
pip install tensorflow-macos==2.12.0
pip install tensorflow-metal==0.8.0

# For GPU support (Linux)
pip install tensorflow[and-cuda]==2.12.0

# Verify
python -c "import tensorflow as tf; print(tf.__version__)"
```

**Note:** System will work without TensorFlow using only Isolation Forest and LOF

---

### Issue 5: "SHAP values taking too long"

**Cause:** Large dataset or too many background samples

**Solution:**
Edit `src/explainable_ai.py`:
```python
# Line 52: Reduce background samples
n_background_samples = 50  # Default: 100

# Line 180: Sample data before explaining
data_sample = data.sample(n=500)  # Explain only 500 anomalies
```

---

### Issue 6: Dashboard port already in use

**Error:** `OSError: [Errno 48] Address already in use`

**Solution:**
```bash
# Option 1: Use different port
streamlit run dashboard/streamlit_app.py --server.port 8502

# Option 2: Kill existing process
lsof -ti:8501 | xargs kill -9

# Option 3: Find and kill manually
lsof -i :8501
kill -9 <PID>
```

---

### Issue 7: Memory Error during Autoencoder training

**Error:** `MemoryError: Unable to allocate array`

**Solution:**
Edit `src/anomaly_detectors.py`:
```python
# Line 235: Reduce batch size
batch_size = 128  # Default: 256

# Line 230: Simplify architecture
encoder_dims = [32, 16, 5]  # Default: [64, 32, 10]
```

---

### Issue 8: Low model performance

**Symptoms:** Precision < 0.7, many false positives

**Solutions:**

**A. Adjust contamination rate:**
```python
# src/anomaly_detectors.py
contamination = 0.05  # Default: 0.1 (if anomalies are rare)
```

**B. Improve feature engineering:**
```python
# src/data_preprocessing.py
# Add more domain-specific features
data['PM_ratio'] = data['PM2.5'] / data['PM10']
data['pollution_index'] = data[['PM2.5', 'CO', 'NO2']].mean(axis=1)
```

**C. Tune hyperparameters:**
```python
# For Isolation Forest
n_estimators = 200  # More trees
max_features = 0.8  # Use 80% features per tree

# For LOF
n_neighbors = 30  # Larger neighborhood
```

---

## Production Deployment

### Option 1: Docker Deployment

#### Step 1: Create Dockerfile
```dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Step 2: Build Image
```bash
docker build -t aqi-anomaly-detection .
```

#### Step 3: Run Container
```bash
docker run -p 8501:8501 aqi-anomaly-detection
```

---

### Option 2: Cloud Deployment (Streamlit Cloud)

#### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main
```

#### Step 2: Deploy on Streamlit Cloud
1. Visit https://streamlit.io/cloud
2. Sign in with GitHub
3. Click "New app"
4. Select repository: `<your-username>/<repo-name>`
5. Main file path: `dashboard/streamlit_app.py`
6. Click "Deploy"

**Access URL:** `https://<app-name>.streamlit.app`

---

### Option 3: AWS Deployment (EC2)

#### Step 1: Launch EC2 Instance
- AMI: Ubuntu Server 20.04 LTS
- Instance Type: t3.medium (2 vCPU, 4 GB RAM)
- Security Group: Allow inbound TCP 8501

#### Step 2: Connect and Setup
```bash
# SSH into instance
ssh -i key.pem ubuntu@<ec2-public-ip>

# Update system
sudo apt-get update
sudo apt-get install python3-pip

# Clone project
git clone <repo-url>
cd Project

# Install dependencies
pip3 install -r requirements.txt

# Run setup
./setup.sh
```

#### Step 3: Run as Service
```bash
# Create systemd service
sudo nano /etc/systemd/system/aqi-dashboard.service
```

**Service file:**
```ini
[Unit]
Description=AQI Anomaly Detection Dashboard
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Project
ExecStart=/usr/bin/streamlit run dashboard/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl enable aqi-dashboard
sudo systemctl start aqi-dashboard
sudo systemctl status aqi-dashboard
```

**Access:** `http://<ec2-public-ip>:8501`

---

## API Reference

### Command-Line Interface

#### Preprocessing Module
```bash
python src/data_preprocessing.py [OPTIONS]

Options:
  --input PATH     Input CSV path (default: data/dataset.csv)
  --output DIR     Output directory (default: data/)
  --verbose        Enable detailed logging
```

#### Anomaly Detection Module
```bash
python src/anomaly_detectors.py [OPTIONS]

Options:
  --train-all              Train all models
  --model NAME             Train specific model (isolation_forest, lof, autoencoder)
  --contamination FLOAT    Anomaly rate (default: 0.1)
  --evaluate               Run evaluation metrics
```

#### Explainability Module
```bash
python src/explainable_ai.py [OPTIONS]

Options:
  --model NAME             Model to explain (default: isolation_forest)
  --n-samples INT          Number of anomalies to explain (default: 100)
  --background-size INT    SHAP background samples (default: 100)
```

---

### Python API

#### Load Trained Models
```python
import joblib

# Load specific model
iso_forest = joblib.load('models/isolation_forest_model.pkl')

# Predict on new data
predictions = iso_forest.predict(new_data)
```

#### Detect Anomalies
```python
from src.anomaly_detectors import AnomalyDetectionSystem

system = AnomalyDetectionSystem()
system.load_models()

# Detect anomalies
anomalies = system.detect_anomalies(data)
```

#### Generate Explanations
```python
from src.explainable_ai import ExplainableAnomalyDetection

explainer = ExplainableAnomalyDetection()
explainer.load_model('isolation_forest')

# Explain specific instance
explanation = explainer.explain_instance(instance_data)
print(explanation['top_features'])
```

---

## Monitoring & Maintenance

### Performance Monitoring

**Track key metrics:**
```bash
# Model performance
python src/anomaly_detectors.py --evaluate

# Output: Precision, Recall, F1-Score
```

**Monitor dashboard usage:**
```bash
# Check Streamlit logs
tail -f ~/.streamlit/streamlit.log

# Monitor resource usage
htop  # or top
```

### Model Retraining

**When to retrain:**
- Quarterly or bi-annually (to capture seasonal patterns)
- After significant data drift
- When performance degrades (F1 < 0.75)

**Retraining process:**
```bash
# Backup old models
cp -r models/ models_backup_$(date +%Y%m%d)/

# Add new data to dataset.csv

# Retrain pipeline
./setup.sh  # Or run steps individually
```

### Data Updates

**Append new data:**
```bash
# Option 1: Concatenate CSVs
cat new_data.csv >> data/dataset.csv

# Option 2: Use Python
python <<EOF
import pandas as pd
old = pd.read_csv('data/dataset.csv')
new = pd.read_csv('new_data.csv')
combined = pd.concat([old, new]).drop_duplicates()
combined.to_csv('data/dataset.csv', index=False)
EOF

# Rerun preprocessing
python src/data_preprocessing.py
```

### Backup & Recovery

**Backup critical files:**
```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/$DATE"
mkdir -p $BACKUP_DIR

# Backup models and data
cp -r models/ $BACKUP_DIR/
cp -r data/ $BACKUP_DIR/
cp -r results/ $BACKUP_DIR/

echo "Backup created at $BACKUP_DIR"
EOF

chmod +x backup.sh
./backup.sh
```

**Restore from backup:**
```bash
# List backups
ls backups/

# Restore specific backup
cp -r backups/20241117_143022/models/ models/
```

---

## Security Best Practices

### 1. Environment Variables
```bash
# Store sensitive configs in .env
cat > .env << EOF
DATABASE_URL=postgresql://user:pass@host:5432/db
API_KEY=your-api-key
SECRET_KEY=your-secret-key
EOF

# Load in code
from dotenv import load_dotenv
load_dotenv()
```

### 2. Dependency Updates
```bash
# Check for vulnerabilities
pip install safety
safety check

# Update packages
pip list --outdated
pip install --upgrade <package>
```

### 3. Access Control
```bash
# Add authentication to Streamlit (requires streamlit-authenticator)
pip install streamlit-authenticator

# See: https://github.com/mkhorasani/Streamlit-Authenticator
```

---

## Performance Optimization

### 1. Data Caching
```python
# In dashboard/streamlit_app.py
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    return pd.read_csv('data/processed_data.csv')
```

### 2. Model Serving
```bash
# Use TensorFlow Serving for Autoencoder
docker run -p 8500:8500 \
  --mount type=bind,source=/path/to/models,target=/models \
  tensorflow/serving --model_base_path=/models/autoencoder
```

### 3. Database Integration
```python
# Replace CSV with PostgreSQL for large datasets
import psycopg2

conn = psycopg2.connect(DATABASE_URL)
df = pd.read_sql("SELECT * FROM aqi_data", conn)
```

---

## Support & Resources

### Documentation
- **README.md**: Project overview
- **TECHNICAL_REPORT.md**: Detailed methodology
- **This file**: Deployment guide

### Community
- GitHub Issues: Report bugs
- Discussions: Ask questions
- Pull Requests: Contribute improvements

### Contact
- Email: <your-email>
- GitHub: <your-github>

---

**Last Updated:** November 17, 2025  
**Version:** 1.0  
**Maintained by:** TY Sem 5 AIML Team
