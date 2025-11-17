# 🚀 Quick Start Guide - Enhanced AQI System

## ✨ What's New?

Your AQI Anomaly Detection System now includes **powerful prediction capabilities** with live weather data integration!

### New Features:
1. **🌤️ Live Weather Integration** - Real-time AQI data from OpenWeatherMap
2. **🤖 Multiple ML Models** - 13 models (7 regression, 6 classification)
3. **📊 Clustering Analysis** - K-Means, DBSCAN, Hierarchical clustering
4. **🎯 AQI Prediction** - Predict future AQI values with high accuracy
5. **📈 Interactive Dashboard** - New prediction page with visualizations

---

## 🏃‍♂️ Quick Start (5 Minutes)

### Step 1: Get OpenWeatherMap API Key (Free)

1. Visit https://openweathermap.org/api
2. Sign up for a free account
3. Copy your API key from the dashboard
4. Free tier: 60 calls/minute, 1M calls/month

### Step 2: Configure API Key

```bash
# Copy example file
cp .env.example .env

# Edit .env file and add your key
nano .env
```

Add this line:
```
OPENWEATHER_API_KEY=your_actual_api_key_here
```

### Step 3: Install New Dependencies

```bash
# From Project directory
pip install requests
# OR
pip install -r requirements.txt
```

### Step 4: Train Prediction Models (Optional)

```bash
cd src
python aqi_predictor.py
```

This trains all 13 ML models and saves them to `models/` directory.

**Note:** Pre-trained models are included, but you can retrain with your own data.

### Step 5: Launch Enhanced Dashboard

```bash
cd dashboard
streamlit run streamlit_app.py
```

### Step 6: Use AQI Prediction

1. Click **🌤️ AQI Prediction** in the sidebar
2. Enter your API key (if not in .env)
3. Enter a city name (e.g., "Delhi", "Mumbai")
4. Click **Fetch Live Data**
5. Review live pollutant levels
6. Click **Predict AQI**
7. See predictions, accuracy, and health advisory

---

## 📊 Dashboard Pages

### Original Pages (Enhanced)
1. **🏠 Dashboard** - System overview with anomaly statistics
2. **🔍 Anomaly Explorer** - Browse and filter detected anomalies
3. **🧠 Explainable AI** - SHAP/LIME visualizations
4. **⚠️ Alert Center** - Intelligent alert management
5. **📊 Model Performance** - Anomaly detector comparison

### 🆕 NEW PAGE
6. **🌤️ AQI Prediction** - Live weather + ML predictions

---

## 🎯 Use Cases

### Use Case 1: Check Current Air Quality
```
1. Go to 🌤️ AQI Prediction page
2. Enter your city
3. Fetch live data
4. See real-time AQI and pollutant levels
```

### Use Case 2: Predict Tomorrow's AQI
```
1. Fetch live data for your city
2. System auto-fills yesterday's values
3. Click "Predict AQI"
4. Get ML prediction with accuracy metrics
```

### Use Case 3: Compare Real vs. Predicted AQI
```
1. After fetching live data
2. Make prediction
3. View comparison: Real AQI vs. Predicted AQI
4. Check accuracy percentage and error margin
```

### Use Case 4: Get Health Advisory
```
1. Make AQI prediction
2. System shows health advisory based on AQI category
3. Get recommendations for outdoor activities
```

---

## 🤖 Available ML Models

### Regression Models (AQI Value Prediction)
- Random Forest Regressor ⭐ (Best: R²~0.92)
- Gradient Boosting Regressor
- AdaBoost Regressor
- Decision Tree Regressor
- Linear Regression
- Ridge Regression
- KNN Regressor

### Classification Models (AQI Category Prediction)
- Random Forest Classifier ⭐ (Best: Acc~84%)
- Gradient Boosting Classifier
- AdaBoost Classifier
- Decision Tree Classifier
- Logistic Regression
- KNN Classifier
- Naive Bayes

### Clustering Models (Pattern Analysis)
- K-Means (6 clusters for AQI categories)
- DBSCAN (density-based)
- Hierarchical Clustering

---

## 📝 Python API Usage

### Train Models Programmatically

```python
from src.aqi_predictor import AQIPredictorSystem
import pandas as pd

# Initialize system
predictor = AQIPredictorSystem(models_dir='models')

# Load your data
df = pd.read_csv('data/City_Day.csv')

# Prepare features
X, y_reg, y_cls, df_processed = predictor.prepare_features(df)

# Split data
from sklearn.model_selection import train_test_split
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

print("✅ Training complete!")
print(f"Best Regressor: {predictor.best_regressor[0]}")
print(f"Best Classifier: {predictor.best_classifier[0]}")
```

### Make Predictions

```python
from src.aqi_predictor import AQIPredictorSystem
import pandas as pd

# Load trained models
predictor = AQIPredictorSystem(models_dir='models')
predictor.load_models()

# Prepare feature vector
features = pd.DataFrame([{
    'PM2.5': 45.5,
    'PM10': 85.2,
    'NO': 12.3,
    'NO2': 35.8,
    'NOx': 48.1,
    'NH3': 8.5,
    'CO': 1.2,
    'SO2': 18.5,
    'O3': 65.2,
    'Benzene': 2.1,
    'Toluene': 5.3,
    'Xylene': 3.2,
    'DayOfWeek': 2,  # Tuesday
    'Month': 11,  # November
    'AQI_lag1': 120.5,
    'PM2.5_lag1': 42.3,
    'City_Delhi': 1,
    'City_Mumbai': 0,
    # ... other city columns set to 0
}])

# Make prediction
result = predictor.predict(features)

print(f"Predicted AQI: {result['predicted_aqi']:.2f}")
print(f"Predicted Category: {result['predicted_bucket_name']}")
```

### Fetch Live Weather Data

```python
from src.weather_api import WeatherAPI, get_aqi_category

# Initialize API
api = WeatherAPI(api_key="your_key_here")

# Fetch data for a city
data = api.get_live_aqi_data("Bangalore", country_code="IN")

if data:
    print(f"City: {data['city']}")
    print(f"Current AQI: {data['actual_aqi']:.2f}")
    print(f"PM2.5: {data['PM2.5']:.2f} μg/m³")
    print(f"PM10: {data['PM10']:.2f} μg/m³")
    print(f"NO2: {data['NO2']:.2f} μg/m³")
    
    category, code, color = get_aqi_category(data['actual_aqi'])
    print(f"Category: {category}")
```

---

## 🔧 Troubleshooting

### Issue: "API key not working"
**Solution:**
- Verify key is active at https://home.openweathermap.org/api_keys
- Check .env file has no quotes: `OPENWEATHER_API_KEY=abc123`
- Wait 10 minutes after creating new API key
- Free tier has 60 calls/minute limit

### Issue: "Models not found"
**Solution:**
```bash
cd Project/src
python aqi_predictor.py
```

### Issue: "Module not found"
**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "City not found"
**Solution:**
- Check spelling of city name
- Try with country code: `api.get_live_aqi_data("Delhi", "IN")`
- Use English name, not local language

---

## 📊 Expected Performance

### Prediction Accuracy:
- **Regression Models:** R² Score ~0.92 (92% variance explained)
- **Classification Models:** Accuracy ~84%
- **Real-time Predictions:** ±15-20 AQI units error

### API Response Time:
- **Weather API:** 1-2 seconds
- **ML Prediction:** < 100ms
- **Total End-to-End:** 2-3 seconds

---

## 🎨 UI Features

### Interactive Visualizations:
- **Pollutant Bar Charts** - Color-coded by concentration
- **Accuracy Gauge** - Model confidence indicator
- **AQI Category Cards** - Color-coded health status
- **Comparison Cards** - Real vs. Predicted AQI
- **Health Advisory Boxes** - Contextual recommendations

### Color Scheme:
- **Good** (0-50): Green
- **Satisfactory** (51-100): Yellow
- **Moderate** (101-200): Orange
- **Poor** (201-300): Red
- **Very Poor** (301-400): Purple
- **Severe** (401+): Maroon

---

## 📚 Additional Resources

### Documentation:
- `FEATURE_ENHANCEMENTS.md` - Detailed feature descriptions
- `README.md` - Main project documentation
- `QUICK_REFERENCE.md` - API reference guide
- `PROJECT_COMPLETION_REPORT.md` - Implementation details

### Code Examples:
- `src/weather_api.py` - API integration examples
- `src/aqi_predictor.py` - Model training examples
- `dashboard/pages/aqi_prediction.py` - UI implementation

### External Links:
- OpenWeatherMap API: https://openweathermap.org/api/air-pollution
- Indian AQI Standards: https://app.cpcbccr.com/ccr_docs/FINAL-REPORT_AQI_.pdf
- SHAP Documentation: https://shap.readthedocs.io/

---

## 🎉 You're All Set!

Your enhanced AQI system is now ready to:
- ✅ Detect anomalies in real-time
- ✅ Predict future AQI values
- ✅ Fetch live weather data
- ✅ Generate health advisories
- ✅ Visualize patterns and trends
- ✅ Compare multiple ML models
- ✅ Explain AI decisions with SHAP/LIME

**Launch the dashboard and explore!** 🚀

```bash
cd dashboard
streamlit run streamlit_app.py
```

Then navigate to **🌤️ AQI Prediction** in the sidebar.

---

## 💬 Need Help?

1. Check `FEATURE_ENHANCEMENTS.md` for detailed explanations
2. Review error messages in the dashboard
3. Check API key configuration in `.env`
4. Verify all dependencies are installed
5. Try the example code snippets above

**Happy Predicting!** 🎯
