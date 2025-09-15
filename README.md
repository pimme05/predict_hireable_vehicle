# 🚗 Car Hire Prediction System

A machine learning project that predicts vehicle hire status using CatBoost classification with geospatial and temporal features.

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Project Goals](#project-goals)
- [Features](#features)
- [Model Overview](#model-overview)
- [Data Pipeline](#data-pipeline)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [License](#license)

## 🎯 Problem Statement

This project addresses the critical need to predict vehicle availability for hire in real-time taxi/ride-sharing operations. By accurately predicting when vehicles will have their "for hire" lights on, the system enables:

- **Demand-Supply Optimization**: Better matching of available vehicles with passenger demand
- **Resource Allocation**: Strategic positioning of vehicles in high-demand areas
- **Revenue Maximization**: Reducing idle time and increasing operational efficiency
- **Customer Experience**: Faster response times and reduced wait times for passengers

## 🎪 Project Goals

- **Primary Objective**: Achieve >75% accuracy in predicting vehicle hire light status using geospatial and temporal features
- **Business Impact**: Enable real-time fleet optimization and demand forecasting
- **Technical Goals**: 
  - Process large-scale GPS tracking data (5.4M+ records)
  - Implement robust geospatial feature engineering with reverse geocoding
  - Deploy production-ready CatBoost model with GPU acceleration
  - Validate model performance on blind test data from different time periods

## ✨ Features

This project implements a comprehensive car hire prediction system with:

- **Temporal Analysis**: Hour-based patterns and weekend detection
- **Geospatial Intelligence**: GPS coordinates and location-based features
- **Vehicle Dynamics**: Speed, acceleration, and stationary status tracking
- **Location Context**: Named locations and geographical feature classification
- **Interactive Visualization**: Dynamic maps showing vehicle trajectories and hire status

## 🤖 Model Overview

### Architecture
- **Algorithm**: CatBoostClassifier
- **Type**: Binary Classification
- **Target**: `for_hire_light` status prediction

### Feature Engineering Details
```python
# Temporal Features
df["hour"] = df["timestamp"].dt.hour
df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
df["part_of_day"] = df["hour"].apply(get_part_of_day)  # morning/afternoon/evening/night

# Vehicle Dynamics
df["is_stationary"] = df["speed"] == 0
df["stop_duration"] = df.groupby("VehicleID")["is_stationary"].apply(lambda x: x.cumsum() * x)

# Speed Analytics
speed_analysis = df.groupby("VehicleID")["speed"].agg({
    "average_speed": "mean",
    "max_speed": "max", 
    "speed_variance": "var"
})

# Geospatial Context (GeoNames Database Integration)
def get_location_details(lat, lon):
    # Reverse geocoding with caching for performance
    return location_name, feature_class, feature_code, population

# Interaction Features
df["speed_engine_interaction"] = df["speed"] * df["engine_acc"]
df["hour_weekend_interaction"] = df["hour"] * df["is_weekend"]

# Lag Features
df["lag_speed"] = df.groupby("VehicleID")["speed"].shift(1)
df["lag_stop_duration"] = df.groupby("VehicleID")["stop_duration"].shift(1)
```

### Model Training & Configuration
```python
model = CatBoostClassifier(
    iterations=15000,           # Maximum boosting rounds
    learning_rate=0.025,        # Conservative learning rate
    depth=7,                    # Tree depth
    loss_function='CrossEntropy', # Binary classification loss
    eval_metric='Accuracy',     # Optimization metric
    random_seed=42,            # Reproducibility
    verbose=100,               # Progress logging
    task_type="GPU",           # GPU acceleration
    early_stopping_rounds=100  # Overfitting prevention
)

# Training with categorical feature support
model.fit(
    X_train, y_train, 
    eval_set=(X_test, y_test),
    cat_features=["feature_class", "feature_code", "part_of_day", "location_name"],
    early_stopping_rounds=100
)
```

## 🔄 Data Pipeline

### 1. **Data Loading & Initial Processing**
```python
# Load multiple CSV files from March 2024 (3 days of data)
file_list = ["20240301.csv.out", "20240302.csv.out", "20240303.csv.out"]
headers = ["VehicleID", "gpsvalid", "lat", "lon", "timestamp", "speed", "heading", "for_hire_light", "engine_acc"]
```

### 2. **Advanced Feature Engineering**
- **Temporal Features**: Hour extraction, weekend detection, part of day classification
- **Geospatial Features**: Reverse geocoding using GeoNames database for location context
- **Vehicle Dynamics**: Speed analysis, stationary detection, stop duration calculation
- **Lag Features**: Previous speed and stop duration for temporal patterns
- **Interaction Features**: Speed-engine and hour-weekend interactions

### 3. **Robust Data Processing**
- **Outlier Removal**: IQR-based speed filtering
- **Missing Value Handling**: Strategic imputation for GPS and sensor data
- **Chunked Processing**: 5000-record chunks for memory efficiency with threading
- **Data Validation**: Remove invalid GPS coordinates and null feature codes

### 4. **Model Training Pipeline**
- **CatBoost Implementation**: GPU-accelerated training with categorical feature support
- **Hyperparameter Configuration**: 15k iterations, 0.025 learning rate, depth=7
- **Early Stopping**: Prevent overfitting with 100-iteration patience
- **Cross-entropy Loss**: Optimized for binary classification

### 5. **Comprehensive Evaluation**
- **Train/Test Split**: 90/10 split with stratification
- **Blind Test Validation**: Separate dataset from different time period (March 31, 2024)
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Model Persistence**: Joblib serialization for production deployment

## 📊 Evaluation

### Metrics
- **Accuracy Score**: Overall prediction accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity measure
- **F1-Score**: Harmonic mean of precision and recall

### Implementation
```python
from sklearn.metrics import classification_report, accuracy_score

# Generate predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
```

## 🌍 Visualization

Interactive mapping capabilities using **Folium**:

- **Vehicle Trajectories**: Track movement patterns over time
- **Hire Status Overlay**: Visual representation of availability
- **Geographic Context**: Location-based insights and patterns
- **Real-time Updates**: Dynamic visualization of current status

## 📈 Results

### **Model Performance Metrics**

#### Training Dataset Performance (540K samples)
- **Overall Accuracy**: **78.5%**
- **Precision (Class 0 - Not For Hire)**: 82% 
- **Precision (Class 1 - For Hire)**: 71%
- **Recall (Class 0)**: 86%
- **Recall (Class 1)**: 64%
- **F1-Score**: 0.78 (weighted average)

#### Blind Test Validation (50K samples from March 31, 2024)
- **Blind Test Accuracy**: **71.9%**
- **Model Generalization**: Strong performance on unseen temporal data
- **Robustness**: <7% accuracy drop demonstrates good temporal stability

### **Business Impact Insights**
- Successfully identifies **82%** of vehicles that are NOT for hire (reducing false dispatches)
- Captures **64%** of available vehicles (enabling efficient demand matching)
- Real-time processing capability with **chunked inference** pipeline
- **Geospatial hotspot detection** for strategic vehicle positioning

### **Key Performance Indicators**
```
Training Metrics:
├── Accuracy: 0.785
├── Macro F1-Score: 0.76
├── Weighted F1-Score: 0.78
└── Training Time: ~44 minutes (GPU)

Blind Test Validation:
├── Accuracy: 0.719  
├── Precision: 0.71
├── Recall: 0.72
└── Generalization Gap: 6.6%
```

## 🚀 Installation

### Clone Repository
```bash
git clone <repo_url>
cd car-hire-prediction
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Setup GeoNames Database (First-time only)
```bash
# Download Thailand GeoNames data
wget https://download.geonames.org/export/dump/TH.zip
unzip TH.zip

# Setup database (run the first cell in the notebook)
# This creates geonames.db for reverse geocoding
```

### Verify GPU Support (Optional)
```python
import catboost
print(f"CatBoost GPU support: {catboost.cuda().have_cuda}")
# If True, GPU acceleration will be used automatically
```

## 💻 Usage

### **Initial Setup & Data Preprocessing**
```bash
# First-time setup: Create GeoNames database (run once)
python -c "from setup import setup_geonames_database; setup_geonames_database('TH.txt', 'geonames.db')"
```

### **Training Pipeline**
```bash
# Execute the complete training pipeline
jupyter notebook hire-light-predict.ipynb

# Or run specific sections:
# 1. Data loading and preprocessing
# 2. Feature engineering with geospatial enrichment  
# 3. CatBoost model training
# 4. Model evaluation and persistence
```

### **Model Inference**
```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load("catboost_model_20k-0.025-logloss.pkl")

# Prepare features for prediction
X_features = df[["hour", "lat", "lon", "is_weekend", "engine_acc", 
                 "gpsvalid", "speed", "is_stationary", "stop_duration", 
                 "feature_class", "feature_code", "part_of_day", "location_name"]]

# Generate predictions
predictions = model.predict(X_features)
probabilities = model.predict_proba(X_features)
```

### **Visualization & Analysis**
```python
# Generate heatmap visualization
import folium
from folium.plugins import HeatMap

# Create hotspot visualization
heatmap_data = df[df["predicted_for_hire"] == 1][["lat", "lon"]].values.tolist()
m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=12)
HeatMap(heatmap_data).add_to(m)
m.save("hire_hotspots.html")
```

## 📌 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM recommended
- **Storage**: 1GB available space

### Python Libraries
```txt
# Core Data Processing
pandas>=1.3.0
numpy>=1.21.0
joblib>=1.0.0

# Machine Learning
catboost>=1.0.0
scikit-learn>=1.0.0

# Geospatial Processing
geopy>=2.2.0
sqlite3  # Built-in Python module

# Visualization
folium>=0.12.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Development Environment
jupyter>=1.0.0
tqdm>=4.62.0  # Progress bars
concurrent.futures  # Built-in threading

# Optional: GPU Acceleration
# CUDA-compatible GPU required for CatBoost GPU training
```

### Installation Command
```bash
pip install -r requirements.txt
```

## 🗂️ Project Structure

```
car-hire-prediction/
├── hire-light-predict.ipynb      # Main analysis and training notebook
├── requirements.txt              # Python dependencies
├── data/                        # Data directory
│   ├── files/                   # Training data (March 1-3, 2024)
│   │   ├── 20240301.csv.out
│   │   ├── 20240302.csv.out
│   │   └── 20240303.csv.out
│   └── blind_eval/              # Blind test data
│       └── 20240331.csv.out
├── models/                      # Saved model files
│   └── catboost_model_20k-0.025-logloss.pkl
├── processed_data/              # Processed datasets
│   ├── processed_df_training.csv
│   └── processed_df_blind_test.csv
├── geonames_data/              # Geographic reference data
│   ├── TH.txt                  # Thailand GeoNames data
│   └── geonames.db            # SQLite database
├── visualizations/             # Generated maps and plots
│   └── hire_hotspots.html
└── README.md                   # This file
```

## 📜 License

This project utilizes open datasets from:
- **Longdo Traffic**: Traffic and vehicle data
- **GeoNames**: Geographic location data

Please review their respective terms of use and licensing agreements before using this project in commercial applications.

**Note**: This project is for educational and research purposes. Ensure compliance with all data usage policies and regulations in your jurisdiction.
