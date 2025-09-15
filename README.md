# 🚖 Car Hire Prediction using Probe Data

## 📌 Overview
This project predicts whether a vehicle is **hireable or not**, based on **GPS probe data** and **enriched geographical features**.  
The prediction is modeled on the `for_hire_light` variable, which indicates the hire status of a car.

**Workflow:**
- Raw probe data ingestion (vehicle telemetry and GPS signals)  
- Feature engineering with geographical datasets  
- Data cleaning & transformation (outlier removal, missing value handling)  
- Lag features for speed and stop duration  
- Efficient chunk-based processing to handle large datasets  
- Model training using **CatBoostClassifier**  
- Evaluation & visualization with metrics and interactive maps  

---

## 📂 Data Sources

### Probe Vehicle Data (Raw Data) – *Longdo Traffic Open Data*
Contains:
- `VehicleID`, `gpsvalid`, `lat`, `lon`, `timestamp`  
- `speed`, `heading`, `for_hire_light`, `engine_acc`  

### Geographical Enrichment Data – *GeoNames Open Dataset*
Used for:
- `feature_class`, `feature_code`, `location_name`  

Stored locally in **SQLite** for efficient querying (avoiding API overhead).  

---

## ⚙️ Feature Engineering
- **Datetime features**: Hour of day, weekend flag, part of day  
- **Geographical features**: `feature_class`, `feature_code`, `location_name`  
- **Speed transformation**: Convert speed to mph  
- **Missing value handling**: Fill NA values with appropriate strategies  
- **Outlier removal**: Interquartile Range (IQR) method  
- **Lag features**: Speed history and stop duration  
- **Stationary detection**: Flag vehicles with no movement  

---

## 🏗️ Model Pipeline

### Data Preparation
- Process raw data in **chunks** for scalability  
- Join with enriched features from **SQLite**  

### Train-Test Split
```python
from sklearn.model_selection import train_test_split
