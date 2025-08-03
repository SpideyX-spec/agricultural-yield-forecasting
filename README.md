Crop Yield Prediction Model
A comprehensive machine learning project that predicts crop yields based on agricultural requirements including pesticide usage, rainfall, temperature, and other environmental factors.

📊 Project Overview
This project develops a predictive model for crop yields using agricultural data from multiple sources. The model achieved excellent performance with an R² score of 0.9747, explaining 97.47% of the variance in crop yield data.

🎯 Key Results
Model Performance Metrics
Mean Absolute Error (MAE): 3,244.89 hg/ha

Mean Squared Error (MSE): 123,556,110.26

Root Mean Squared Error (RMSE): 11,115.58 hg/ha

R-squared (R² Score): 0.9747 (97.47% variance explained)

Cross-Validation Results
CV R² Scores: [0.8848, 0.8847, 0.8917]

Mean CV R² Score: 0.8871

Feature Selection Results
Simplified Model R² Score: 0.8805 (using top 20 features)

Top 10 features contribute: 72.46% of total importance

Features needed for 90% predictive power: 34

📁 Dataset
The project combines multiple agricultural datasets:
Crop Yield Data (yield.csv-https://github.com/user-attachments/files/21566437/yield.csv & [yield_df.csv]- https://github.com/user-attachments/files/21566445/yield_df.csv) - Historical crop yield records

Pesticide Usage (pesticides.csv-https://github.com/user-attachments/files/21566432/pesticides.csv) - Pesticide application data

Rainfall Data (rainfall.csv-https://github.com/user-attachments/files/21566435/rainfall.csv) - Annual rainfall measurements

Temperature Data (temp.csv-https://github.com/user-attachments/files/21566436/temp.csv) - Average temperature records

Data Sources
Countries: 212 unique countries

Crops: 10 crop types (Maize, Potatoes, Rice, Wheat, Sweet potatoes, etc.)

Time Period: Multi-year agricultural data

Total Records: 109,366 data points after merging

🔬 Methodology
1. Data Preprocessing
Data Cleaning: Handled missing values using mean imputation

Feature Engineering: One-hot encoding for categorical variables

Data Merging: Combined datasets on country and year keys

Normalization: MinMaxScaler applied to scale features to 0-1 range

2. Model Development
Algorithm: Random Forest Regressor

Features: 224 total features after preprocessing

Train-Test Split: 80-20 split with random state 42

Scaling: Both features and target variables normalized

3. Model Validation
Cross-Validation: 3-fold CV for robustness testing

Feature Importance Analysis: Identified most predictive factors

Simplified Model: Created lightweight version with top 20 features

📈 Feature Importance Analysis
Top 10 Most Important Features for Crop Yield Prediction:
Crop_Potatoes (25.59%) - Potato crop type

Crop_Sweet potatoes (8.78%) - Sweet potato crop type

Year (8.42%) - Temporal trends

Crop_Rice, paddy (7.25%) - Rice crop type

Average rainfall (4.91%) - Annual precipitation

Average temperature (4.40%) - Climate conditions

Pesticides usage (4.22%) - Agricultural inputs

Crop_Maize (3.97%) - Maize crop type

Country_India (2.77%) - Geographic factor

Crop_Wheat (2.15%) - Wheat crop type

🛠️ Technical Implementation
Dependencies
python
pandas
scikit-learn
numpy
matplotlib
joblib
Model Pipeline
Data Loading: Import and merge multiple CSV files

Preprocessing: Clean, encode, and scale data

Training: RandomForestRegressor with 100 estimators

Evaluation: Multiple metrics and cross-validation

Feature Analysis: Importance ranking and selection

Model Persistence: Save trained models and scalers

Key Code Components
Data Integration: Merges 4 separate agricultural datasets

Scaling Pipeline: Separate scalers for features and targets

Cross-Validation: Robust performance assessment

Feature Selection: Automated importance-based reduction

📋 Usage
Training the Model
python
# Load and preprocess data
python crop_yield_prediction.py

# The script will:
# 1. Merge all datasets
# 2. Preprocess and scale features
# 3. Train the Random Forest model
# 4. Evaluate performance
# 5. Save the trained model
Making Predictions
python
import joblib

# Load saved components
model = joblib.load('crop_yield_model.pkl')
feature_scaler = joblib.load('feature_scaler.pkl')
target_scaler = joblib.load('target_scaler.pkl')

# Scale new data and predict
scaled_features = feature_scaler.transform(new_data)
scaled_prediction = model.predict(scaled_features)
final_prediction = target_scaler.inverse_transform(scaled_prediction)
📊 Model Interpretability
Agricultural Insights
Crop Type: Major factor determining yield potential

Climate Variables: Rainfall and temperature significantly impact yields

Geographic Factors: Country-specific conditions matter

Temporal Trends: Year indicates technological/methodological improvements

Input Usage: Pesticide application correlates with yield outcomes

Business Applications
Yield Forecasting: Predict crop outputs for planning

Resource Optimization: Optimize pesticide and water usage

Risk Assessment: Identify climate-related yield risks

Policy Planning: Support agricultural decision-making

🔄 Model Versions
Full Model
Features: 224 (all available features)

R² Score: 0.9747

Use Case: Maximum accuracy applications

Simplified Model
Features: 20 (top important features)

R² Score: 0.8805

Use Case: Faster predictions, reduced complexity

📁 File Structure
text
crop-yield-prediction/
├── data/
│   ├── yield.csv
│   ├── pesticides.csv
│   ├── rainfall.csv
│   ├── temp.csv
│   └── final_merged_data.csv
├── models/
│   ├── crop_yield_model.pkl
│   ├── feature_scaler.pkl
│   ├── target_scaler.pkl
│   └── feature_names.pkl
├── notebooks/
│   └── crop_yield_prediction_data.ipynb
├── src/
│   └── crop_yield_prediction.py
└── README.md
🚀 Future Improvements
Advanced Models: Experiment with Gradient Boosting, Neural Networks

Feature Engineering: Create interaction terms, seasonal features

Ensemble Methods: Combine multiple algorithms

Real-time Data: Integrate weather APIs for live predictions

Spatial Analysis: Include geographical coordinate features
