import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

def train_model():
    base_path = r"c:\Users\param\OneDrive\Desktop\Data Hackathon UDAI"
    data_path = os.path.join(base_path, 'aoee_output', 'aoee_unified_dataset.csv')
    output_dir = os.path.join(base_path, 'aoee_output')
    model_path = os.path.join(output_dir, 'aoee_model.pkl')
    
    print("Loading data for training...")
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # Target: Forecast Biometric Update Demand
    # We want to predict 'Biometric_Updates' (Demand)
    target = 'Biometric_Updates'
    
    # Features: Demographics, Service Desert Score, Historic Enrollment, Anomaly Flags
    features = ['Vulnerability_Index', 'Total_Enrollment', 'Service_Desert_Score', 
                'Is_Anomaly', 'age_5_17', 'age_18_greater']
    
    # Handle NaNs
    df_clean = df.dropna(subset=features + [target])
    
    X = df_clean[features]
    y = df_clean[target]
    
    # Train Model: Random Forest Regressor
    print("Training Random Forest Regressor...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Using limited estimators for speed in prototype
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"Model R^2 Score (Random Forest): {score:.4f}")
    
    # Feature Importance
    importances = dict(zip(features, model.feature_importances_))
    print("Feature Importance:")
    for f, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {f}: {imp:.4f}")
    
    # Save Model
    joblib.dump(model, model_path)
    joblib.dump(features, os.path.join(output_dir, 'model_features.pkl'))
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
