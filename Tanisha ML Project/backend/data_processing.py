import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def preprocess_data(df):
    # Copy dataframe to avoid modifying original
    processed_df = df.copy()
    
    # Feature Engineering
    # 1. Average of previous scores
    processed_df['Prev_Avg_Score'] = (processed_df['Exam_Score_1'] + 
                                     processed_df['Exam_Score_2'] + 
                                     processed_df['Exam_Score_3']) / 3
    
    # 2. Score trend (improvement or decline)
    processed_df['Score_Trend'] = processed_df['Exam_Score_3'] - processed_df['Exam_Score_1']
    
    # Categorical Encoding
    label_encoders = {}
    categorical_cols = ['Class', 'Subject', 'Participation_Level']
    
    for col in categorical_cols:
        le = LabelEncoder()
        processed_df[col] = le.fit_transform(processed_df[col])
        label_encoders[col] = le
        
    # Save encoders for later use in API
    os.makedirs('models', exist_ok=True)
    joblib.dump(label_encoders, 'models/label_encoders.joblib')
    
    return processed_df, label_encoders

def prepare_features(df):
    # Features for the model
    feature_cols = [
        'Class', 'Subject', 'Exam_Score_1', 'Exam_Score_2', 'Exam_Score_3',
        'Attendance_Percentage', 'Study_Hours_Per_Week', 
        'Assignment_Completion_Rate', 'Participation_Level',
        'Prev_Avg_Score', 'Score_Trend'
    ]
    
    X = df[feature_cols]
    y = df['Final_Score']
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.joblib')
    # Save feature names to ensure consistency
    joblib.dump(feature_cols, 'models/feature_cols.joblib')
    
    return X_scaled, y, scaler

if __name__ == "__main__":
    import pandas as pd
    try:
        df = pd.read_csv('data/student_performance.csv')
        processed_df, _ = preprocess_data(df)
        X, y, _ = prepare_features(processed_df)
        print(f"Data preprocessed. Feature shape: {X.shape}")
    except FileNotFoundError:
        print("Data file not found. Run data_generation.py first.")
