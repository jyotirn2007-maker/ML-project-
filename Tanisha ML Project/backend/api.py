from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import os
from pydantic import BaseModel
from typing import List, Optional
from model_training import (
    load_data, preprocess_data, train_models,
    evaluate_models, save_model, predict_performance,
    DATA_PATH as ML_DATA_PATH
)

app = FastAPI(title="Student Performance Prediction API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store data and models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENT_DATA_PATH = os.path.join(BASE_DIR, 'data', 'student_performance.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.joblib')
ENCODERS_PATH = os.path.join(BASE_DIR, 'models', 'encoders.joblib')

class WhatIfRequest(BaseModel):
    student_id: str
    subject: str
    new_study_hours: float
    new_attendance: float

@app.on_event("startup")
async def startup_event():
    # Ensure data and models exist
    if not os.path.exists(STUDENT_DATA_PATH):
        from data_generation import generate_student_data
        generate_student_data()
    
    if not os.path.exists(MODEL_PATH):
        # Run the full training pipeline
        df = load_data(STUDENT_DATA_PATH)
        X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names = preprocess_data(df)
        trained_models = train_models(X_train, y_train)
        best_model, best_model_name = evaluate_models(trained_models, X_test, y_test)
        save_model(best_model, scaler, label_encoders, feature_names)

@app.get("/students")
def get_students(cls: Optional[str] = None, subject: Optional[str] = None):
    try:
        df = pd.read_csv(STUDENT_DATA_PATH)
        if cls:
            df = df[df['Class'] == cls]
        if subject:
            df = df[df['Subject'] == subject]
        
        # Get unique students (aggregated performance)
        students = df.groupby('Student_ID').agg({
            'Name': 'first',
            'Class': 'first',
            'Final_Score': 'mean',
            'Attendance_Percentage': 'mean',
            'Study_Hours_Per_Week': 'mean'
        }).reset_index()
        
        return students.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/students/search")
def search_students(q: str):
    try:
        df = pd.read_csv(STUDENT_DATA_PATH)
        # Filter by name or ID (case-insensitive)
        results = df[
            (df['Name'].str.contains(q, case=False, na=False)) | 
            (df['Student_ID'].str.contains(q, case=False, na=False))
        ]
        
        # Get unique student records (aggregated)
        students = results.groupby('Student_ID').agg({
            'Name': 'first',
            'Class': 'first',
            'Final_Score': 'mean',
            'Attendance_Percentage': 'mean'
        }).reset_index()
        
        return students.head(10).to_dict(orient='records') # Limit to top 10 results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/notifications")
def get_notifications():
    try:
        df = pd.read_csv(STUDENT_DATA_PATH)
        
        # At risk: low attendance (<75) or low avg score (<50)
        at_risk = df.groupby('Student_ID').agg({
            'Name': 'first',
            'Attendance_Percentage': 'mean',
            'Final_Score': 'mean'
        }).reset_index()
        
        at_risk_list = at_risk[(at_risk['Attendance_Percentage'] < 75) | (at_risk['Final_Score'] < 50)]
        
        notifications = []
        for _, row in at_risk_list.iterrows():
            reason = []
            if row['Attendance_Percentage'] < 75: reason.append("low attendance")
            if row['Final_Score'] < 50: reason.append("declining performance")
            
            notifications.append({
                "id": row['Student_ID'],
                "type": "alert",
                "title": "At-Risk Alert",
                "message": f"{row['Name']} ({row['Student_ID']}) requires attention due to {' and '.join(reason)}.",
                "student_id": row['Student_ID'],
                "timestamp": "Today"
            })
            
        return notifications
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/student/{student_id}")
def get_student_details(student_id: str):
    df = pd.read_csv(STUDENT_DATA_PATH)
    student_records = df[df['Student_ID'] == student_id]
    
    if student_records.empty:
        raise HTTPException(status_code=404, detail="Student not found")
    
    return student_records.to_dict(orient='records')

@app.post("/train")
def trigger_train():
    try:
        df = load_data(STUDENT_DATA_PATH)
        X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names = preprocess_data(df)
        trained_models = train_models(X_train, y_train)
        best_model, best_model_name = evaluate_models(trained_models, X_test, y_test)
        save_model(best_model, scaler, label_encoders, feature_names)
        return {"status": "success", "best_model": best_model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/{student_id}")
def predict_student_performance(student_id: str):
    try:
        df = pd.read_csv(STUDENT_DATA_PATH)
        student_data = df[df['Student_ID'] == student_id]
        
        if student_data.empty:
            raise HTTPException(status_code=404, detail="Student not found")
        
        predictions = []
        for _, row in student_data.iterrows():
            input_data = {
                "Class": row['Class'],
                "Subject": row['Subject'],
                "Exam_Score_1": row['Exam_Score_1'],
                "Exam_Score_2": row['Exam_Score_2'],
                "Exam_Score_3": row['Exam_Score_3'],
                "Attendance_Percentage": row['Attendance_Percentage'],
                "Study_Hours_Per_Week": row['Study_Hours_Per_Week'],
                "Assignment_Completion_Rate": row['Assignment_Completion_Rate'],
                "Participation_Level": row['Participation_Level'],
            }
            pred = predict_performance(input_data)
            predictions.append({
                "subject": row['Subject'],
                "actual_score": row['Final_Score'],
                "predicted_score": pred
            })
            
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/what-if")
def what_if_analysis(request: WhatIfRequest):
    try:
        df = pd.read_csv(STUDENT_DATA_PATH)
        student_sub_data = df[(df['Student_ID'] == request.student_id) & (df['Subject'] == request.subject)]
        
        if student_sub_data.empty:
            raise HTTPException(status_code=404, detail="Student/Subject combination not found")
        
        row = student_sub_data.iloc[0]
        
        def predict_with_params(study_h, attendance):
            input_data = {
                "Class": row['Class'],
                "Subject": row['Subject'],
                "Exam_Score_1": row['Exam_Score_1'],
                "Exam_Score_2": row['Exam_Score_2'],
                "Exam_Score_3": row['Exam_Score_3'],
                "Attendance_Percentage": attendance,
                "Study_Hours_Per_Week": study_h,
                "Assignment_Completion_Rate": row['Assignment_Completion_Rate'],
                "Participation_Level": row['Participation_Level'],
            }
            return predict_performance(input_data)

        current_pred = predict_with_params(row['Study_Hours_Per_Week'], row['Attendance_Percentage'])
        simulated_pred = predict_with_params(request.new_study_hours, request.new_attendance)
        
        return {
            "subject": request.subject,
            "current_study_hours": float(row['Study_Hours_Per_Week']),
            "current_attendance": float(row['Attendance_Percentage']),
            "new_study_hours": request.new_study_hours,
            "new_attendance": request.new_attendance,
            "current_predicted_score": current_pred,
            "simulated_predicted_score": simulated_pred,
            "improvement": round(simulated_pred - current_pred, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/global")
def get_global_analytics():
    df = pd.read_csv(STUDENT_DATA_PATH)
    
    # At risk: low attendance (<75) or low avg score (<50)
    at_risk = df.groupby('Student_ID').agg({
        'Name': 'first',
        'Attendance_Percentage': 'mean',
        'Final_Score': 'mean'
    })
    at_risk_list = at_risk[(at_risk['Attendance_Percentage'] < 75) | (at_risk['Final_Score'] < 50)].reset_index()
    
    # Top performers
    top_performers = at_risk.sort_values(by='Final_Score', ascending=False).head(10).reset_index()
    
    # Class-wise performance
    class_perf = df.groupby('Class')['Final_Score'].mean().to_dict()
    
    # Subject-wise performance
    sub_perf = df.groupby('Subject')['Final_Score'].mean().to_dict()
    
    return {
        "at_risk_count": len(at_risk_list),
        "at_risk_students": at_risk_list.to_dict(orient='records'),
        "top_performers": top_performers.to_dict(orient='records'),
        "class_performance": class_perf,
        "subject_performance": sub_perf
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
