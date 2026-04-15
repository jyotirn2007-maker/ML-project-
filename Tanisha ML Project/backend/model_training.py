# =============================================================================
# Student Performance Prediction System - ML Pipeline
# File: model_training.py
# Description: Complete machine learning pipeline for predicting student
#              academic performance using Linear Regression, Random Forest,
#              and Gradient Boosting models.
# =============================================================================

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =============================================================================
# Configuration / Paths
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_performance.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
ENCODERS_SAVE_PATH = os.path.join(MODEL_DIR, "encoders.joblib")

# Target column
TARGET = "Final_Score"

# Categorical columns to encode
CATEGORICAL_COLS = ["Class", "Subject", "Participation_Level"]

# Columns to drop (identifiers, not features)
DROP_COLS = ["Student_ID", "Name"]

# Exam score columns used for feature engineering
SCORE_COLS = ["Exam_Score_1", "Exam_Score_2", "Exam_Score_3"]


# =============================================================================
# 1. load_data() — Load dataset from CSV
# =============================================================================
def load_data(filepath=DATA_PATH):
    """
    Loads the student performance dataset from the given CSV filepath.
    Prints a preview of the first 5 rows and the dataset shape.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)

    print("=" * 60)
    print(" STEP 1: DATA LOADING")
    print("=" * 60)
    print(df.head())
    print(f"\nDataset Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data Types:\n{df.dtypes}")
    print()

    return df


# =============================================================================
# 2. preprocess_data() — Handle missing values, encode, engineer features
# =============================================================================
def preprocess_data(df):
    """
    Performs all data preprocessing steps:
      - Drops rows where the target (Final_Score) is missing
      - Fills remaining missing numeric values with column mean
      - Fills remaining missing categorical values with column mode
      - Creates engineered features: average_score, improvement_trend
      - Drops identifier columns (Student_ID, Name)
      - Encodes categorical columns using LabelEncoder
      - Scales all features using StandardScaler

    Returns:
      X_train, X_test, y_train, y_test — split and scaled arrays
      scaler        — fitted StandardScaler instance
      label_encoders — dict of {column_name: fitted LabelEncoder}
      feature_names  — list of feature column names (in order)
    """
    df = df.copy()

    print("=" * 60)
    print(" STEP 2: DATA PREPROCESSING")
    print("=" * 60)

    # --- 2a. Handle missing values ---
    # Drop rows where target is missing
    before = len(df)
    df = df.dropna(subset=[TARGET])
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing target '{TARGET}'")

    # Fill missing numeric values with column mean
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            df[col] = df[col].fillna(df[col].mean())
            print(f"  Filled {missing} missing values in '{col}' with mean")

    # Fill missing categorical values with column mode
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            print(f"  Filled {missing} missing values in '{col}' with mode")

    print(f"  Missing values after cleaning: {df.isnull().sum().sum()}")

    # --- 2b. Feature Engineering ---
    print("\n  Feature Engineering:")

    # average_score = (Exam_Score_1 + Exam_Score_2 + Exam_Score_3) / 3
    if all(col in df.columns for col in SCORE_COLS):
        df["average_score"] = df[SCORE_COLS].mean(axis=1)
        print("    Created 'average_score' = mean of 3 exam scores")

    # improvement_trend = Exam_Score_3 - Exam_Score_1
    if "Exam_Score_3" in df.columns and "Exam_Score_1" in df.columns:
        df["improvement_trend"] = df["Exam_Score_3"] - df["Exam_Score_1"]
        print("    Created 'improvement_trend' = Exam_Score_3 - Exam_Score_1")

    # --- 2c. Drop identifier columns ---
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    print(f"\n  Dropped identifier columns: {DROP_COLS}")

    # --- 2d. Encode categorical columns using LabelEncoder ---
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"  Encoded '{col}' - {len(le.classes_)} unique classes: {list(le.classes_)}")

    # --- 2e. Separate features (X) and target (y) ---
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    feature_names = X.columns.tolist()

    print(f"\n  Feature columns ({len(feature_names)}): {feature_names}")
    print(f"  Target column: {TARGET}")

    # --- 2f. Scale features using StandardScaler ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("  Applied StandardScaler to all features")

    # --- 2g. Train-Test Split (80-20) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"\n  Train-Test Split (80/20):")
    print(f"    Training samples: {X_train.shape[0]}")
    print(f"    Testing samples:  {X_test.shape[0]}")
    print()

    return X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names


# =============================================================================
# 3. train_models() — Train all three regression models
# =============================================================================
def train_models(X_train, y_train):
    """
    Trains three regression models on the training data:
      - Linear Regression
      - Random Forest Regressor (100 estimators)
      - Gradient Boosting Regressor (100 estimators)

    Returns:
      trained_models — dict of {model_name: fitted model object}
    """
    print("=" * 60)
    print(" STEP 3: MODEL TRAINING")
    print("=" * 60)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
    }

    trained_models = {}
    for name, model in models.items():
        print(f"  Training {name}...", end=" ")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print("Done")

    print()
    return trained_models


# =============================================================================
# 4. evaluate_models() — Evaluate all models and select the best one
# =============================================================================
def evaluate_models(trained_models, X_test, y_test):
    """
    Evaluates each trained model on the test set using:
      - R² Score
      - MAE (Mean Absolute Error)
      - RMSE (Root Mean Squared Error)

    Automatically selects the best model based on highest R² score.

    Returns:
      best_model      — the fitted model object with the highest R²
      best_model_name — name string of the best model
    """
    print("=" * 60)
    print(" STEP 4: MODEL EVALUATION")
    print("=" * 60)

    results = {}

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results[name] = {"R2": r2, "MAE": mae, "RMSE": rmse}

        print(f"\n  {name}:")
        print(f"    R² Score : {r2:.4f}")
        print(f"    MAE      : {mae:.4f}")
        print(f"    RMSE     : {rmse:.4f}")

    # --- Model Selection: pick highest R² ---
    best_model_name = max(results, key=lambda k: results[k]["R2"])
    best_model = trained_models[best_model_name]

    print("\n" + "-" * 60)
    print(f"  * Best Model: {best_model_name}")
    print(f"    R² = {results[best_model_name]['R2']:.4f}")
    print(f"    MAE = {results[best_model_name]['MAE']:.4f}")
    print(f"    RMSE = {results[best_model_name]['RMSE']:.4f}")
    print("-" * 60)
    print()

    return best_model, best_model_name, results


# =============================================================================
# 5. save_model() — Save best model and preprocessing artifacts
# =============================================================================
def save_model(model, scaler, label_encoders, feature_names):
    """
    Saves the trained model and all preprocessing artifacts to disk:
      - best_model.joblib    — the trained model
      - encoders.joblib      — scaler, label encoders, and feature name order
    """
    print("=" * 60)
    print(" STEP 5: SAVING MODEL & ARTIFACTS")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the trained model
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"  Model saved to: {MODEL_SAVE_PATH}")

    # Save preprocessing artifacts (scaler, encoders, feature order)
    artifacts = {
        "scaler": scaler,
        "label_encoders": label_encoders,
        "feature_names": feature_names,
    }
    joblib.dump(artifacts, ENCODERS_SAVE_PATH)
    print(f"  Encoders & Scaler saved to: {ENCODERS_SAVE_PATH}")
    print()


def train_best_model():
    """
    Executes the full training pipeline and saves all model artifacts needed by the API.

    Returns:
      best_model   -- fitted sklearn model instance
      best_name    -- name of the selected best model
      results      -- evaluation metrics for all candidate models
    """
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names = preprocess_data(df)
    trained_models = train_models(X_train, y_train)
    best_model, best_model_name, results = evaluate_models(trained_models, X_test, y_test)

    save_model(best_model, scaler, label_encoders, feature_names)

    # Save separate artifacts for API compatibility
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.joblib"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_cols.joblib"))
    joblib.dump(best_model_name, os.path.join(MODEL_DIR, "best_model_name.joblib"))
    joblib.dump(results, os.path.join(MODEL_DIR, "training_results.joblib"))

    return best_model, best_model_name, results


# =============================================================================
# 6. predict_performance() — Prediction function for FastAPI integration
# =============================================================================
def predict_performance(input_data: dict) -> float:
    """
    Accepts a dictionary of student data and returns the predicted Final_Score.

    Expected input_data keys:
      - Class                    (str)  e.g. "Grade 10"
      - Subject                  (str)  e.g. "Math"
      - Exam_Score_1             (float)
      - Exam_Score_2             (float)
      - Exam_Score_3             (float)
      - Attendance_Percentage    (float)
      - Study_Hours_Per_Week     (float)
      - Assignment_Completion_Rate (float)
      - Participation_Level      (str)  e.g. "High"

    Returns:
      predicted_score (float)
    """
    # Load saved model and artifacts
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_SAVE_PATH}. "
            "Run model_training.py first."
        )
    if not os.path.exists(ENCODERS_SAVE_PATH):
        raise FileNotFoundError(
            f"Encoders not found at {ENCODERS_SAVE_PATH}. "
            "Run model_training.py first."
        )

    model = joblib.load(MODEL_SAVE_PATH)
    artifacts = joblib.load(ENCODERS_SAVE_PATH)

    scaler = artifacts["scaler"]
    label_encoders = artifacts["label_encoders"]
    feature_names = artifacts["feature_names"]

    # Convert input dictionary to a single-row DataFrame
    df = pd.DataFrame([input_data])

    # Apply the same feature engineering as training
    if all(col in df.columns for col in SCORE_COLS):
        df["average_score"] = df[SCORE_COLS].mean(axis=1)
        df["improvement_trend"] = df["Exam_Score_3"] - df["Exam_Score_1"]

    # Drop identifier columns if present
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Apply label encoding (handle unseen labels gracefully)
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col].astype(str))

    # Ensure all expected feature columns exist in the correct order
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0  # Fill missing features with 0

    df = df[feature_names]

    # Apply StandardScaler
    X_scaled = scaler.transform(df)

    # Predict and return
    prediction = model.predict(X_scaled)
    return round(float(prediction[0]), 2)


# =============================================================================
# Main Pipeline Execution
# =============================================================================
if __name__ == "__main__":
    print()
    print("+" + "-" * 58 + "+")
    print("|   Student Performance Prediction - ML Training Pipeline  |")
    print("+" + "-" * 58 + "+")
    print()

    # Step 1: Load data
    df = load_data(DATA_PATH)

    # Step 2: Preprocess data (clean, engineer, encode, scale, split)
    X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names = (
        preprocess_data(df)
    )

    # Step 3: Train all three models
    trained_models = train_models(X_train, y_train)

    # Step 4: Evaluate and select the best model
    best_model, best_model_name, results = evaluate_models(trained_models, X_test, y_test)

    # Step 5: Save the best model and preprocessing artifacts
    save_model(best_model, scaler, label_encoders, feature_names)

    # Step 6: Test the prediction function with sample data
    print("=" * 60)
    print(" STEP 6: TESTING PREDICTION FUNCTION")
    print("=" * 60)

    sample_student = {
        "Class": "Grade 10",
        "Subject": "Math",
        "Exam_Score_1": 70.0,
        "Exam_Score_2": 75.0,
        "Exam_Score_3": 80.0,
        "Attendance_Percentage": 90.0,
        "Study_Hours_Per_Week": 15.0,
        "Assignment_Completion_Rate": 95.0,
        "Participation_Level": "High",
    }

    predicted_score = predict_performance(sample_student)
    print(f"\n  Sample Input: {sample_student}")
    print(f"  Predicted Final Score: {predicted_score}")

    print()
    print("=" * 60)
    print(" [OK] Pipeline complete. Ready for FastAPI integration.")
    print("=" * 60)
    print()
