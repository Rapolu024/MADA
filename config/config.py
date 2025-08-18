"""
MADA (Medical AI Diagnosis Assistant) Configuration
Main configuration file for the entire system
"""

import os
from pathlib import Path
from typing import List, Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "data" / "models"
LOGS_DIR = BASE_DIR / "logs"

# Database Configuration
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "mada_db"),
    "username": os.getenv("DB_USER", "mada_user"),
    "password": os.getenv("DB_PASSWORD", ""),
    "pool_size": 10,
    "max_overflow": 20
}

# ML Model Configuration
MODEL_CONFIG = {
    "xgboost": {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    },
    "lightgbm": {
        "n_estimators": 1000,
        "max_depth": -1,
        "learning_rate": 0.01,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    },
    "random_forest": {
        "n_estimators": 500,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    }
}

# NLP Configuration
NLP_CONFIG = {
    "model_name": "bert-base-uncased",
    "max_length": 512,
    "batch_size": 16,
    "learning_rate": 2e-5
}

# Risk Stratification Thresholds
RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.7,
    "high": 0.9
}

# Feature Engineering
FEATURE_CONFIG = {
    "numerical_features": [
        "age", "weight", "height", "bmi", "systolic_bp", "diastolic_bp",
        "heart_rate", "temperature", "respiratory_rate"
    ],
    "categorical_features": [
        "gender", "smoking_status", "alcohol_consumption", "exercise_frequency",
        "diet_type", "occupation"
    ],
    "text_features": [
        "chief_complaint", "history_of_present_illness", "review_of_systems"
    ]
}

# Disease Categories (expandable)
DISEASE_CATEGORIES = {
    "cardiovascular": [
        "hypertension", "coronary_artery_disease", "heart_failure", 
        "arrhythmia", "myocardial_infarction"
    ],
    "respiratory": [
        "asthma", "copd", "pneumonia", "bronchitis", "pulmonary_embolism"
    ],
    "endocrine": [
        "diabetes_type_1", "diabetes_type_2", "hypothyroidism", 
        "hyperthyroidism", "metabolic_syndrome"
    ],
    "neurological": [
        "migraine", "epilepsy", "stroke", "alzheimer", "parkinson"
    ],
    "gastrointestinal": [
        "gastritis", "peptic_ulcer", "ibd", "liver_disease", "gallstones"
    ]
}

# Continuous Learning Configuration
LEARNING_CONFIG = {
    "retrain_threshold": 100,  # New cases before retraining
    "model_performance_threshold": 0.8,  # Minimum accuracy
    "feedback_weight": 0.1,  # Weight for doctor feedback
    "update_frequency": "daily"  # How often to check for updates
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    "title": "MADA - Medical AI Diagnosis Assistant",
    "theme": "plotly_white",
    "refresh_interval": 30,  # seconds
    "max_patients_display": 100
}

# Security Configuration
SECURITY_CONFIG = {
    "encryption_key": os.getenv("ENCRYPTION_KEY", ""),
    "session_timeout": 3600,  # seconds
    "max_login_attempts": 3,
    "password_min_length": 8
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(LOGS_DIR / "mada.log"),
    "max_bytes": 10485760,  # 10MB
    "backup_count": 5
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "timeout": 120
}

# Create necessary directories
def create_directories():
    """Create necessary directories if they don't exist"""
    for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    create_directories()
    print("MADA configuration loaded successfully!")
