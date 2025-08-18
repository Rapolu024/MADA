#!/usr/bin/env python3
"""
MADA System Demonstration
Shows how all components work together for medical diagnosis
"""

import sys
import os
import logging
from datetime import datetime, date
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import MADA components
from src.patient_intake.models import PatientIntakeForm, Demographics, VitalSigns, MedicalHistory, ChiefComplaint, Symptom
from src.patient_intake.intake_processor import PatientIntakeProcessor
from src.preprocessing.data_preprocessor import MedicalDataPreprocessor
from src.diagnosis_agent.diagnosis_agent import DiagnosisAgent
from config.config import create_directories

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_patient_data():
    """Create sample patient data for demonstration"""
    
    # Patient 1: Potential Diabetes Case
    patient1 = {
        "demographics": {
            "patient_id": "demo_001",
            "first_name": "John",
            "last_name": "Smith", 
            "age": 52,
            "gender": "male",
            "email": "john.smith@email.com"
        },
        "vital_signs": {
            "systolic_bp": 145,
            "diastolic_bp": 92,
            "heart_rate": 78,
            "temperature": 98.6,
            "respiratory_rate": 16,
            "oxygen_saturation": 98,
            "height": 175,  # cm
            "weight": 95     # kg
        },
        "medical_history": {
            "smoking_status": "former",
            "alcohol_consumption": "moderate",
            "exercise_frequency": "rare",
            "allergies": ["penicillin"],
            "current_medications": [
                {"name": "metformin", "dosage": "500mg", "frequency": "twice daily"}
            ],
            "chronic_conditions": ["prediabetes"],
            "family_history": {
                "diabetes": ["father", "grandmother"],
                "cardiovascular": ["father"]
            }
        },
        "chief_complaint": {
            "primary_complaint": "Increased thirst and frequent urination for the past 3 weeks",
            "pain_scale": 2,
            "symptoms": [
                {
                    "description": "excessive thirst",
                    "severity": 6,
                    "duration": "3 weeks",
                    "frequency": "constant"
                },
                {
                    "description": "frequent urination",
                    "severity": 7,
                    "duration": "3 weeks", 
                    "frequency": "every 1-2 hours"
                },
                {
                    "description": "fatigue",
                    "severity": 5,
                    "duration": "2 weeks",
                    "frequency": "daily"
                }
            ]
        },
        "lab_results": [
            {
                "test_name": "glucose",
                "value": 145,
                "unit": "mg/dL",
                "reference_range": "70-100 mg/dL",
                "abnormal": True,
                "test_date": datetime.now()
            },
            {
                "test_name": "hemoglobin",
                "value": 14.2,
                "unit": "g/dL", 
                "reference_range": "12-16 g/dL",
                "abnormal": False,
                "test_date": datetime.now()
            }
        ]
    }
    
    # Patient 2: Potential Cardiovascular Case
    patient2 = {
        "demographics": {
            "patient_id": "demo_002",
            "first_name": "Sarah",
            "last_name": "Johnson",
            "age": 48,
            "gender": "female",
            "email": "sarah.johnson@email.com"
        },
        "vital_signs": {
            "systolic_bp": 165,
            "diastolic_bp": 98,
            "heart_rate": 95,
            "temperature": 98.4,
            "respiratory_rate": 18,
            "oxygen_saturation": 96,
            "height": 168,
            "weight": 78
        },
        "medical_history": {
            "smoking_status": "current",
            "alcohol_consumption": "occasional",
            "exercise_frequency": "none",
            "allergies": [],
            "current_medications": [],
            "chronic_conditions": [],
            "family_history": {
                "cardiovascular": ["mother", "uncle"],
                "hypertension": ["mother", "father"]
            }
        },
        "chief_complaint": {
            "primary_complaint": "Chest tightness and shortness of breath during minimal exertion",
            "pain_scale": 6,
            "symptoms": [
                {
                    "description": "chest tightness",
                    "severity": 7,
                    "duration": "1 week",
                    "frequency": "during exertion",
                    "triggers": ["walking upstairs", "carrying groceries"],
                    "relief_factors": ["rest"]
                },
                {
                    "description": "shortness of breath",
                    "severity": 6,
                    "duration": "1 week",
                    "frequency": "during activity"
                }
            ]
        },
        "lab_results": [
            {
                "test_name": "cholesterol_total",
                "value": 245,
                "unit": "mg/dL",
                "reference_range": "0-200 mg/dL", 
                "abnormal": True,
                "test_date": datetime.now()
            }
        ]
    }
    
    return [patient1, patient2]

def demonstrate_patient_intake():
    """Demonstrate patient intake processing"""
    logger.info("=== PATIENT INTAKE DEMONSTRATION ===")
    
    # Initialize intake processor
    intake_processor = PatientIntakeProcessor()
    
    # Create sample patients
    sample_patients = create_sample_patient_data()
    
    processed_patients = []
    
    for i, patient_data in enumerate(sample_patients, 1):
        logger.info(f"\nProcessing Patient {i}: {patient_data['demographics']['first_name']} {patient_data['demographics']['last_name']}")
        
        try:
            # Create intake form (this would normally come from web form)
            intake_form = PatientIntakeForm(**patient_data)
            
            # Process the intake form
            processed_data = intake_processor.process_intake_form(intake_form)
            processed_patients.append(processed_data)
            
            # Calculate urgency score
            urgency = intake_processor.get_urgency_score(intake_form)
            
            logger.info(f"✓ Patient processed successfully")
            logger.info(f"  - Patient ID: {processed_data.patient_id}")
            logger.info(f"  - Features extracted: {len(processed_data.processed_features)}")
            logger.info(f"  - Risk factors: {len(processed_data.risk_factors)}")
            logger.info(f"  - Urgency score: {urgency:.3f}")
            logger.info(f"  - Top risk factors: {processed_data.risk_factors[:3]}")
            
        except Exception as e:
            logger.error(f"✗ Failed to process patient {i}: {e}")
    
    return processed_patients

def demonstrate_preprocessing():
    """Demonstrate data preprocessing"""
    logger.info("\n=== DATA PREPROCESSING DEMONSTRATION ===")
    
    # Get sample patient data
    sample_patients = create_sample_patient_data()
    
    # Initialize preprocessor
    preprocessor = MedicalDataPreprocessor()
    
    # Validate data quality
    logger.info("\n--- Data Quality Assessment ---")
    quality_report = preprocessor.validate_data_quality(sample_patients)
    
    logger.info(f"✓ Data quality assessment completed")
    logger.info(f"  - Total patients: {quality_report['total_patients']}")
    
    for field, data in quality_report['data_completeness'].items():
        logger.info(f"  - {field}: {data['percentage']:.1f}% complete")
    
    if quality_report['recommendations']:
        logger.info("  - Recommendations:")
        for rec in quality_report['recommendations'][:3]:
            logger.info(f"    • {rec}")
    
    # Demonstrate feature engineering
    logger.info("\n--- Feature Engineering ---")
    try:
        # Create dummy training data (normally this would be from real diagnoses)
        dummy_diagnoses = [["diabetes_type_2"], ["hypertension"]]
        
        # Create training dataset
        X, y = preprocessor.create_training_dataset(sample_patients, dummy_diagnoses, balance_classes=False)
        
        logger.info(f"✓ Training dataset created")
        logger.info(f"  - Feature matrix shape: {X.shape}")
        logger.info(f"  - Target classes: {y.unique().tolist()}")
        logger.info(f"  - Sample features: {X.columns[:10].tolist()}")
        
        return X, y, preprocessor
        
    except Exception as e:
        logger.error(f"✗ Preprocessing failed: {e}")
        return None, None, None

def demonstrate_diagnosis():
    """Demonstrate AI diagnosis"""
    logger.info("\n=== AI DIAGNOSIS DEMONSTRATION ===")
    
    # Initialize diagnosis agent
    diagnosis_agent = DiagnosisAgent(load_models=False)
    
    # Get sample data
    sample_patients = create_sample_patient_data()
    X, y, preprocessor = demonstrate_preprocessing()
    
    if X is None or y is None:
        logger.error("Cannot demonstrate diagnosis without preprocessed data")
        return
    
    logger.info("\n--- Training Diagnosis Models ---")
    
    try:
        # Train the diagnosis agent
        training_summary = diagnosis_agent.train(
            sample_patients, 
            [["diabetes_type_2"], ["hypertension"]], 
            save_models=False
        )
        
        logger.info(f"✓ Model training completed")
        logger.info(f"  - Training patients: {training_summary['num_patients']}")
        logger.info(f"  - Features used: {training_summary['num_features']}")
        logger.info(f"  - Disease classes: {training_summary['num_classes']}")
        logger.info(f"  - Best model: {training_summary['best_model']}")
        
        # Get model performance
        performance = training_summary['model_performance']
        if performance:
            logger.info(f"  - Model performance summary:")
            for model_name, perf in performance.items():
                latest_perf = perf['latest_performance']
                logger.info(f"    • {model_name}: Accuracy={latest_perf.get('accuracy', 0):.3f}, F1={latest_perf.get('f1_score', 0):.3f}")
        
    except Exception as e:
        logger.error(f"✗ Model training failed: {e}")
        logger.info("Continuing with untrained models for demonstration...")
    
    logger.info("\n--- Generating Diagnoses ---")
    
    # Generate diagnoses for sample patients
    for i, patient_data in enumerate(sample_patients, 1):
        try:
            logger.info(f"\n--- Diagnosing Patient {i} ---")
            
            # Generate diagnosis
            diagnosis = diagnosis_agent.diagnose(patient_data, include_reasoning=True)
            
            logger.info(f"✓ Diagnosis generated for {diagnosis.patient_id}")
            logger.info(f"  - Timestamp: {diagnosis.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Top predictions
            if diagnosis.predictions:
                logger.info("  - Top predictions:")
                for j, pred in enumerate(diagnosis.predictions[:3], 1):
                    logger.info(f"    {j}. {pred['condition']}: {pred['probability']:.3f} ({pred['confidence_level']})")
            
            # Risk assessment
            risk = diagnosis.risk_assessment
            logger.info(f"  - Risk level: {risk['overall_risk']} (score: {risk['risk_score']:.3f})")
            if risk['risk_factors']:
                logger.info(f"  - Risk factors: {', '.join(risk['risk_factors'][:3])}")
            
            # Recommendations
            if diagnosis.recommendations:
                logger.info(f"  - Recommendations:")
                for rec in diagnosis.recommendations[:2]:
                    logger.info(f"    • {rec}")
            
            # Urgent flags
            if diagnosis.urgent_flags:
                logger.info(f"  - ⚠️  Urgent flags: {', '.join(diagnosis.urgent_flags)}")
            
        except Exception as e:
            logger.error(f"✗ Diagnosis failed for patient {i}: {e}")

def demonstrate_system_summary():
    """Show overall system capabilities"""
    logger.info("\n=== MADA SYSTEM SUMMARY ===")
    
    logger.info("✓ Patient Intake System")
    logger.info("  • Structured data collection and validation")
    logger.info("  • NLP processing of free-text symptoms")
    logger.info("  • Automatic risk factor identification")
    logger.info("  • Urgency scoring for triage")
    
    logger.info("\n✓ Data Storage & Management")
    logger.info("  • HIPAA-compliant database schema")
    logger.info("  • Encryption for sensitive data")
    logger.info("  • Comprehensive audit logging")
    logger.info("  • Patient data versioning")
    
    logger.info("\n✓ Preprocessing & Feature Engineering")
    logger.info("  • Medical-specific feature extraction")
    logger.info("  • Vital signs normalization")
    logger.info("  • Lab results processing")
    logger.info("  • Text embedding generation")
    
    logger.info("\n✓ AI Diagnosis Engine")
    logger.info("  • Multiple ML models (XGBoost, LightGBM, Random Forest, Neural Networks)")
    logger.info("  • Weighted ensemble predictions")
    logger.info("  • Clinical reasoning integration")
    logger.info("  • Confidence scoring")
    
    logger.info("\n✓ Self-Learning Capabilities")
    logger.info("  • Online learning from new cases")
    logger.info("  • Doctor feedback incorporation")
    logger.info("  • Automatic model retraining")
    logger.info("  • Performance monitoring")
    
    logger.info("\n🎯 Key Benefits:")
    logger.info("  • Early disease detection")
    logger.info("  • Reduced misdiagnosis risk")
    logger.info("  • Improved patient outcomes")
    logger.info("  • Scalable across clinics")
    logger.info("  • Continuous improvement")

def main():
    """Run the complete MADA system demonstration"""
    logger.info("🏥 MADA - Medical AI Early Diagnosis Assistant")
    logger.info("=" * 60)
    logger.info("Starting comprehensive system demonstration...")
    
    # Create necessary directories
    create_directories()
    
    try:
        # Demonstrate each component
        processed_patients = demonstrate_patient_intake()
        demonstrate_diagnosis()
        demonstrate_system_summary()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ MADA system demonstration completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Integrate with real medical datasets (MIMIC-IV)")
        logger.info("2. Deploy with Docker containers")
        logger.info("3. Build web dashboard interface")
        logger.info("4. Setup continuous improvement pipeline")
        logger.info("5. Ensure HIPAA compliance certification")
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
