"""
Diagnosis Agent
Main orchestrator for AI-powered medical diagnosis
Combines multiple models, NLP analysis, and medical reasoning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
import json

from .diagnosis_models import DiagnosisModelEnsemble
from src.preprocessing.data_preprocessor import MedicalDataPreprocessor
from src.patient_intake.nlp_processor import MedicalNLPProcessor
from config.config import RISK_THRESHOLDS, DISEASE_CATEGORIES

logger = logging.getLogger(__name__)


class DiagnosisPrediction:
    """Structure for diagnosis prediction results"""
    
    def __init__(self, patient_id: str, predictions: List[Dict[str, Any]], 
                 confidence_scores: Dict[str, float], risk_assessment: Dict[str, Any],
                 recommendations: List[str], urgent_flags: List[str]):
        self.patient_id = patient_id
        self.predictions = predictions
        self.confidence_scores = confidence_scores
        self.risk_assessment = risk_assessment
        self.recommendations = recommendations
        self.urgent_flags = urgent_flags
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'patient_id': self.patient_id,
            'predictions': self.predictions,
            'confidence_scores': self.confidence_scores,
            'risk_assessment': self.risk_assessment,
            'recommendations': self.recommendations,
            'urgent_flags': self.urgent_flags,
            'timestamp': self.timestamp.isoformat(),
            'prediction_id': f"{self.patient_id}_{int(self.timestamp.timestamp())}"
        }


class MedicalReasoningEngine:
    """
    Medical reasoning engine for enhanced diagnosis
    Applies clinical rules and domain knowledge
    """
    
    def __init__(self):
        # Clinical decision rules and guidelines
        self.clinical_rules = self._load_clinical_rules()
        self.symptom_disease_mapping = self._build_symptom_disease_mapping()
        self.urgency_indicators = self._load_urgency_indicators()
    
    def _load_clinical_rules(self) -> Dict[str, Any]:
        """Load clinical decision rules"""
        return {
            'diabetes_screening': {
                'age_threshold': 45,
                'bmi_threshold': 25,
                'required_tests': ['glucose', 'hba1c'],
                'family_history_weight': 0.3
            },
            'hypertension_diagnosis': {
                'systolic_threshold': 140,
                'diastolic_threshold': 90,
                'repeated_measurements': True,
                'lifestyle_factors': ['smoking', 'obesity', 'sedentary']
            },
            'cardiovascular_risk': {
                'age_male_threshold': 40,
                'age_female_threshold': 50,
                'cholesterol_threshold': 200,
                'smoking_multiplier': 2.0
            }
        }
    
    def _build_symptom_disease_mapping(self) -> Dict[str, List[str]]:
        """Build mapping of symptoms to potential diseases"""
        return {
            'chest_pain': [
                'myocardial_infarction', 'angina', 'pulmonary_embolism', 
                'pneumonia', 'gastroesophageal_reflux'
            ],
            'shortness_of_breath': [
                'heart_failure', 'asthma', 'copd', 'pulmonary_embolism', 
                'pneumonia', 'anxiety'
            ],
            'abdominal_pain': [
                'appendicitis', 'gallstones', 'peptic_ulcer', 'gastritis',
                'bowel_obstruction', 'pancreatitis'
            ],
            'headache': [
                'migraine', 'tension_headache', 'cluster_headache',
                'sinusitis', 'hypertension', 'meningitis'
            ],
            'fever': [
                'infection', 'pneumonia', 'urinary_tract_infection',
                'influenza', 'sepsis', 'meningitis'
            ]
        }
    
    def _load_urgency_indicators(self) -> Dict[str, float]:
        """Load indicators for urgent conditions"""
        return {
            'chest_pain_with_radiation': 0.9,
            'difficulty_breathing_severe': 0.85,
            'high_fever_with_confusion': 0.9,
            'severe_abdominal_pain': 0.8,
            'loss_of_consciousness': 0.95,
            'severe_bleeding': 0.9,
            'stroke_symptoms': 0.95,
            'severe_allergic_reaction': 0.9
        }
    
    def apply_clinical_reasoning(self, patient_data: Dict[str, Any], 
                               ml_predictions: Dict[str, float]) -> Dict[str, Any]:
        """Apply clinical reasoning to enhance ML predictions"""
        
        reasoning_results = {
            'enhanced_predictions': ml_predictions.copy(),
            'clinical_flags': [],
            'rule_based_scores': {},
            'contraindications': []
        }
        
        # Apply diabetes screening rules
        diabetes_score = self._apply_diabetes_rules(patient_data)
        reasoning_results['rule_based_scores']['diabetes'] = diabetes_score
        
        # Apply cardiovascular risk assessment
        cv_risk = self._apply_cardiovascular_rules(patient_data)
        reasoning_results['rule_based_scores']['cardiovascular_risk'] = cv_risk
        
        # Check for clinical contraindications
        contraindications = self._check_contraindications(patient_data)
        reasoning_results['contraindications'] = contraindications
        
        # Enhance ML predictions with clinical rules
        reasoning_results['enhanced_predictions'] = self._enhance_predictions(
            ml_predictions, reasoning_results['rule_based_scores']
        )
        
        return reasoning_results
    
    def _apply_diabetes_rules(self, patient_data: Dict[str, Any]) -> float:
        """Apply diabetes screening rules"""
        score = 0.0
        
        demographics = patient_data.get('demographics', {})
        medical_history = patient_data.get('medical_history', {})
        vital_signs = patient_data.get('vital_signs', {})
        
        # Age factor
        age = demographics.get('age', 0)
        if age >= 45:
            score += 0.3
        elif age >= 35:
            score += 0.1
        
        # BMI factor
        bmi = vital_signs.get('bmi', 0)
        if bmi >= 30:
            score += 0.3
        elif bmi >= 25:
            score += 0.2
        
        # Family history
        family_history = medical_history.get('family_history', {})
        if 'diabetes' in family_history:
            score += 0.3
        
        # Lifestyle factors
        if medical_history.get('exercise_frequency') == 'none':
            score += 0.1
        
        return min(score, 1.0)
    
    def _apply_cardiovascular_rules(self, patient_data: Dict[str, Any]) -> float:
        """Apply cardiovascular risk assessment rules"""
        score = 0.0
        
        demographics = patient_data.get('demographics', {})
        medical_history = patient_data.get('medical_history', {})
        vital_signs = patient_data.get('vital_signs', {})
        lab_results = patient_data.get('lab_results', [])
        
        age = demographics.get('age', 0)
        gender = demographics.get('gender', '')
        
        # Age and gender risk
        if gender == 'male' and age >= 40:
            score += 0.2
        elif gender == 'female' and age >= 50:
            score += 0.2
        
        # Hypertension
        systolic_bp = vital_signs.get('systolic_bp', 0)
        if systolic_bp >= 140:
            score += 0.3
        
        # Smoking
        if medical_history.get('smoking_status') == 'current':
            score += 0.25
        
        # Cholesterol (simplified)
        for lab in lab_results:
            if lab.get('test_name') == 'cholesterol_total':
                if lab.get('value', 0) > 200:
                    score += 0.2
        
        # Family history
        family_history = medical_history.get('family_history', {})
        if 'cardiovascular' in family_history:
            score += 0.2
        
        return min(score, 1.0)
    
    def _check_contraindications(self, patient_data: Dict[str, Any]) -> List[str]:
        """Check for clinical contraindications"""
        contraindications = []
        
        medical_history = patient_data.get('medical_history', {})
        allergies = medical_history.get('allergies', [])
        medications = medical_history.get('current_medications', [])
        
        # Medication allergies
        if 'penicillin' in [a.lower() for a in allergies]:
            contraindications.append('penicillin_allergy')\n        \n        # Drug interactions (simplified)\n        med_names = [m.get('name', '').lower() for m in medications]\n        if 'warfarin' in med_names and 'aspirin' in med_names:\n            contraindications.append('bleeding_risk')\n        \n        return contraindications\n    \n    def _enhance_predictions(self, ml_predictions: Dict[str, float], \n                           rule_scores: Dict[str, float]) -> Dict[str, float]:\n        \"\"\"Enhance ML predictions with rule-based scores\"\"\"\n        enhanced = ml_predictions.copy()\n        \n        # Combine ML and rule-based scores\n        for condition, rule_score in rule_scores.items():\n            if condition in enhanced:\n                # Weighted combination (70% ML, 30% rules)\n                enhanced[condition] = 0.7 * enhanced[condition] + 0.3 * rule_score\n            else:\n                enhanced[condition] = rule_score\n        \n        return enhanced\n\n\nclass DiagnosisAgent:\n    \"\"\"\n    Main AI Diagnosis Agent\n    Orchestrates the entire diagnosis process using multiple models and clinical reasoning\n    \"\"\"\n    \n    def __init__(self, load_models: bool = True):\n        # Initialize components\n        self.preprocessor = MedicalDataPreprocessor(load_preprocessors=load_models)\n        self.model_ensemble = DiagnosisModelEnsemble()\n        self.nlp_processor = MedicalNLPProcessor()\n        self.reasoning_engine = MedicalReasoningEngine()\n        \n        # Load pre-trained models if available\n        if load_models:\n            self.load_models()\n        \n        # Diagnosis history for learning\n        self.diagnosis_history = []\n        \n        logger.info(\"Diagnosis Agent initialized\")\n    \n    def train(self, patient_data_list: List[Dict[str, Any]], \n              diagnoses_list: List[List[str]], save_models: bool = True) -> Dict[str, Any]:\n        \"\"\"\n        Train the diagnosis agent on labeled patient data\n        \"\"\"\n        logger.info(f\"Training diagnosis agent on {len(patient_data_list)} patients\")\n        \n        # Validate data quality\n        quality_report = self.preprocessor.validate_data_quality(patient_data_list)\n        logger.info(f\"Data quality report: {quality_report['recommendations']}\")\n        \n        # Create training dataset\n        X, y = self.preprocessor.create_training_dataset(patient_data_list, diagnoses_list)\n        \n        # Train the model ensemble\n        self.model_ensemble.fit(X, y)\n        \n        # Save models if requested\n        if save_models:\n            self.save_models()\n        \n        # Return training summary\n        training_summary = {\n            'num_patients': len(patient_data_list),\n            'num_features': X.shape[1],\n            'num_classes': len(y.unique()),\n            'data_quality': quality_report,\n            'model_performance': self.model_ensemble.performance_tracker.get_performance_summary(),\n            'best_model': self.model_ensemble.performance_tracker.get_best_model()\n        }\n        \n        logger.info(\"Training completed successfully\")\n        return training_summary\n    \n    def diagnose(self, patient_data: Dict[str, Any], \n                include_reasoning: bool = True) -> DiagnosisPrediction:\n        \"\"\"\n        Generate AI diagnosis for a patient\n        \"\"\"\n        logger.info(f\"Generating diagnosis for patient: {patient_data.get('patient_id', 'unknown')}\")\n        \n        patient_id = patient_data.get('patient_id', f'patient_{int(datetime.now().timestamp())}')\n        \n        try:\n            # Preprocess patient data\n            processed_features = self.preprocessor.process_single_patient(patient_data)\n            \n            # Get ML predictions\n            predictions = self.model_ensemble.predict(processed_features)\n            probabilities = self.model_ensemble.predict_proba(processed_features)\n            \n            # Convert to probability dictionary\n            class_names = self.model_ensemble.class_names\n            prob_dict = {}\n            if len(probabilities.shape) > 1 and probabilities.shape[1] > 1:\n                for i, class_name in enumerate(class_names):\n                    prob_dict[class_name] = float(probabilities[0][i])\n            else:\n                # Binary classification\n                prob_dict[class_names[1]] = float(probabilities[0])\n                prob_dict[class_names[0]] = 1.0 - float(probabilities[0])\n            \n            # Apply clinical reasoning if requested\n            if include_reasoning:\n                reasoning_results = self.reasoning_engine.apply_clinical_reasoning(\n                    patient_data, prob_dict\n                )\n                final_predictions = reasoning_results['enhanced_predictions']\n                clinical_flags = reasoning_results['clinical_flags']\n            else:\n                final_predictions = prob_dict\n                clinical_flags = []\n            \n            # Rank predictions by probability\n            sorted_predictions = sorted(\n                final_predictions.items(), \n                key=lambda x: x[1], \n                reverse=True\n            )[:5]  # Top 5 predictions\n            \n            # Structure predictions\n            structured_predictions = []\n            for condition, probability in sorted_predictions:\n                structured_predictions.append({\n                    'condition': condition,\n                    'probability': float(probability),\n                    'confidence_level': self._get_confidence_level(probability),\n                    'category': self._get_disease_category(condition)\n                })\n            \n            # Calculate overall confidence scores\n            confidence_scores = self._calculate_confidence_scores(final_predictions)\n            \n            # Perform risk assessment\n            risk_assessment = self._assess_risk(patient_data, final_predictions)\n            \n            # Generate recommendations\n            recommendations = self._generate_recommendations(\n                patient_data, structured_predictions, risk_assessment\n            )\n            \n            # Check for urgent flags\n            urgent_flags = self._check_urgent_flags(patient_data, structured_predictions)\n            \n            # Create diagnosis prediction\n            diagnosis = DiagnosisPrediction(\n                patient_id=patient_id,\n                predictions=structured_predictions,\n                confidence_scores=confidence_scores,\n                risk_assessment=risk_assessment,\n                recommendations=recommendations,\n                urgent_flags=urgent_flags\n            )\n            \n            # Store in history for learning\n            self.diagnosis_history.append(diagnosis)\n            \n            logger.info(f\"Diagnosis completed for patient {patient_id}\")\n            return diagnosis\n            \n        except Exception as e:\n            logger.error(f\"Diagnosis failed for patient {patient_id}: {e}\")\n            # Return empty diagnosis with error\n            return DiagnosisPrediction(\n                patient_id=patient_id,\n                predictions=[],\n                confidence_scores={'overall': 0.0},\n                risk_assessment={'overall_risk': 'unknown', 'risk_factors': []},\n                recommendations=[\"Unable to generate diagnosis. Please consult a healthcare professional.\"],\n                urgent_flags=[\"diagnosis_error\"]\n            )\n    \n    def _get_confidence_level(self, probability: float) -> str:\n        \"\"\"Convert probability to confidence level\"\"\"\n        if probability >= 0.8:\n            return 'high'\n        elif probability >= 0.6:\n            return 'medium'\n        elif probability >= 0.4:\n            return 'low'\n        else:\n            return 'very_low'\n    \n    def _get_disease_category(self, condition: str) -> str:\n        \"\"\"Get disease category for a condition\"\"\"\n        condition_lower = condition.lower()\n        for category, diseases in DISEASE_CATEGORIES.items():\n            if any(disease.lower() in condition_lower for disease in diseases):\n                return category\n        return 'other'\n    \n    def _calculate_confidence_scores(self, predictions: Dict[str, float]) -> Dict[str, float]:\n        \"\"\"Calculate various confidence metrics\"\"\"\n        if not predictions:\n            return {'overall': 0.0}\n        \n        probabilities = list(predictions.values())\n        top_prob = max(probabilities)\n        second_prob = sorted(probabilities, reverse=True)[1] if len(probabilities) > 1 else 0\n        \n        return {\n            'overall': float(top_prob),\n            'top_prediction': float(top_prob),\n            'certainty': float(top_prob - second_prob),  # Difference between top 2\n            'entropy': float(-sum(p * np.log(p + 1e-10) for p in probabilities if p > 0))\n        }\n    \n    def _assess_risk(self, patient_data: Dict[str, Any], \n                    predictions: Dict[str, float]) -> Dict[str, Any]:\n        \"\"\"Assess overall patient risk\"\"\"\n        risk_factors = []\n        risk_score = 0.0\n        \n        # Get top prediction probability\n        top_prob = max(predictions.values()) if predictions else 0\n        \n        # Age-based risk\n        demographics = patient_data.get('demographics', {})\n        age = demographics.get('age', 0)\n        if age >= 65:\n            risk_factors.append('elderly')\n            risk_score += 0.2\n        \n        # Vital signs risk\n        vital_signs = patient_data.get('vital_signs', {})\n        if vital_signs.get('systolic_bp', 0) > 160:\n            risk_factors.append('severe_hypertension')\n            risk_score += 0.3\n        \n        # High certainty predictions add to risk\n        if top_prob > 0.8:\n            risk_score += 0.2\n        \n        # Determine overall risk level\n        if risk_score >= RISK_THRESHOLDS['high']:\n            overall_risk = 'high'\n        elif risk_score >= RISK_THRESHOLDS['medium']:\n            overall_risk = 'medium'\n        else:\n            overall_risk = 'low'\n        \n        return {\n            'overall_risk': overall_risk,\n            'risk_score': float(risk_score),\n            'risk_factors': risk_factors,\n            'requires_immediate_attention': overall_risk == 'high'\n        }\n    \n    def _generate_recommendations(self, patient_data: Dict[str, Any],\n                                predictions: List[Dict[str, Any]],\n                                risk_assessment: Dict[str, Any]) -> List[str]:\n        \"\"\"Generate clinical recommendations based on diagnosis\"\"\"\n        recommendations = []\n        \n        if not predictions:\n            recommendations.append(\"Insufficient data for diagnosis. Please provide more information.\")\n            return recommendations\n        \n        top_prediction = predictions[0]\n        condition = top_prediction['condition']\n        probability = top_prediction['probability']\n        confidence = top_prediction['confidence_level']\n        \n        # High confidence recommendations\n        if confidence == 'high':\n            recommendations.append(f\"Strong indication of {condition}. Recommend immediate specialist consultation.\")\n        elif confidence == 'medium':\n            recommendations.append(f\"Moderate probability of {condition}. Consider further testing and specialist referral.\")\n        else:\n            recommendations.append(\"Inconclusive results. Recommend comprehensive examination and additional testing.\")\n        \n        # Risk-based recommendations\n        if risk_assessment['overall_risk'] == 'high':\n            recommendations.append(\"High-risk patient. Priority scheduling and close monitoring recommended.\")\n        \n        # Condition-specific recommendations\n        category = top_prediction['category']\n        if category == 'cardiovascular':\n            recommendations.append(\"Consider ECG, cardiac enzymes, and cardiology consultation.\")\n        elif category == 'respiratory':\n            recommendations.append(\"Consider chest X-ray, pulmonary function tests, and respiratory evaluation.\")\n        elif category == 'endocrine':\n            recommendations.append(\"Consider comprehensive metabolic panel and endocrinology consultation.\")\n        \n        # Lifestyle recommendations\n        medical_history = patient_data.get('medical_history', {})\n        if medical_history.get('smoking_status') == 'current':\n            recommendations.append(\"Strongly recommend smoking cessation counseling and support.\")\n        \n        return recommendations\n    \n    def _check_urgent_flags(self, patient_data: Dict[str, Any],\n                          predictions: List[Dict[str, Any]]) -> List[str]:\n        \"\"\"Check for conditions requiring urgent attention\"\"\"\n        urgent_flags = []\n        \n        # High probability of serious conditions\n        for prediction in predictions[:3]:  # Check top 3\n            if prediction['probability'] > 0.7:\n                condition = prediction['condition'].lower()\n                if any(urgent in condition for urgent in \n                      ['infarction', 'stroke', 'embolism', 'sepsis', 'failure']):\n                    urgent_flags.append(f\"possible_{prediction['condition']}\")\n        \n        # Vital signs urgency\n        vital_signs = patient_data.get('vital_signs', {})\n        if vital_signs.get('systolic_bp', 0) > 180:\n            urgent_flags.append('hypertensive_crisis')\n        if vital_signs.get('heart_rate', 0) > 130:\n            urgent_flags.append('severe_tachycardia')\n        if vital_signs.get('temperature', 0) > 103:\n            urgent_flags.append('high_fever')\n        \n        # Symptom-based urgency\n        chief_complaint = patient_data.get('chief_complaint', {})\n        if chief_complaint:\n            complaint_text = chief_complaint.get('primary_complaint', '').lower()\n            if any(urgent_symptom in complaint_text for urgent_symptom in \n                  ['chest pain', 'difficulty breathing', 'severe pain', 'bleeding']):\n                urgent_flags.append('urgent_symptoms')\n        \n        return urgent_flags\n    \n    def update_with_feedback(self, patient_id: str, correct_diagnosis: str, \n                           doctor_feedback: str = None):\n        \"\"\"Update models with doctor feedback (online learning)\"\"\"\n        logger.info(f\"Updating models with feedback for patient {patient_id}\")\n        \n        # Find the original diagnosis\n        original_diagnosis = None\n        for diagnosis in self.diagnosis_history:\n            if diagnosis.patient_id == patient_id:\n                original_diagnosis = diagnosis\n                break\n        \n        if not original_diagnosis:\n            logger.warning(f\"Original diagnosis not found for patient {patient_id}\")\n            return\n        \n        # Store feedback for analysis\n        feedback_record = {\n            'patient_id': patient_id,\n            'original_prediction': original_diagnosis.predictions[0]['condition'] if original_diagnosis.predictions else None,\n            'correct_diagnosis': correct_diagnosis,\n            'doctor_feedback': doctor_feedback,\n            'timestamp': datetime.now().isoformat()\n        }\n        \n        # This is where we would implement reinforcement learning\n        # For now, just log the feedback\n        logger.info(f\"Feedback recorded: {feedback_record}\")\n    \n    def save_models(self, filepath: str = None) -> bool:\n        \"\"\"Save all trained models and preprocessors\"\"\"\n        try:\n            # Save model ensemble\n            model_saved = self.model_ensemble.save_models(filepath)\n            \n            # Save preprocessor\n            preprocessor_saved = self.preprocessor.save_preprocessors()\n            \n            return model_saved and preprocessor_saved\n            \n        except Exception as e:\n            logger.error(f\"Failed to save models: {e}\")\n            return False\n    \n    def load_models(self, filepath: str = None) -> bool:\n        \"\"\"Load trained models and preprocessors\"\"\"\n        try:\n            # Load model ensemble\n            model_loaded = self.model_ensemble.load_models(filepath)\n            \n            # Load preprocessor\n            preprocessor_loaded = self.preprocessor.load_preprocessors()\n            \n            return model_loaded and preprocessor_loaded\n            \n        except Exception as e:\n            logger.error(f\"Failed to load models: {e}\")\n            return False\n    \n    def get_agent_summary(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive summary of the diagnosis agent\"\"\"\n        return {\n            'model_ensemble': self.model_ensemble.get_model_summary(),\n            'preprocessor': self.preprocessor.get_preprocessing_summary(),\n            'diagnosis_history_count': len(self.diagnosis_history),\n            'last_diagnosis': self.diagnosis_history[-1].to_dict() if self.diagnosis_history else None,\n            'agent_status': 'ready' if self.model_ensemble.is_fitted else 'untrained'\n        }
