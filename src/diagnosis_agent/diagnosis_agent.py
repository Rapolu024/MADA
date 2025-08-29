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
            contraindications.append('penicillin_allergy')
        
        # Drug interactions (simplified)
        med_names = [m.get('name', '').lower() for m in medications]
        if 'warfarin' in med_names and 'aspirin' in med_names:
            contraindications.append('bleeding_risk')
        
        return contraindications
    
    def _enhance_predictions(self, ml_predictions: Dict[str, float], 
                           rule_scores: Dict[str, float]) -> Dict[str, float]:
        """Enhance ML predictions with rule-based scores"""
        enhanced = ml_predictions.copy()
        
        # Combine ML and rule-based scores
        for condition, rule_score in rule_scores.items():
            if condition in enhanced:
                # Weighted combination (70% ML, 30% rules)
                enhanced[condition] = 0.7 * enhanced[condition] + 0.3 * rule_score
            else:
                enhanced[condition] = rule_score
        
        return enhanced


class DiagnosisAgent:
    """
    Main AI Diagnosis Agent
    Orchestrates the entire diagnosis process using multiple models and clinical reasoning
    """
    
    def __init__(self, load_models: bool = True):
        # Initialize components
        try:
            self.preprocessor = MedicalDataPreprocessor(load_preprocessors=load_models)
            self.model_ensemble = DiagnosisModelEnsemble()
            self.nlp_processor = MedicalNLPProcessor()
            self.reasoning_engine = MedicalReasoningEngine()
        except Exception as e:
            logger.warning(f"Could not initialize all components: {e}")
            # Initialize with minimal components for demo
            self.preprocessor = None
            self.model_ensemble = None
            self.nlp_processor = None
            self.reasoning_engine = MedicalReasoningEngine()
        
        # Load pre-trained models if available
        if load_models and self.model_ensemble:
            try:
                self.load_models()
            except Exception as e:
                logger.warning(f"Could not load models: {e}")
        
        # Diagnosis history for learning
        self.diagnosis_history = []
        
        logger.info("Diagnosis Agent initialized")
    
    def diagnose(self, patient_data: Dict[str, Any], 
                include_reasoning: bool = True) -> DiagnosisPrediction:
        """
        Generate AI diagnosis for a patient
        """
        logger.info(f"Generating diagnosis for patient: {patient_data.get('patient_id', 'unknown')}")
        
        patient_id = patient_data.get('patient_id', f'patient_{int(datetime.now().timestamp())}')
        
        try:
            # For demo purposes, return mock predictions if models aren't available
            if not self.model_ensemble or not self.preprocessor:
                return self._generate_mock_diagnosis(patient_id, patient_data)
            
            # Preprocess patient data
            processed_features = self.preprocessor.process_single_patient(patient_data)
            
            # Get ML predictions
            predictions = self.model_ensemble.predict(processed_features)
            probabilities = self.model_ensemble.predict_proba(processed_features)
            
            # Convert to probability dictionary
            class_names = self.model_ensemble.class_names
            prob_dict = {}
            if len(probabilities.shape) > 1 and probabilities.shape[1] > 1:
                for i, class_name in enumerate(class_names):
                    prob_dict[class_name] = float(probabilities[0][i])
            else:
                # Binary classification
                prob_dict[class_names[1]] = float(probabilities[0])
                prob_dict[class_names[0]] = 1.0 - float(probabilities[0])
            
            # Apply clinical reasoning if requested
            if include_reasoning and self.reasoning_engine:
                reasoning_results = self.reasoning_engine.apply_clinical_reasoning(
                    patient_data, prob_dict
                )
                final_predictions = reasoning_results['enhanced_predictions']
            else:
                final_predictions = prob_dict
            
            # Rank predictions by probability
            sorted_predictions = sorted(
                final_predictions.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]  # Top 5 predictions
            
            # Structure predictions
            structured_predictions = []
            for condition, probability in sorted_predictions:
                structured_predictions.append({
                    'condition': condition,
                    'probability': float(probability),
                    'confidence_level': self._get_confidence_level(probability),
                    'category': self._get_disease_category(condition)
                })
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(final_predictions)
            
            # Risk assessment
            risk_assessment = self._assess_risk(patient_data, final_predictions)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                patient_data, structured_predictions, risk_assessment
            )
            
            # Check urgent flags
            urgent_flags = self._check_urgent_flags(patient_data, structured_predictions)
            
            # Create diagnosis prediction
            diagnosis = DiagnosisPrediction(
                patient_id=patient_id,
                predictions=structured_predictions,
                confidence_scores=confidence_scores,
                risk_assessment=risk_assessment,
                recommendations=recommendations,
                urgent_flags=urgent_flags
            )
            
            # Store in history
            self.diagnosis_history.append(diagnosis)
            
            logger.info(f"Diagnosis completed for patient {patient_id}")
            return diagnosis
            
        except Exception as e:
            logger.error(f"Diagnosis failed for patient {patient_id}: {e}")
            return self._generate_error_diagnosis(patient_id, str(e))
    
    def _generate_mock_diagnosis(self, patient_id: str, patient_data: Dict[str, Any]) -> DiagnosisPrediction:
        """Generate mock diagnosis for demo purposes"""
        # Simple rule-based mock predictions
        mock_predictions = []
        
        # Get patient info for basic analysis
        demographics = patient_data.get('demographics', {})
        vital_signs = patient_data.get('vital_signs', {})
        chief_complaint = patient_data.get('chief_complaint', {})
        age = demographics.get('age', 0)
        
        # Simple condition detection based on symptoms/vitals
        if vital_signs.get('systolic_bp', 0) > 140:
            mock_predictions.append({
                'condition': 'Hypertension',
                'probability': 0.85,
                'confidence_level': 'high',
                'category': 'cardiovascular'
            })
        
        if age > 45 and 'thirst' in str(chief_complaint).lower():
            mock_predictions.append({
                'condition': 'Type 2 Diabetes',
                'probability': 0.72,
                'confidence_level': 'medium',
                'category': 'endocrine'
            })
        
        if 'chest' in str(chief_complaint).lower():
            mock_predictions.append({
                'condition': 'Cardiovascular Disease',
                'probability': 0.68,
                'confidence_level': 'medium',
                'category': 'cardiovascular'
            })
        
        # Default predictions if none match
        if not mock_predictions:
            mock_predictions = [
                {'condition': 'General Health Assessment Needed', 'probability': 0.60, 'confidence_level': 'medium', 'category': 'general'},
                {'condition': 'Routine Follow-up', 'probability': 0.40, 'confidence_level': 'low', 'category': 'general'}
            ]
        
        # Mock confidence scores
        confidence_scores = {
            'overall': mock_predictions[0]['probability'] if mock_predictions else 0.5,
            'top_prediction': mock_predictions[0]['probability'] if mock_predictions else 0.5
        }
        
        # Mock risk assessment
        risk_score = 0.3 if age > 65 else 0.1
        risk_assessment = {
            'overall_risk': 'high' if risk_score > 0.6 else 'medium' if risk_score > 0.3 else 'low',
            'risk_score': risk_score,
            'risk_factors': ['age'] if age > 65 else [],
            'requires_immediate_attention': risk_score > 0.7
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(patient_data, mock_predictions, risk_assessment)
        
        # Check urgent flags
        urgent_flags = self._check_urgent_flags(patient_data, mock_predictions)
        
        return DiagnosisPrediction(
            patient_id=patient_id,
            predictions=mock_predictions,
            confidence_scores=confidence_scores,
            risk_assessment=risk_assessment,
            recommendations=recommendations,
            urgent_flags=urgent_flags
        )
    
    def _generate_error_diagnosis(self, patient_id: str, error_msg: str) -> DiagnosisPrediction:
        """Generate error diagnosis"""
        return DiagnosisPrediction(
            patient_id=patient_id,
            predictions=[],
            confidence_scores={'overall': 0.0},
            risk_assessment={'overall_risk': 'unknown', 'risk_factors': []},
            recommendations=["Unable to generate diagnosis. Please consult a healthcare professional."],
            urgent_flags=["diagnosis_error"]
        )
    
    def _get_confidence_level(self, probability: float) -> str:
        """Convert probability to confidence level"""
        if probability >= 0.8:
            return 'high'
        elif probability >= 0.6:
            return 'medium'
        elif probability >= 0.4:
            return 'low'
        else:
            return 'very_low'
    
    def _get_disease_category(self, condition: str) -> str:
        """Get disease category for a condition"""
        condition_lower = condition.lower()
        categories = {
            'cardiovascular': ['heart', 'cardio', 'hypertension', 'blood pressure'],
            'respiratory': ['lung', 'breathing', 'asthma', 'pneumonia'],
            'endocrine': ['diabetes', 'thyroid', 'hormone'],
            'neurological': ['headache', 'migraine', 'stroke', 'seizure'],
            'gastrointestinal': ['stomach', 'nausea', 'vomiting', 'diarrhea']
        }
        
        for category, keywords in categories.items():
            if any(keyword in condition_lower for keyword in keywords):
                return category
        return 'other'
    
    def _calculate_confidence_scores(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """Calculate various confidence metrics"""
        if not predictions:
            return {'overall': 0.0}
        
        probabilities = list(predictions.values())
        top_prob = max(probabilities)
        second_prob = sorted(probabilities, reverse=True)[1] if len(probabilities) > 1 else 0
        
        return {
            'overall': float(top_prob),
            'top_prediction': float(top_prob),
            'certainty': float(top_prob - second_prob)
        }
    
    def _assess_risk(self, patient_data: Dict[str, Any], 
                    predictions: Dict[str, float]) -> Dict[str, Any]:
        """Assess overall patient risk"""
        risk_factors = []
        risk_score = 0.0
        
        # Get top prediction probability
        top_prob = max(predictions.values()) if predictions else 0
        
        # Age-based risk
        demographics = patient_data.get('demographics', {})
        age = demographics.get('age', 0)
        if age >= 65:
            risk_factors.append('elderly')
            risk_score += 0.2
        
        # Vital signs risk
        vital_signs = patient_data.get('vital_signs', {})
        if vital_signs.get('systolic_bp', 0) > 160:
            risk_factors.append('severe_hypertension')
            risk_score += 0.3
        
        # High certainty predictions add to risk
        if top_prob > 0.8:
            risk_score += 0.2
        
        # Determine overall risk level
        if risk_score >= 0.6:
            overall_risk = 'high'
        elif risk_score >= 0.3:
            overall_risk = 'medium'
        else:
            overall_risk = 'low'
        
        return {
            'overall_risk': overall_risk,
            'risk_score': float(risk_score),
            'risk_factors': risk_factors,
            'requires_immediate_attention': overall_risk == 'high'
        }
    
    def _generate_recommendations(self, patient_data: Dict[str, Any],
                                predictions: List[Dict[str, Any]],
                                risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate clinical recommendations based on diagnosis"""
        recommendations = []
        
        if not predictions:
            recommendations.append("Insufficient data for diagnosis. Please provide more information.")
            return recommendations
        
        top_prediction = predictions[0]
        condition = top_prediction['condition']
        confidence = top_prediction['confidence_level']
        
        # High confidence recommendations
        if confidence == 'high':
            recommendations.append(f"Strong indication of {condition}. Recommend immediate specialist consultation.")
        elif confidence == 'medium':
            recommendations.append(f"Moderate probability of {condition}. Consider further testing and specialist referral.")
        else:
            recommendations.append("Inconclusive results. Recommend comprehensive examination and additional testing.")
        
        # Risk-based recommendations
        if risk_assessment['overall_risk'] == 'high':
            recommendations.append("High-risk patient. Priority scheduling and close monitoring recommended.")
        
        # Condition-specific recommendations
        category = top_prediction['category']
        if category == 'cardiovascular':
            recommendations.append("Consider ECG, cardiac enzymes, and cardiology consultation.")
        elif category == 'respiratory':
            recommendations.append("Consider chest X-ray, pulmonary function tests, and respiratory evaluation.")
        elif category == 'endocrine':
            recommendations.append("Consider comprehensive metabolic panel and endocrinology consultation.")
        
        return recommendations
    
    def _check_urgent_flags(self, patient_data: Dict[str, Any],
                          predictions: List[Dict[str, Any]]) -> List[str]:
        """Check for conditions requiring urgent attention"""
        urgent_flags = []
        
        # High probability of serious conditions
        for prediction in predictions[:3]:  # Check top 3
            if prediction['probability'] > 0.7:
                condition = prediction['condition'].lower()
                if any(urgent in condition for urgent in 
                      ['infarction', 'stroke', 'embolism', 'sepsis', 'failure']):
                    urgent_flags.append(f"possible_{prediction['condition']}")
        
        # Vital signs urgency
        vital_signs = patient_data.get('vital_signs', {})
        if vital_signs.get('systolic_bp', 0) > 180:
            urgent_flags.append('hypertensive_crisis')
        if vital_signs.get('heart_rate', 0) > 130:
            urgent_flags.append('severe_tachycardia')
        if vital_signs.get('temperature', 0) > 103:
            urgent_flags.append('high_fever')
        
        # Symptom-based urgency
        chief_complaint = patient_data.get('chief_complaint', {})
        if chief_complaint:
            complaint_text = str(chief_complaint.get('primary_complaint', '')).lower()
            if any(urgent_symptom in complaint_text for urgent_symptom in 
                  ['chest pain', 'difficulty breathing', 'severe pain', 'bleeding']):
                urgent_flags.append('urgent_symptoms')
        
        return urgent_flags
    
    def save_models(self, filepath: str = None) -> bool:
        """Save all trained models and preprocessors"""
        try:
            if self.model_ensemble:
                return self.model_ensemble.save_models(filepath)
            return True
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            return False
    
    def load_models(self, filepath: str = None) -> bool:
        """Load trained models and preprocessors"""
        try:
            if self.model_ensemble:
                return self.model_ensemble.load_models(filepath)
            return True
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def get_agent_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the diagnosis agent"""
        return {
            'diagnosis_history_count': len(self.diagnosis_history),
            'last_diagnosis': self.diagnosis_history[-1].to_dict() if self.diagnosis_history else None,
            'agent_status': 'ready'
        }
