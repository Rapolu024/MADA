"""
Patient Intake Processor
Handles patient data collection, validation, and initial processing
"""

import uuid
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

from .models import PatientIntakeForm, ProcessedPatientData, Demographics, ChiefComplaint
from .nlp_processor import MedicalNLPProcessor

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)


class PatientIntakeProcessor:
    """
    Main processor for patient intake forms
    Handles data validation, cleaning, and initial feature extraction
    """
    
    def __init__(self):
        self.nlp_processor = MedicalNLPProcessor()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Medical keywords for initial screening
        self.urgent_keywords = {
            'chest_pain', 'difficulty_breathing', 'severe_pain', 'bleeding',
            'unconscious', 'seizure', 'stroke_symptoms', 'heart_attack',
            'severe_headache', 'high_fever', 'difficulty_swallowing'
        }
        
        self.symptom_categories = {
            'cardiovascular': [
                'chest pain', 'shortness of breath', 'palpitations', 'dizziness',
                'fatigue', 'swelling', 'irregular heartbeat'
            ],
            'respiratory': [
                'cough', 'wheezing', 'difficulty breathing', 'chest tightness',
                'sputum', 'throat pain'
            ],
            'gastrointestinal': [
                'nausea', 'vomiting', 'diarrhea', 'constipation', 'abdominal pain',
                'bloating', 'heartburn', 'loss of appetite'
            ],
            'neurological': [
                'headache', 'dizziness', 'confusion', 'memory loss', 'numbness',
                'tingling', 'weakness', 'seizure'
            ],
            'musculoskeletal': [
                'joint pain', 'muscle pain', 'stiffness', 'swelling', 'weakness',
                'limited mobility'
            ]
        }

    def process_intake_form(self, intake_form: PatientIntakeForm) -> ProcessedPatientData:
        """
        Process a complete patient intake form
        """
        logger.info(f"Processing intake form for patient: {intake_form.demographics.first_name} {intake_form.demographics.last_name}")
        
        # Generate patient ID if not provided
        patient_id = intake_form.demographics.patient_id or str(uuid.uuid4())
        intake_form.demographics.patient_id = patient_id
        
        # Extract and process features
        processed_features = self._extract_features(intake_form)
        
        # Process text data (symptoms, complaints)
        text_embeddings = self._process_text_data(intake_form)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(intake_form)
        
        # Create processed data object
        processed_data = ProcessedPatientData(
            patient_id=patient_id,
            raw_data=intake_form,
            processed_features=processed_features,
            text_embeddings=text_embeddings,
            risk_factors=risk_factors
        )
        
        logger.info(f"Successfully processed intake form for patient: {patient_id}")
        return processed_data
    
    def _extract_features(self, intake_form: PatientIntakeForm) -> Dict[str, Any]:
        """Extract numerical and categorical features from intake form"""
        features = {}
        
        # Demographic features
        demo = intake_form.demographics
        features['age'] = demo.age
        features['gender_male'] = 1 if demo.gender.value == 'male' else 0
        features['gender_female'] = 1 if demo.gender.value == 'female' else 0
        
        # Vital signs features
        if intake_form.vital_signs:
            vitals = intake_form.vital_signs
            features.update({
                'systolic_bp': vitals.systolic_bp or 0,
                'diastolic_bp': vitals.diastolic_bp or 0,
                'heart_rate': vitals.heart_rate or 0,
                'temperature': vitals.temperature or 0,
                'respiratory_rate': vitals.respiratory_rate or 0,
                'oxygen_saturation': vitals.oxygen_saturation or 0,
                'height': vitals.height or 0,
                'weight': vitals.weight or 0,
                'bmi': vitals.bmi or 0
            })
        
        # Medical history features
        if intake_form.medical_history:
            history = intake_form.medical_history
            features.update({
                'smoking_never': 1 if history.smoking_status and history.smoking_status.value == 'never' else 0,
                'smoking_current': 1 if history.smoking_status and history.smoking_status.value == 'current' else 0,
                'smoking_former': 1 if history.smoking_status and history.smoking_status.value == 'former' else 0,
                'alcohol_none': 1 if history.alcohol_consumption and history.alcohol_consumption.value == 'none' else 0,
                'alcohol_moderate': 1 if history.alcohol_consumption and history.alcohol_consumption.value in ['occasional', 'moderate'] else 0,
                'alcohol_heavy': 1 if history.alcohol_consumption and history.alcohol_consumption.value == 'heavy' else 0,
                'exercise_frequency_score': self._encode_exercise_frequency(history.exercise_frequency),
                'num_allergies': len(history.allergies) if history.allergies else 0,
                'num_medications': len(history.current_medications) if history.current_medications else 0,
                'num_chronic_conditions': len(history.chronic_conditions) if history.chronic_conditions else 0
            })
        
        # Complaint severity
        if intake_form.chief_complaint:
            complaint = intake_form.chief_complaint
            features.update({
                'pain_scale': complaint.pain_scale or 0,
                'num_symptoms': len(complaint.symptoms) if complaint.symptoms else 0
            })
            
            # Add symptom severity scores
            if complaint.symptoms:
                avg_severity = sum(s.severity or 0 for s in complaint.symptoms) / len(complaint.symptoms)
                max_severity = max(s.severity or 0 for s in complaint.symptoms)
                features.update({
                    'avg_symptom_severity': avg_severity,
                    'max_symptom_severity': max_severity
                })
        
        # Lab results features
        if intake_form.lab_results:
            features['num_lab_results'] = len(intake_form.lab_results)
            features['num_abnormal_labs'] = sum(1 for lab in intake_form.lab_results if lab.abnormal)
        
        return features
    
    def _process_text_data(self, intake_form: PatientIntakeForm) -> Dict[str, List[float]]:
        """Process and embed text data from the intake form"""
        text_embeddings = {}
        
        if intake_form.chief_complaint:
            complaint = intake_form.chief_complaint
            
            # Process primary complaint
            if complaint.primary_complaint:
                cleaned_text = self._clean_text(complaint.primary_complaint)
                embedding = self.nlp_processor.get_text_embedding(cleaned_text)
                text_embeddings['primary_complaint'] = embedding
            
            # Process review of systems
            if complaint.review_of_systems:
                cleaned_text = self._clean_text(complaint.review_of_systems)
                embedding = self.nlp_processor.get_text_embedding(cleaned_text)
                text_embeddings['review_of_systems'] = embedding
            
            # Process individual symptoms
            if complaint.symptoms:
                symptom_texts = [s.description for s in complaint.symptoms]
                combined_symptoms = ' '.join(symptom_texts)
                cleaned_text = self._clean_text(combined_symptoms)
                embedding = self.nlp_processor.get_text_embedding(cleaned_text)
                text_embeddings['symptoms'] = embedding
        
        return text_embeddings
    
    def _identify_risk_factors(self, intake_form: PatientIntakeForm) -> List[str]:
        """Identify potential risk factors from patient data"""
        risk_factors = []
        
        # Age-based risk factors
        age = intake_form.demographics.age
        if age >= 65:
            risk_factors.append('elderly')
        elif age >= 45:
            risk_factors.append('middle_aged')
        
        # Vital signs risk factors
        if intake_form.vital_signs:
            vitals = intake_form.vital_signs
            
            if vitals.systolic_bp and vitals.systolic_bp >= 140:
                risk_factors.append('hypertension')
            if vitals.diastolic_bp and vitals.diastolic_bp >= 90:
                risk_factors.append('hypertension')
            if vitals.heart_rate and vitals.heart_rate > 100:
                risk_factors.append('tachycardia')
            if vitals.temperature and vitals.temperature > 100.4:
                risk_factors.append('fever')
            if vitals.bmi and vitals.bmi >= 30:
                risk_factors.append('obesity')
        
        # Medical history risk factors
        if intake_form.medical_history:
            history = intake_form.medical_history
            
            if history.smoking_status and history.smoking_status.value in ['current', 'former']:
                risk_factors.append('smoking_history')
            if history.alcohol_consumption and history.alcohol_consumption.value == 'heavy':
                risk_factors.append('alcohol_abuse')
            if history.chronic_conditions:
                risk_factors.extend([f'chronic_{condition.lower().replace(" ", "_")}' for condition in history.chronic_conditions])
        
        # Symptom-based risk factors
        if intake_form.chief_complaint:
            complaint = intake_form.chief_complaint
            
            # Check for urgent symptoms
            complaint_text = complaint.primary_complaint.lower()
            for urgent_keyword in self.urgent_keywords:
                if urgent_keyword.replace('_', ' ') in complaint_text:
                    risk_factors.append(f'urgent_{urgent_keyword}')
            
            # High pain scale
            if complaint.pain_scale and complaint.pain_scale >= 8:
                risk_factors.append('severe_pain')
        
        return list(set(risk_factors))  # Remove duplicates
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text data"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def _encode_exercise_frequency(self, frequency) -> int:
        """Encode exercise frequency to numerical value"""
        if not frequency:
            return 0
        
        frequency_map = {
            'none': 0,
            'rare': 1,
            'weekly': 2,
            'daily': 3
        }
        return frequency_map.get(frequency.value, 0)
    
    def get_urgency_score(self, intake_form: PatientIntakeForm) -> float:
        """Calculate urgency score for triaging patients"""
        score = 0.0
        
        # High vital signs contribute to urgency
        if intake_form.vital_signs:
            vitals = intake_form.vital_signs
            if vitals.systolic_bp and vitals.systolic_bp > 180:
                score += 0.3
            if vitals.heart_rate and vitals.heart_rate > 120:
                score += 0.2
            if vitals.temperature and vitals.temperature > 103:
                score += 0.2
        
        # Pain scale contributes to urgency
        if intake_form.chief_complaint and intake_form.chief_complaint.pain_scale:
            pain_contribution = intake_form.chief_complaint.pain_scale / 10 * 0.2
            score += pain_contribution
        
        # Urgent keywords in complaint
        if intake_form.chief_complaint:
            complaint_text = intake_form.chief_complaint.primary_complaint.lower()
            urgent_keywords_found = sum(1 for keyword in self.urgent_keywords 
                                      if keyword.replace('_', ' ') in complaint_text)
            score += min(urgent_keywords_found * 0.1, 0.3)
        
        return min(score, 1.0)  # Cap at 1.0
    
    def validate_intake_form(self, form_data: dict) -> tuple[bool, List[str]]:
        """Validate intake form data and return validation results"""
        errors = []
        
        try:
            # Try to create the intake form model
            PatientIntakeForm(**form_data)
            return True, []
        except Exception as e:
            errors.append(str(e))
            return False, errors

# Alias for backward compatibility
PatientIntakeAgent = PatientIntakeProcessor

