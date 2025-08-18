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
        logger.info(f\"Processing intake form for patient: {intake_form.demographics.first_name} {intake_form.demographics.last_name}\")\n        \n        # Generate patient ID if not provided\n        patient_id = intake_form.demographics.patient_id or str(uuid.uuid4())\n        intake_form.demographics.patient_id = patient_id\n        \n        # Extract and process features\n        processed_features = self._extract_features(intake_form)\n        \n        # Process text data (symptoms, complaints)\n        text_embeddings = self._process_text_data(intake_form)\n        \n        # Identify risk factors\n        risk_factors = self._identify_risk_factors(intake_form)\n        \n        # Create processed data object\n        processed_data = ProcessedPatientData(\n            patient_id=patient_id,\n            raw_data=intake_form,\n            processed_features=processed_features,\n            text_embeddings=text_embeddings,\n            risk_factors=risk_factors\n        )\n        \n        logger.info(f\"Successfully processed intake form for patient: {patient_id}\")\n        return processed_data\n    \n    def _extract_features(self, intake_form: PatientIntakeForm) -> Dict[str, Any]:\n        \"\"\"Extract numerical and categorical features from intake form\"\"\"\n        features = {}\n        \n        # Demographic features\n        demo = intake_form.demographics\n        features['age'] = demo.age\n        features['gender_male'] = 1 if demo.gender.value == 'male' else 0\n        features['gender_female'] = 1 if demo.gender.value == 'female' else 0\n        \n        # Vital signs features\n        if intake_form.vital_signs:\n            vitals = intake_form.vital_signs\n            features.update({\n                'systolic_bp': vitals.systolic_bp or 0,\n                'diastolic_bp': vitals.diastolic_bp or 0,\n                'heart_rate': vitals.heart_rate or 0,\n                'temperature': vitals.temperature or 0,\n                'respiratory_rate': vitals.respiratory_rate or 0,\n                'oxygen_saturation': vitals.oxygen_saturation or 0,\n                'height': vitals.height or 0,\n                'weight': vitals.weight or 0,\n                'bmi': vitals.bmi or 0\n            })\n        \n        # Medical history features\n        if intake_form.medical_history:\n            history = intake_form.medical_history\n            features.update({\n                'smoking_never': 1 if history.smoking_status and history.smoking_status.value == 'never' else 0,\n                'smoking_current': 1 if history.smoking_status and history.smoking_status.value == 'current' else 0,\n                'smoking_former': 1 if history.smoking_status and history.smoking_status.value == 'former' else 0,\n                'alcohol_none': 1 if history.alcohol_consumption and history.alcohol_consumption.value == 'none' else 0,\n                'alcohol_moderate': 1 if history.alcohol_consumption and history.alcohol_consumption.value in ['occasional', 'moderate'] else 0,\n                'alcohol_heavy': 1 if history.alcohol_consumption and history.alcohol_consumption.value == 'heavy' else 0,\n                'exercise_frequency_score': self._encode_exercise_frequency(history.exercise_frequency),\n                'num_allergies': len(history.allergies) if history.allergies else 0,\n                'num_medications': len(history.current_medications) if history.current_medications else 0,\n                'num_chronic_conditions': len(history.chronic_conditions) if history.chronic_conditions else 0\n            })\n        \n        # Complaint severity\n        if intake_form.chief_complaint:\n            complaint = intake_form.chief_complaint\n            features.update({\n                'pain_scale': complaint.pain_scale or 0,\n                'num_symptoms': len(complaint.symptoms) if complaint.symptoms else 0\n            })\n            \n            # Add symptom severity scores\n            if complaint.symptoms:\n                avg_severity = sum(s.severity or 0 for s in complaint.symptoms) / len(complaint.symptoms)\n                max_severity = max(s.severity or 0 for s in complaint.symptoms)\n                features.update({\n                    'avg_symptom_severity': avg_severity,\n                    'max_symptom_severity': max_severity\n                })\n        \n        # Lab results features\n        if intake_form.lab_results:\n            features['num_lab_results'] = len(intake_form.lab_results)\n            features['num_abnormal_labs'] = sum(1 for lab in intake_form.lab_results if lab.abnormal)\n        \n        return features\n    \n    def _process_text_data(self, intake_form: PatientIntakeForm) -> Dict[str, List[float]]:\n        \"\"\"Process and embed text data from the intake form\"\"\"\n        text_embeddings = {}\n        \n        if intake_form.chief_complaint:\n            complaint = intake_form.chief_complaint\n            \n            # Process primary complaint\n            if complaint.primary_complaint:\n                cleaned_text = self._clean_text(complaint.primary_complaint)\n                embedding = self.nlp_processor.get_text_embedding(cleaned_text)\n                text_embeddings['primary_complaint'] = embedding\n            \n            # Process review of systems\n            if complaint.review_of_systems:\n                cleaned_text = self._clean_text(complaint.review_of_systems)\n                embedding = self.nlp_processor.get_text_embedding(cleaned_text)\n                text_embeddings['review_of_systems'] = embedding\n            \n            # Process individual symptoms\n            if complaint.symptoms:\n                symptom_texts = [s.description for s in complaint.symptoms]\n                combined_symptoms = ' '.join(symptom_texts)\n                cleaned_text = self._clean_text(combined_symptoms)\n                embedding = self.nlp_processor.get_text_embedding(cleaned_text)\n                text_embeddings['symptoms'] = embedding\n        \n        return text_embeddings\n    \n    def _identify_risk_factors(self, intake_form: PatientIntakeForm) -> List[str]:\n        \"\"\"Identify potential risk factors from patient data\"\"\"\n        risk_factors = []\n        \n        # Age-based risk factors\n        age = intake_form.demographics.age\n        if age >= 65:\n            risk_factors.append('elderly')\n        elif age >= 45:\n            risk_factors.append('middle_aged')\n        \n        # Vital signs risk factors\n        if intake_form.vital_signs:\n            vitals = intake_form.vital_signs\n            \n            if vitals.systolic_bp and vitals.systolic_bp >= 140:\n                risk_factors.append('hypertension')\n            if vitals.diastolic_bp and vitals.diastolic_bp >= 90:\n                risk_factors.append('hypertension')\n            if vitals.heart_rate and vitals.heart_rate > 100:\n                risk_factors.append('tachycardia')\n            if vitals.temperature and vitals.temperature > 100.4:\n                risk_factors.append('fever')\n            if vitals.bmi and vitals.bmi >= 30:\n                risk_factors.append('obesity')\n        \n        # Medical history risk factors\n        if intake_form.medical_history:\n            history = intake_form.medical_history\n            \n            if history.smoking_status and history.smoking_status.value in ['current', 'former']:\n                risk_factors.append('smoking_history')\n            if history.alcohol_consumption and history.alcohol_consumption.value == 'heavy':\n                risk_factors.append('alcohol_abuse')\n            if history.chronic_conditions:\n                risk_factors.extend([f'chronic_{condition.lower().replace(\" \", \"_\")}' for condition in history.chronic_conditions])\n        \n        # Symptom-based risk factors\n        if intake_form.chief_complaint:\n            complaint = intake_form.chief_complaint\n            \n            # Check for urgent symptoms\n            complaint_text = complaint.primary_complaint.lower()\n            for urgent_keyword in self.urgent_keywords:\n                if urgent_keyword.replace('_', ' ') in complaint_text:\n                    risk_factors.append(f'urgent_{urgent_keyword}')\n            \n            # High pain scale\n            if complaint.pain_scale and complaint.pain_scale >= 8:\n                risk_factors.append('severe_pain')\n        \n        return list(set(risk_factors))  # Remove duplicates\n    \n    def _clean_text(self, text: str) -> str:\n        \"\"\"Clean and normalize text data\"\"\"\n        if not text:\n            return \"\"\n        \n        # Convert to lowercase\n        text = text.lower()\n        \n        # Remove special characters and digits\n        text = re.sub(r'[^a-zA-Z\\s]', '', text)\n        \n        # Tokenize\n        tokens = word_tokenize(text)\n        \n        # Remove stopwords and lemmatize\n        tokens = [self.lemmatizer.lemmatize(token) for token in tokens \n                 if token not in self.stop_words and len(token) > 2]\n        \n        return ' '.join(tokens)\n    \n    def _encode_exercise_frequency(self, frequency) -> int:\n        \"\"\"Encode exercise frequency to numerical value\"\"\"\n        if not frequency:\n            return 0\n        \n        frequency_map = {\n            'none': 0,\n            'rare': 1,\n            'weekly': 2,\n            'daily': 3\n        }\n        return frequency_map.get(frequency.value, 0)\n    \n    def get_urgency_score(self, intake_form: PatientIntakeForm) -> float:\n        \"\"\"Calculate urgency score for triaging patients\"\"\"\n        score = 0.0\n        \n        # High vital signs contribute to urgency\n        if intake_form.vital_signs:\n            vitals = intake_form.vital_signs\n            if vitals.systolic_bp and vitals.systolic_bp > 180:\n                score += 0.3\n            if vitals.heart_rate and vitals.heart_rate > 120:\n                score += 0.2\n            if vitals.temperature and vitals.temperature > 103:\n                score += 0.2\n        \n        # Pain scale contributes to urgency\n        if intake_form.chief_complaint and intake_form.chief_complaint.pain_scale:\n            pain_contribution = intake_form.chief_complaint.pain_scale / 10 * 0.2\n            score += pain_contribution\n        \n        # Urgent keywords in complaint\n        if intake_form.chief_complaint:\n            complaint_text = intake_form.chief_complaint.primary_complaint.lower()\n            urgent_keywords_found = sum(1 for keyword in self.urgent_keywords \n                                      if keyword.replace('_', ' ') in complaint_text)\n            score += min(urgent_keywords_found * 0.1, 0.3)\n        \n        return min(score, 1.0)  # Cap at 1.0\n    \n    def validate_intake_form(self, form_data: dict) -> tuple[bool, List[str]]:\n        \"\"\"Validate intake form data and return validation results\"\"\"\n        errors = []\n        \n        try:\n            # Try to create the intake form model\n            PatientIntakeForm(**form_data)\n            return True, []\n        except Exception as e:\n            errors.append(str(e))\n            return False, errors
