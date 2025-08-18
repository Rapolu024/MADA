"""
Sample Medical Data Generator
Creates realistic synthetic patient data for testing and validation
"""

import random
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Any
import uuid
import logging

logger = logging.getLogger(__name__)


class MedicalDataGenerator:
    """
    Generates realistic synthetic medical data for testing
    Mimics real patient data patterns without using actual patient information
    """
    
    def __init__(self, seed: int = 42):
        """Initialize with seed for reproducible data generation"""
        random.seed(seed)
        np.random.seed(seed)
        
        # Medical data patterns and distributions
        self.disease_prevalence = {
            'diabetes_type_2': 0.11,
            'hypertension': 0.45,
            'heart_disease': 0.06,
            'asthma': 0.08,
            'obesity': 0.36,
            'depression': 0.08,
            'anxiety': 0.18
        }
        
        # Symptom associations with diseases
        self.disease_symptoms = {
            'diabetes_type_2': [
                'increased thirst', 'frequent urination', 'fatigue', 
                'blurred vision', 'slow healing wounds'
            ],
            'hypertension': [
                'headaches', 'dizziness', 'chest pain',
                'shortness of breath', 'nosebleeds'
            ],
            'heart_disease': [
                'chest pain', 'shortness of breath', 'fatigue',
                'swelling in legs', 'irregular heartbeat'
            ],
            'asthma': [
                'wheezing', 'coughing', 'shortness of breath',
                'chest tightness', 'difficulty breathing'
            ]
        }
        
        # Lab test normal ranges
        self.lab_ranges = {
            'glucose': {'normal': (70, 100), 'unit': 'mg/dL'},
            'hba1c': {'normal': (4.0, 5.6), 'unit': '%'},
            'cholesterol_total': {'normal': (0, 200), 'unit': 'mg/dL'},
            'hdl_cholesterol': {'normal': (40, 1000), 'unit': 'mg/dL'},
            'ldl_cholesterol': {'normal': (0, 100), 'unit': 'mg/dL'},
            'triglycerides': {'normal': (0, 150), 'unit': 'mg/dL'},
            'hemoglobin': {'normal': (12, 16), 'unit': 'g/dL'},
            'white_blood_cell': {'normal': (4000, 11000), 'unit': '/μL'},
            'platelets': {'normal': (150000, 400000), 'unit': '/μL'}
        }
        
        # Name lists for generating patient names
        self.first_names = {
            'male': ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Joseph', 'Thomas', 'Christopher'],
            'female': ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 'Jessica', 'Sarah', 'Karen']
        }
        
        self.last_names = [
            'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
            'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin'
        ]
        
        # Medication lists by condition
        self.medications_by_condition = {
            'diabetes_type_2': ['metformin', 'insulin', 'glipizide', 'sitagliptin'],
            'hypertension': ['lisinopril', 'amlodipine', 'metoprolol', 'hydrochlorothiazide'],
            'heart_disease': ['atorvastatin', 'aspirin', 'clopidogrel', 'carvedilol'],
            'asthma': ['albuterol', 'fluticasone', 'montelukast', 'budesonide']
        }
    
    def generate_patient_demographics(self) -> Dict[str, Any]:
        """Generate realistic patient demographics"""
        gender = random.choice(['male', 'female'])
        age = int(np.random.normal(45, 15))
        age = max(18, min(85, age))  # Constrain age between 18-85
        
        # Generate birth date based on age
        birth_year = datetime.now().year - age
        birth_date = date(birth_year, random.randint(1, 12), random.randint(1, 28))
        
        return {
            'patient_id': str(uuid.uuid4()),
            'first_name': random.choice(self.first_names[gender]),
            'last_name': random.choice(self.last_names),
            'date_of_birth': birth_date,
            'age': age,
            'gender': gender,
            'phone': f'+1{random.randint(1000000000, 9999999999)}',
            'email': None  # Will be generated if needed
        }
    
    def generate_vital_signs(self, conditions: List[str] = None) -> Dict[str, Any]:
        """Generate vital signs with realistic correlations to health conditions"""
        conditions = conditions or []
        
        # Base vital signs for healthy individual
        base_vitals = {
            'systolic_bp': int(np.random.normal(120, 10)),
            'diastolic_bp': int(np.random.normal(80, 8)),
            'heart_rate': int(np.random.normal(72, 12)),
            'temperature': round(np.random.normal(98.6, 0.5), 1),
            'respiratory_rate': int(np.random.normal(16, 2)),
            'oxygen_saturation': int(np.random.normal(98, 1)),
            'height': round(np.random.normal(170, 10), 0),
            'weight': round(np.random.normal(75, 15), 1)
        }
        
        # Adjust vitals based on conditions
        if 'hypertension' in conditions:
            base_vitals['systolic_bp'] += random.randint(20, 40)
            base_vitals['diastolic_bp'] += random.randint(10, 20)
        
        if 'heart_disease' in conditions:
            base_vitals['heart_rate'] += random.randint(10, 25)
        
        if 'obesity' in conditions:
            base_vitals['weight'] += random.randint(15, 30)
        
        # Calculate BMI
        height_m = base_vitals['height'] / 100
        bmi = base_vitals['weight'] / (height_m ** 2)
        base_vitals['bmi'] = round(bmi, 1)
        
        # Constrain values to realistic ranges
        base_vitals['systolic_bp'] = max(90, min(200, base_vitals['systolic_bp']))
        base_vitals['diastolic_bp'] = max(60, min(120, base_vitals['diastolic_bp']))
        base_vitals['heart_rate'] = max(50, min(120, base_vitals['heart_rate']))
        
        return base_vitals
    
    def generate_medical_history(self, conditions: List[str] = None) -> Dict[str, Any]:
        """Generate medical history based on patient conditions"""
        conditions = conditions or []
        
        history = {
            'smoking_status': random.choices(
                ['never', 'former', 'current'], 
                weights=[0.6, 0.25, 0.15]
            )[0],
            'alcohol_consumption': random.choices(
                ['none', 'occasional', 'moderate', 'heavy'],
                weights=[0.3, 0.4, 0.25, 0.05]
            )[0],
            'exercise_frequency': random.choices(
                ['none', 'rare', 'weekly', 'daily'],
                weights=[0.2, 0.3, 0.35, 0.15]
            )[0],
            'allergies': [],
            'current_medications': [],
            'chronic_conditions': conditions,
            'family_history': {},
            'immunizations': []
        }
        
        # Add allergies (10% chance)
        if random.random() < 0.1:
            allergies = ['penicillin', 'shellfish', 'peanuts', 'latex', 'dust']
            history['allergies'] = random.sample(allergies, random.randint(1, 2))
        
        # Add medications based on conditions
        for condition in conditions:
            if condition in self.medications_by_condition:
                meds = self.medications_by_condition[condition]
                selected_meds = random.sample(meds, random.randint(1, min(2, len(meds))))
                for med in selected_meds:
                    history['current_medications'].append({
                        'name': med,
                        'dosage': f'{random.choice([25, 50, 100, 200])}mg',
                        'frequency': random.choice(['once daily', 'twice daily', 'three times daily'])
                    })
        
        # Add family history (30% chance for each major condition)
        family_conditions = ['diabetes', 'heart_disease', 'cancer', 'hypertension']
        for condition in family_conditions:
            if random.random() < 0.3:
                relatives = random.sample(['father', 'mother', 'sibling', 'grandparent'], 
                                        random.randint(1, 2))
                history['family_history'][condition] = relatives
        
        return history
    
    def generate_chief_complaint(self, conditions: List[str] = None) -> Dict[str, Any]:
        """Generate chief complaint and symptoms based on conditions"""
        conditions = conditions or []
        
        # Select primary condition for complaint
        if conditions:
            primary_condition = random.choice(conditions)
            if primary_condition in self.disease_symptoms:
                primary_symptoms = self.disease_symptoms[primary_condition]
                complaint = random.choice(primary_symptoms)
            else:
                complaint = "General discomfort and fatigue"
        else:
            # General complaints for healthy patients
            complaints = [
                "Routine checkup", "General fatigue", "Minor headaches",
                "Occasional dizziness", "Sleep issues"
            ]
            complaint = random.choice(complaints)
        
        # Generate detailed symptoms
        symptoms = []
        if conditions:
            for condition in conditions:
                if condition in self.disease_symptoms:
                    condition_symptoms = self.disease_symptoms[condition]
                    selected_symptoms = random.sample(
                        condition_symptoms, 
                        random.randint(1, min(3, len(condition_symptoms)))
                    )
                    
                    for symptom in selected_symptoms:
                        symptoms.append({
                            'description': symptom,
                            'severity': random.randint(3, 8),
                            'duration': random.choice([
                                '2 days', '1 week', '2 weeks', '1 month', '3 months'
                            ]),
                            'frequency': random.choice([
                                'constant', 'intermittent', 'daily', 'occasional'
                            ])
                        })
        
        return {
            'primary_complaint': complaint,
            'pain_scale': random.randint(0, 7),
            'symptoms': symptoms,
            'onset_date': datetime.now() - timedelta(days=random.randint(1, 30)),
            'review_of_systems': f"Patient reports {complaint.lower()}"
        }
    
    def generate_lab_results(self, conditions: List[str] = None) -> List[Dict[str, Any]]:
        """Generate lab results based on patient conditions"""
        conditions = conditions or []
        
        lab_results = []
        
        # Common lab tests
        tests_to_generate = ['glucose', 'cholesterol_total', 'hemoglobin', 'white_blood_cell']
        
        # Add condition-specific tests
        if 'diabetes_type_2' in conditions:
            tests_to_generate.extend(['hba1c', 'glucose'])
        
        if any(condition in ['heart_disease', 'hypertension'] for condition in conditions):
            tests_to_generate.extend(['hdl_cholesterol', 'ldl_cholesterol', 'triglycerides'])
        
        # Remove duplicates
        tests_to_generate = list(set(tests_to_generate))
        
        for test_name in tests_to_generate:
            if test_name in self.lab_ranges:
                normal_range = self.lab_ranges[test_name]['normal']
                unit = self.lab_ranges[test_name]['unit']
                
                # Generate value based on conditions
                if 'diabetes_type_2' in conditions and test_name in ['glucose', 'hba1c']:
                    # Elevated values for diabetic patients
                    if test_name == 'glucose':
                        value = round(np.random.normal(140, 20), 1)
                    else:  # hba1c
                        value = round(np.random.normal(7.5, 1.0), 1)
                    abnormal = True
                elif 'heart_disease' in conditions and 'cholesterol' in test_name:
                    # Elevated cholesterol for heart disease patients
                    value = round(np.random.normal(normal_range[1] + 50, 30), 1)
                    abnormal = value > normal_range[1]
                else:
                    # Normal distribution around normal range
                    mean_val = np.mean(normal_range)
                    std_val = (normal_range[1] - normal_range[0]) / 6
                    value = round(np.random.normal(mean_val, std_val), 1)
                    abnormal = value < normal_range[0] or value > normal_range[1]
                
                lab_results.append({
                    'test_name': test_name,
                    'value': value,
                    'unit': unit,
                    'reference_range': f"{normal_range[0]}-{normal_range[1]} {unit}",
                    'abnormal': abnormal,
                    'test_date': datetime.now() - timedelta(days=random.randint(0, 7))
                })
        
        return lab_results
    
    def generate_single_patient(self, force_conditions: List[str] = None) -> Dict[str, Any]:
        """Generate a complete patient record"""
        
        # Determine conditions for this patient
        if force_conditions:
            conditions = force_conditions
        else:
            conditions = []
            for condition, prevalence in self.disease_prevalence.items():
                if random.random() < prevalence:
                    conditions.append(condition)
        
        # Generate all patient data
        demographics = self.generate_patient_demographics()
        vital_signs = self.generate_vital_signs(conditions)
        medical_history = self.generate_medical_history(conditions)
        chief_complaint = self.generate_chief_complaint(conditions)
        lab_results = self.generate_lab_results(conditions)
        
        return {
            'demographics': demographics,
            'vital_signs': vital_signs,
            'medical_history': medical_history,
            'chief_complaint': chief_complaint,
            'lab_results': lab_results,
            '_metadata': {
                'generated_conditions': conditions,
                'generation_timestamp': datetime.now(),
                'data_source': 'synthetic'
            }
        }
    
    def generate_patient_cohort(self, n_patients: int = 100, 
                               condition_distribution: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Generate a cohort of patients with specified condition distribution"""
        
        patients = []
        
        for i in range(n_patients):
            # Determine conditions for this patient
            if condition_distribution:
                conditions = []
                for condition, probability in condition_distribution.items():
                    if random.random() < probability:
                        conditions.append(condition)
                patient = self.generate_single_patient(conditions)
            else:
                patient = self.generate_single_patient()
            
            patients.append(patient)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{n_patients} patients")
        
        logger.info(f"Successfully generated cohort of {n_patients} patients")
        return patients
    
    def generate_training_dataset(self, n_patients: int = 1000) -> Tuple[List[Dict], List[List[str]]]:
        """Generate a balanced training dataset with known diagnoses"""
        
        # Create balanced dataset with known conditions
        conditions_per_group = n_patients // len(self.disease_prevalence)
        patients = []
        diagnoses = []
        
        for condition in self.disease_prevalence.keys():
            for _ in range(conditions_per_group):
                # Generate patient with specific condition
                patient = self.generate_single_patient([condition])
                patients.append(patient)
                diagnoses.append([condition])
        
        # Add some patients with multiple conditions
        multi_condition_patients = n_patients - len(patients)
        for _ in range(multi_condition_patients):
            # Randomly select 1-3 conditions
            selected_conditions = random.sample(
                list(self.disease_prevalence.keys()), 
                random.randint(1, 3)
            )
            patient = self.generate_single_patient(selected_conditions)
            patients.append(patient)
            diagnoses.append(selected_conditions)
        
        # Shuffle the dataset
        combined = list(zip(patients, diagnoses))
        random.shuffle(combined)
        patients, diagnoses = zip(*combined)
        
        logger.info(f"Generated training dataset: {len(patients)} patients")
        return list(patients), list(diagnoses)


def main():
    """Demo script showing data generation capabilities"""
    generator = MedicalDataGenerator()
    
    print("🏥 MADA Sample Data Generator")
    print("=" * 40)
    
    # Generate single patient
    print("\n1. Single Patient Example:")
    patient = generator.generate_single_patient(['diabetes_type_2', 'hypertension'])
    print(f"Patient: {patient['demographics']['first_name']} {patient['demographics']['last_name']}")
    print(f"Age: {patient['demographics']['age']}")
    print(f"Conditions: {patient['_metadata']['generated_conditions']}")
    print(f"Chief Complaint: {patient['chief_complaint']['primary_complaint']}")
    
    # Generate small cohort
    print(f"\n2. Patient Cohort (n=10):")
    cohort = generator.generate_patient_cohort(n_patients=10)
    for i, p in enumerate(cohort[:5]):
        conditions = p['_metadata']['generated_conditions']
        print(f"  Patient {i+1}: {len(conditions)} conditions")
    
    # Generate training dataset
    print(f"\n3. Training Dataset (n=50):")
    patients, diagnoses = generator.generate_training_dataset(n_patients=50)
    condition_counts = {}
    for diag_list in diagnoses:
        for condition in diag_list:
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
    
    print("Condition distribution:")
    for condition, count in condition_counts.items():
        print(f"  {condition}: {count} patients")


if __name__ == "__main__":
    main()
