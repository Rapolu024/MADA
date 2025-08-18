"""
Patient Data Models
Pydantic models for patient data validation and structure
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"


class SmokingStatus(str, Enum):
    NEVER = "never"
    CURRENT = "current"
    FORMER = "former"


class AlcoholConsumption(str, Enum):
    NONE = "none"
    OCCASIONAL = "occasional"
    MODERATE = "moderate"
    HEAVY = "heavy"


class ExerciseFrequency(str, Enum):
    NONE = "none"
    RARE = "rare"
    WEEKLY = "weekly"
    DAILY = "daily"


class Demographics(BaseModel):
    """Patient demographic information"""
    patient_id: Optional[str] = None
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    date_of_birth: date
    gender: Gender
    phone: Optional[str] = Field(None, regex=r'^\+?[\d\s\-\(\)]+$')
    email: Optional[str] = Field(None, regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    address: Optional[str] = None
    emergency_contact: Optional[str] = None
    emergency_phone: Optional[str] = None
    
    @validator('date_of_birth')
    def validate_age(cls, v):
        today = date.today()
        age = today.year - v.year - ((today.month, today.day) < (v.month, v.day))
        if age < 0 or age > 150:
            raise ValueError('Invalid age')
        return v
    
    @property
    def age(self) -> int:
        """Calculate age from date of birth"""
        today = date.today()
        return today.year - self.date_of_birth.year - (
            (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
        )


class VitalSigns(BaseModel):
    """Patient vital signs"""
    systolic_bp: Optional[int] = Field(None, ge=60, le=250)
    diastolic_bp: Optional[int] = Field(None, ge=30, le=150)
    heart_rate: Optional[int] = Field(None, ge=30, le=200)
    temperature: Optional[float] = Field(None, ge=95.0, le=110.0)  # Fahrenheit
    respiratory_rate: Optional[int] = Field(None, ge=8, le=40)
    oxygen_saturation: Optional[int] = Field(None, ge=70, le=100)
    height: Optional[float] = Field(None, ge=50, le=250)  # cm
    weight: Optional[float] = Field(None, ge=30, le=500)  # kg
    
    @property
    def bmi(self) -> Optional[float]:
        """Calculate BMI if height and weight are available"""
        if self.height and self.weight:
            height_m = self.height / 100
            return round(self.weight / (height_m ** 2), 2)
        return None


class Medication(BaseModel):
    """Individual medication entry"""
    name: str = Field(..., min_length=1, max_length=200)
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    prescribing_doctor: Optional[str] = None
    notes: Optional[str] = None


class MedicalHistory(BaseModel):
    """Patient medical history"""
    allergies: Optional[List[str]] = []
    current_medications: Optional[List[Medication]] = []
    past_medications: Optional[List[str]] = []
    chronic_conditions: Optional[List[str]] = []
    past_surgeries: Optional[List[Dict[str, Any]]] = []
    family_history: Optional[Dict[str, List[str]]] = {}
    immunizations: Optional[List[Dict[str, Any]]] = []
    smoking_status: Optional[SmokingStatus] = None
    alcohol_consumption: Optional[AlcoholConsumption] = None
    exercise_frequency: Optional[ExerciseFrequency] = None
    diet_preferences: Optional[List[str]] = []
    occupation: Optional[str] = None


class Symptom(BaseModel):
    """Individual symptom entry"""
    description: str = Field(..., min_length=1, max_length=500)
    severity: Optional[int] = Field(None, ge=1, le=10)  # 1-10 scale
    duration: Optional[str] = None  # e.g., "3 days", "2 weeks"
    frequency: Optional[str] = None  # e.g., "constant", "intermittent"
    triggers: Optional[List[str]] = []
    relief_factors: Optional[List[str]] = []
    associated_symptoms: Optional[List[str]] = []


class ChiefComplaint(BaseModel):
    """Patient's primary complaint and symptoms"""
    primary_complaint: str = Field(..., min_length=1, max_length=1000)
    symptoms: List[Symptom] = []
    onset_date: Optional[date] = None
    precipitating_factors: Optional[List[str]] = []
    review_of_systems: Optional[str] = None
    pain_scale: Optional[int] = Field(None, ge=0, le=10)


class LabResult(BaseModel):
    """Laboratory test result"""
    test_name: str
    value: float
    unit: str
    reference_range: Optional[str] = None
    abnormal: Optional[bool] = None
    test_date: datetime
    notes: Optional[str] = None


class PatientIntakeForm(BaseModel):
    """Complete patient intake form"""
    timestamp: datetime = Field(default_factory=datetime.now)
    demographics: Demographics
    vital_signs: Optional[VitalSigns] = None
    medical_history: Optional[MedicalHistory] = None
    chief_complaint: Optional[ChiefComplaint] = None
    lab_results: Optional[List[LabResult]] = []
    insurance_info: Optional[Dict[str, str]] = {}
    consent_given: bool = False
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat()
        }


class ProcessedPatientData(BaseModel):
    """Patient data after preprocessing and feature extraction"""
    patient_id: str
    raw_data: PatientIntakeForm
    processed_features: Dict[str, Any] = {}
    text_embeddings: Optional[Dict[str, List[float]]] = {}
    risk_factors: Optional[List[str]] = []
    preprocessing_timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
