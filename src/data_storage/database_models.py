"""
Database Models
SQLAlchemy models for patient data storage with HIPAA compliance
"""

from datetime import datetime, date
from typing import Optional, List
import uuid
from sqlalchemy import (
    Column, Integer, String, DateTime, Date, Float, Boolean, Text, 
    ForeignKey, JSON, LargeBinary, Enum as SQLEnum, Table
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.dialects.postgresql import UUID
import enum

Base = declarative_base()


class GenderEnum(enum.Enum):
    MALE = "male"
    FEMALE = "female" 
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"


class SmokingStatusEnum(enum.Enum):
    NEVER = "never"
    CURRENT = "current"
    FORMER = "former"


class AlcoholConsumptionEnum(enum.Enum):
    NONE = "none"
    OCCASIONAL = "occasional"
    MODERATE = "moderate"
    HEAVY = "heavy"


class ExerciseFrequencyEnum(enum.Enum):
    NONE = "none"
    RARE = "rare"
    WEEKLY = "weekly"
    DAILY = "daily"


class RiskLevelEnum(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Association table for patient allergies
patient_allergies = Table(
    'patient_allergies',
    Base.metadata,
    Column('patient_id', UUID(as_uuid=True), ForeignKey('patients.id')),
    Column('allergy_id', Integer, ForeignKey('allergies.id'))
)

# Association table for patient conditions
patient_conditions = Table(
    'patient_conditions',
    Base.metadata,
    Column('patient_id', UUID(as_uuid=True), ForeignKey('patients.id')),
    Column('condition_id', Integer, ForeignKey('medical_conditions.id'))
)


class Patient(Base):
    """Main patient table with demographic information"""
    __tablename__ = 'patients'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    first_name = Column(String(100), nullable=False, index=True)
    last_name = Column(String(100), nullable=False, index=True)
    date_of_birth = Column(Date, nullable=False)
    gender = Column(SQLEnum(GenderEnum), nullable=False)
    phone = Column(String(20), nullable=True)
    email = Column(String(255), nullable=True, index=True)
    address = Column(Text, nullable=True)
    emergency_contact = Column(String(200), nullable=True)
    emergency_phone = Column(String(20), nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    medical_history = relationship("MedicalHistory", back_populates="patient", uselist=False)
    vital_signs = relationship("VitalSigns", back_populates="patient")
    lab_results = relationship("LabResult", back_populates="patient")
    diagnoses = relationship("Diagnosis", back_populates="patient")
    medications = relationship("Medication", back_populates="patient")
    visits = relationship("Visit", back_populates="patient")
    allergies = relationship("Allergy", secondary=patient_allergies, back_populates="patients")
    conditions = relationship("MedicalCondition", secondary=patient_conditions, back_populates="patients")
    
    @property
    def age(self) -> int:
        """Calculate patient age"""
        today = date.today()
        return today.year - self.date_of_birth.year - (
            (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
        )
    
    @property
    def full_name(self) -> str:
        """Get patient full name"""
        return f"{self.first_name} {self.last_name}"


class MedicalHistory(Base):
    """Patient medical history"""
    __tablename__ = 'medical_histories'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(UUID(as_uuid=True), ForeignKey('patients.id'), unique=True, nullable=False)
    
    # Lifestyle factors
    smoking_status = Column(SQLEnum(SmokingStatusEnum), nullable=True)
    alcohol_consumption = Column(SQLEnum(AlcoholConsumptionEnum), nullable=True)
    exercise_frequency = Column(SQLEnum(ExerciseFrequencyEnum), nullable=True)
    diet_preferences = Column(JSON, nullable=True)  # List of diet preferences
    occupation = Column(String(200), nullable=True)
    
    # Family history (JSON format for flexibility)
    family_history = Column(JSON, nullable=True)
    
    # Immunizations
    immunizations = Column(JSON, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="medical_history")


class VitalSigns(Base):
    """Patient vital signs measurements"""
    __tablename__ = 'vital_signs'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(UUID(as_uuid=True), ForeignKey('patients.id'), nullable=False, index=True)
    visit_id = Column(Integer, ForeignKey('visits.id'), nullable=True)
    
    # Vital measurements
    systolic_bp = Column(Integer, nullable=True)
    diastolic_bp = Column(Integer, nullable=True)
    heart_rate = Column(Integer, nullable=True)
    temperature = Column(Float, nullable=True)  # Fahrenheit
    respiratory_rate = Column(Integer, nullable=True)
    oxygen_saturation = Column(Integer, nullable=True)
    height = Column(Float, nullable=True)  # cm
    weight = Column(Float, nullable=True)  # kg
    
    # Calculated fields
    bmi = Column(Float, nullable=True)
    
    # Metadata
    measured_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    patient = relationship("Patient", back_populates="vital_signs")
    visit = relationship("Visit", back_populates="vital_signs")


class LabResult(Base):
    """Laboratory test results"""
    __tablename__ = 'lab_results'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(UUID(as_uuid=True), ForeignKey('patients.id'), nullable=False, index=True)
    visit_id = Column(Integer, ForeignKey('visits.id'), nullable=True)
    
    test_name = Column(String(200), nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(50), nullable=False)
    reference_range = Column(String(100), nullable=True)
    abnormal = Column(Boolean, nullable=True)
    notes = Column(Text, nullable=True)
    
    # Metadata
    test_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    patient = relationship("Patient", back_populates="lab_results")
    visit = relationship("Visit", back_populates="lab_results")


class Visit(Base):
    """Patient visits to clinic"""
    __tablename__ = 'visits'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(UUID(as_uuid=True), ForeignKey('patients.id'), nullable=False, index=True)
    
    visit_date = Column(DateTime, nullable=False)
    visit_type = Column(String(100), nullable=True)  # consultation, follow-up, emergency
    chief_complaint = Column(Text, nullable=True)
    history_of_present_illness = Column(Text, nullable=True)
    review_of_systems = Column(Text, nullable=True)
    physical_examination = Column(Text, nullable=True)
    assessment = Column(Text, nullable=True)
    plan = Column(Text, nullable=True)
    
    # Pain and urgency
    pain_scale = Column(Integer, nullable=True)  # 0-10 scale
    urgency_score = Column(Float, nullable=True)  # Calculated urgency
    
    # Provider information
    provider_name = Column(String(200), nullable=True)
    provider_id = Column(String(100), nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="visits")
    vital_signs = relationship("VitalSigns", back_populates="visit")
    lab_results = relationship("LabResult", back_populates="visit")
    diagnoses = relationship("Diagnosis", back_populates="visit")
    symptoms = relationship("Symptom", back_populates="visit")


class Symptom(Base):
    """Individual symptoms reported by patient"""
    __tablename__ = 'symptoms'
    
    id = Column(Integer, primary_key=True)
    visit_id = Column(Integer, ForeignKey('visits.id'), nullable=False, index=True)
    
    description = Column(String(500), nullable=False)
    severity = Column(Integer, nullable=True)  # 1-10 scale
    duration = Column(String(100), nullable=True)  # e.g., "3 days", "2 weeks"
    frequency = Column(String(100), nullable=True)  # constant, intermittent
    triggers = Column(JSON, nullable=True)  # List of triggers
    relief_factors = Column(JSON, nullable=True)  # List of relief factors
    associated_symptoms = Column(JSON, nullable=True)  # List of associated symptoms
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    visit = relationship("Visit", back_populates="symptoms")


class Diagnosis(Base):
    """AI and doctor diagnoses"""
    __tablename__ = 'diagnoses'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(UUID(as_uuid=True), ForeignKey('patients.id'), nullable=False, index=True)
    visit_id = Column(Integer, ForeignKey('visits.id'), nullable=True, index=True)
    
    # Diagnosis information
    condition_name = Column(String(200), nullable=False, index=True)
    icd_10_code = Column(String(10), nullable=True, index=True)
    confidence_score = Column(Float, nullable=True)  # AI confidence (0-1)
    risk_level = Column(SQLEnum(RiskLevelEnum), nullable=True)
    
    # Source information
    diagnosis_type = Column(String(50), nullable=False)  # 'ai_prediction', 'doctor_confirmed', 'differential'
    diagnosed_by = Column(String(200), nullable=True)  # doctor name or AI model version
    
    # Supporting information
    supporting_evidence = Column(JSON, nullable=True)  # Symptoms, labs, etc.
    recommendations = Column(JSON, nullable=True)  # Recommended actions
    
    # Status
    is_confirmed = Column(Boolean, default=False, nullable=False)
    is_ruled_out = Column(Boolean, default=False, nullable=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="diagnoses")
    visit = relationship("Visit", back_populates="diagnoses")


class Medication(Base):
    """Patient medications"""
    __tablename__ = 'medications'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(UUID(as_uuid=True), ForeignKey('patients.id'), nullable=False, index=True)
    
    name = Column(String(200), nullable=False, index=True)
    dosage = Column(String(100), nullable=True)
    frequency = Column(String(100), nullable=True)
    route = Column(String(50), nullable=True)  # oral, IV, etc.
    
    # Dates
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    
    # Prescriber
    prescribing_doctor = Column(String(200), nullable=True)
    prescription_date = Column(Date, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Notes
    notes = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="medications")


class Allergy(Base):
    """Allergy reference table"""
    __tablename__ = 'allergies'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False, unique=True, index=True)
    category = Column(String(100), nullable=True)  # drug, food, environmental
    severity = Column(String(50), nullable=True)  # mild, moderate, severe
    
    # Relationships
    patients = relationship("Patient", secondary=patient_allergies, back_populates="allergies")


class MedicalCondition(Base):
    """Medical condition reference table"""
    __tablename__ = 'medical_conditions'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False, unique=True, index=True)
    icd_10_code = Column(String(10), nullable=True, index=True)
    category = Column(String(100), nullable=True)  # cardiovascular, respiratory, etc.
    description = Column(Text, nullable=True)
    
    # Relationships
    patients = relationship("Patient", secondary=patient_conditions, back_populates="conditions")


class MLModelMetadata(Base):
    """Track ML model versions and performance"""
    __tablename__ = 'ml_model_metadata'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # xgboost, lightgbm, etc.
    
    # Performance metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    # Training information
    training_data_size = Column(Integer, nullable=True)
    training_date = Column(DateTime, nullable=False)
    hyperparameters = Column(JSON, nullable=True)
    
    # Model file information
    model_path = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=False, nullable=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class AuditLog(Base):
    """Audit log for HIPAA compliance"""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    user_id = Column(String(100), nullable=True, index=True)
    action = Column(String(100), nullable=False, index=True)  # create, read, update, delete
    table_name = Column(String(100), nullable=True)
    record_id = Column(String(100), nullable=True)
    
    # Details
    details = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)


class EncryptedData(Base):
    """Store encrypted sensitive data"""
    __tablename__ = 'encrypted_data'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(UUID(as_uuid=True), ForeignKey('patients.id'), nullable=False, index=True)
    data_type = Column(String(100), nullable=False)  # notes, images, etc.
    encrypted_content = Column(LargeBinary, nullable=False)
    encryption_method = Column(String(50), nullable=False, default='AES-256')
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
