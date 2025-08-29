#!/usr/bin/env python3
"""
MADA Dashboard Runner
Simplified standalone version of the Medical AI Diagnosis Assistant
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Any
import time
import random

# Page configuration
st.set_page_config(
    page_title="MADA - Medical AI Diagnosis Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
    }
    
    .urgent-alert {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .diagnosis-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def generate_sample_data():
    """Generate sample patient data for demonstration"""
    return [
        {
            "patient_id": "DEMO_001",
            "name": "John Smith",
            "age": 52,
            "gender": "Male",
            "last_visit": "2024-08-15",
            "chief_complaint": "Increased thirst and frequent urination",
            "risk_level": "High",
            "urgent_flags": ["possible_diabetes"],
            "status": "Pending Review",
            "vitals": {
                "bp": "145/92",
                "heart_rate": 78,
                "temperature": 98.6,
                "weight": 95
            }
        },
        {
            "patient_id": "DEMO_002", 
            "name": "Sarah Johnson",
            "age": 48,
            "gender": "Female",
            "last_visit": "2024-08-16",
            "chief_complaint": "Chest tightness during exertion",
            "risk_level": "High",
            "urgent_flags": ["cardiovascular_risk"],
            "status": "Under Investigation",
            "vitals": {
                "bp": "165/98",
                "heart_rate": 95,
                "temperature": 98.4,
                "weight": 78
            }
        },
        {
            "patient_id": "DEMO_003",
            "name": "Michael Brown",
            "age": 35,
            "gender": "Male", 
            "last_visit": "2024-08-17",
            "chief_complaint": "Persistent headaches",
            "risk_level": "Medium",
            "urgent_flags": [],
            "status": "Scheduled Follow-up",
            "vitals": {
                "bp": "130/85",
                "heart_rate": 72,
                "temperature": 98.2,
                "weight": 82
            }
        }
    ]

def generate_smart_diagnosis(patient_data):
    """Generate smart diagnosis based on patient data using rule-based logic"""
    diagnoses = []
    recommendations = {
        'immediate': [],
        'follow_up': [],
        'lifestyle': [],
        'monitoring': []
    }
    
    age = patient_data['age']
    gender = patient_data['gender']
    systolic_bp = patient_data['systolic_bp']
    diastolic_bp = patient_data['diastolic_bp']
    heart_rate = patient_data['heart_rate']
    temperature = patient_data['temperature']
    chief_complaint = patient_data['chief_complaint'].lower() if patient_data['chief_complaint'] else ''
    symptoms = patient_data['symptoms']
    duration = patient_data['duration']
    pain_scale = patient_data['pain_scale']
    
    # Risk scoring
    risk_score = 0.0
    risk_factors = []
    
    # Hypertension assessment
    if systolic_bp >= 180 or diastolic_bp >= 110:
        diagnoses.append({
            'condition': 'Hypertensive Crisis',
            'confidence': 0.95,
            'urgency': 'High',
            'reasoning': f'Severely elevated BP ({systolic_bp}/{diastolic_bp})'
        })
        recommendations['immediate'].append('Emergency medical evaluation required')
        risk_score += 0.4
        risk_factors.append('Severe hypertension')
    elif systolic_bp >= 140 or diastolic_bp >= 90:
        diagnoses.append({
            'condition': 'Hypertension',
            'confidence': 0.85,
            'urgency': 'Medium',
            'reasoning': f'Elevated blood pressure ({systolic_bp}/{diastolic_bp})'
        })
        recommendations['follow_up'].append('24-hour blood pressure monitoring')
        recommendations['lifestyle'].append('Reduce sodium intake to <2300mg/day')
        risk_score += 0.2
        risk_factors.append('Elevated blood pressure')
    
    # Diabetes screening
    if age > 45 or ('thirst' in chief_complaint and 'urination' in chief_complaint):
        confidence = 0.75 if ('thirst' in chief_complaint and 'urination' in chief_complaint) else 0.35
        reasoning = 'Classic symptoms present' if confidence > 0.5 else 'Age-based screening indicated'
        
        diagnoses.append({
            'condition': 'Type 2 Diabetes Mellitus',
            'confidence': confidence,
            'urgency': 'Medium' if confidence > 0.5 else 'Low',
            'reasoning': reasoning
        })
        recommendations['follow_up'].append('Fasting glucose test')
        recommendations['follow_up'].append('HbA1c test')
        if confidence > 0.5:
            risk_score += 0.25
            risk_factors.append('Diabetic symptoms')
    
    # Cardiovascular assessment
    if 'chest' in chief_complaint or 'Chest pain' in symptoms:
        confidence = 0.80 if age > 40 else 0.60
        diagnoses.append({
            'condition': 'Coronary Artery Disease',
            'confidence': confidence,
            'urgency': 'High',
            'reasoning': 'Chest symptoms with cardiovascular risk factors'
        })
        recommendations['immediate'].append('12-lead ECG')
        recommendations['immediate'].append('Cardiac enzymes (troponin)')
        recommendations['follow_up'].append('Cardiology consultation')
        risk_score += 0.3
        risk_factors.append('Chest symptoms')
    
    # Respiratory assessment
    if 'Shortness of breath' in symptoms or 'Cough' in symptoms:
        diagnoses.append({
            'condition': 'Respiratory Infection',
            'confidence': 0.65,
            'urgency': 'Medium',
            'reasoning': 'Respiratory symptoms present'
        })
        if temperature > 101:
            recommendations['immediate'].append('Chest X-ray')
            risk_score += 0.15
        else:
            recommendations['follow_up'].append('Consider chest X-ray if symptoms persist')
    
    # Fever assessment
    if temperature > 101.3 or 'Fever' in symptoms:
        diagnoses.append({
            'condition': 'Infectious Process',
            'confidence': 0.70,
            'urgency': 'Medium' if temperature < 103 else 'High',
            'reasoning': f'Elevated temperature ({temperature}°F)'
        })
        recommendations['follow_up'].append('Complete blood count with differential')
        recommendations['follow_up'].append('Blood cultures if fever persists')
        if temperature > 103:
            risk_score += 0.2
            risk_factors.append('High fever')
    
    # Neurological assessment
    if 'Headache' in symptoms or 'headache' in chief_complaint:
        confidence = 0.45 if pain_scale < 7 else 0.65
        urgency = 'Low' if pain_scale < 7 else 'Medium'
        
        diagnoses.append({
            'condition': 'Primary Headache Disorder',
            'confidence': confidence,
            'urgency': urgency,
            'reasoning': f'Headache symptoms (pain level {pain_scale}/10)'
        })
        
        if pain_scale >= 8:
            recommendations['immediate'].append('Neurological examination')
            risk_score += 0.1
        else:
            recommendations['lifestyle'].append('Stress management techniques')
            recommendations['lifestyle'].append('Regular sleep schedule')
    
    # Heart rate assessment
    if heart_rate > 100:
        diagnoses.append({
            'condition': 'Tachycardia',
            'confidence': 0.90,
            'urgency': 'Medium' if heart_rate < 130 else 'High',
            'reasoning': f'Elevated heart rate ({heart_rate} bpm)'
        })
        recommendations['immediate'].append('ECG to evaluate rhythm')
        if heart_rate > 130:
            risk_score += 0.2
            risk_factors.append('Severe tachycardia')
    elif heart_rate < 60:
        diagnoses.append({
            'condition': 'Bradycardia',
            'confidence': 0.85,
            'urgency': 'Low',
            'reasoning': f'Low heart rate ({heart_rate} bpm)'
        })
        recommendations['follow_up'].append('ECG to evaluate conduction')
    
    # Age-related risk factors
    if age > 65:
        risk_score += 0.1
        risk_factors.append('Advanced age')
        recommendations['monitoring'].append('Regular health screenings')
    
    # Gender-specific recommendations
    if gender == 'Female' and age > 50:
        recommendations['follow_up'].append('Mammography screening')
        recommendations['follow_up'].append('Bone density screening')
    elif gender == 'Male' and age > 50:
        recommendations['follow_up'].append('PSA screening discussion')
    
    # Duration-based urgency adjustment
    duration_multiplier = {
        'Less than 24 hours': 1.2,
        '1-3 days': 1.0,
        '1 week': 0.9,
        '2-4 weeks': 0.8,
        'More than 1 month': 0.7
    }
    
    # Apply duration multiplier to confidence scores
    multiplier = duration_multiplier.get(duration, 1.0)
    for diag in diagnoses:
        diag['confidence'] = min(diag['confidence'] * multiplier, 0.99)
    
    # Sort diagnoses by confidence
    diagnoses.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Ensure we have at least some diagnoses
    if not diagnoses:
        diagnoses.append({
            'condition': 'Non-specific symptoms',
            'confidence': 0.30,
            'urgency': 'Low',
            'reasoning': 'Symptoms require further evaluation'
        })
        recommendations['follow_up'].append('Comprehensive physical examination')
        recommendations['follow_up'].append('Basic laboratory studies')
    
    # General recommendations
    recommendations['lifestyle'].extend([
        'Maintain healthy diet with fruits and vegetables',
        'Regular physical activity as tolerated',
        'Adequate hydration',
        'Follow up with primary care physician'
    ])
    
    recommendations['monitoring'].extend([
        'Monitor vital signs regularly',
        'Keep symptom diary',
        'Return if symptoms worsen'
    ])
    
    # Risk assessment
    if risk_score >= 0.7:
        risk_level = 'Critical'
        urgency = 'Immediate'
        timeframe = 'Within hours'
    elif risk_score >= 0.5:
        risk_level = 'High'
        urgency = 'Urgent'
        timeframe = 'Within 24 hours'
    elif risk_score >= 0.3:
        risk_level = 'Medium'
        urgency = 'Semi-urgent'
        timeframe = 'Within 1-2 days'
    else:
        risk_level = 'Low'
        urgency = 'Routine'
        timeframe = 'Within 1-2 weeks'
    
    risk_assessment = {
        'level': risk_level,
        'score': risk_score,
        'urgency': urgency,
        'timeframe': timeframe,
        'factors': risk_factors
    }
    
    return {
        'diagnoses': diagnoses[:5],  # Top 5 diagnoses
        'recommendations': recommendations,
        'risk_assessment': risk_assessment
    }

def render_header():
    """Render dashboard header"""
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🏥 MADA - Medical AI Diagnosis Assistant</h1>
        <p style="color: #e0e0e0; margin: 0;">AI-Powered Early Disease Detection System</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar navigation"""
    st.sidebar.title("🏥 MADA Navigation")
    
    # Navigation menu
    page = st.sidebar.selectbox(
        "Select Page",
        [
            "🏠 Dashboard Overview",
            "👤 Patient Management", 
            "🔬 AI Diagnosis",
            "📊 Analytics & Reports",
            "⚙️ System Status",
            "🎯 Model Performance"
        ]
    )
    
    # System status indicator
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    
    # Mock system metrics
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("AI Models", "5", delta="Active")
    with col2:
        st.metric("Accuracy", "94.2%", delta="2.1%")
    
    st.sidebar.success("🟢 All Systems Operational")
    
    return page

def render_dashboard_overview():
    """Render main dashboard overview"""
    st.header("📋 Dashboard Overview")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Patients Today",
            value="23",
            delta="5 new"
        )
    
    with col2:
        st.metric(
            label="High-Risk Cases", 
            value="7",
            delta="3 urgent",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="AI Diagnoses Generated",
            value="18",
            delta="12 today"
        )
    
    with col4:
        st.metric(
            label="System Uptime",
            value="99.8%",
            delta="0.2%"
        )
    
    st.markdown("---")
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Patient Volume Trends")
        # Generate sample data for chart
        dates = pd.date_range(start='2024-08-01', end='2024-08-19', freq='D')
        patient_counts = [random.randint(15, 35) for _ in range(len(dates))]
        
        fig = px.line(
            x=dates,
            y=patient_counts,
            title="Daily Patient Visits"
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Patients"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Risk Distribution")
        risk_data = {
            'Risk Level': ['Low', 'Medium', 'High', 'Critical'],
            'Count': [45, 23, 12, 3]
        }
        
        fig = px.pie(
            values=risk_data['Count'],
            names=risk_data['Risk Level'],
            title="Patient Risk Level Distribution",
            color_discrete_map={
                'Low': '#28a745',
                'Medium': '#ffc107', 
                'High': '#fd7e14',
                'Critical': '#dc3545'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

def render_patient_management():
    """Render patient management interface"""
    st.header("👤 Patient Management")
    
    # Sample patient data
    patients = generate_sample_data()
    
    # Patient search/filter
    col1, col2, col3 = st.columns(3)
    with col1:
        search_name = st.text_input("🔍 Search by Name")
    with col2:
        filter_risk = st.selectbox("Filter by Risk Level", ["All", "Low", "Medium", "High", "Critical"])
    with col3:
        filter_status = st.selectbox("Filter by Status", ["All", "Pending Review", "Under Investigation", "Scheduled Follow-up"])
    
    st.markdown("---")
    
    # Display patients
    for i, patient in enumerate(patients):
        with st.expander(f"Patient: {patient['name']} ({patient['patient_id']})", expanded=(i==0)):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Age:** {patient['age']}")
                st.markdown(f"**Gender:** {patient['gender']}")
                st.markdown(f"**Last Visit:** {patient['last_visit']}")
                
                # Risk level with color coding
                risk_color = {
                    "Low": "🟢",
                    "Medium": "🟡", 
                    "High": "🟠",
                    "Critical": "🔴"
                }
                st.markdown(f"**Risk Level:** {risk_color.get(patient['risk_level'], '⚪')} {patient['risk_level']}")
            
            with col2:
                st.markdown(f"**Chief Complaint:**")
                st.write(patient['chief_complaint'])
                st.markdown(f"**Status:** {patient['status']}")
                
                if patient['urgent_flags']:
                    st.markdown("**⚠️ Urgent Flags:**")
                    for flag in patient['urgent_flags']:
                        st.warning(f"• {flag.replace('_', ' ').title()}")
            
            with col3:
                st.markdown("**Vital Signs:**")
                vitals = patient['vitals']
                st.markdown(f"• **BP:** {vitals['bp']} mmHg")
                st.markdown(f"• **Heart Rate:** {vitals['heart_rate']} bpm")
                st.markdown(f"• **Temperature:** {vitals['temperature']}°F")
                st.markdown(f"• **Weight:** {vitals['weight']} kg")
                
                # Action buttons
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button(f"View Details {i+1}", key=f"view_{i}"):
                        st.info(f"Opening detailed view for {patient['name']}")
                with col_btn2:
                    if st.button(f"Generate Report {i+1}", key=f"report_{i}"):
                        st.success(f"Report generated for {patient['name']}")

def render_ai_diagnosis():
    """Render AI diagnosis interface"""
    st.header("🔬 AI Diagnosis Interface")
    
    st.markdown("""
    <div class="info-box">
        <strong>🤖 AI-Powered Diagnosis System</strong><br>
        Enter patient symptoms and medical history to generate AI-assisted diagnostic suggestions.
    </div>
    """, unsafe_allow_html=True)
    
    # Input form
    with st.form("diagnosis_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Information")
            age = st.number_input("Age", min_value=1, max_value=120, value=45)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
            st.subheader("Vital Signs")
            systolic_bp = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
            diastolic_bp = st.number_input("Diastolic BP", min_value=40, max_value=130, value=80)
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=72)
            temperature = st.number_input("Temperature (°F)", min_value=95.0, max_value=110.0, value=98.6)
        
        with col2:
            st.subheader("Symptoms & Complaints")
            chief_complaint = st.text_area("Chief Complaint", 
                                         placeholder="Describe the main symptoms...",
                                         height=100)
            
            symptoms = st.multiselect(
                "Select Symptoms",
                ["Fever", "Cough", "Shortness of breath", "Chest pain", "Headache", 
                 "Nausea", "Vomiting", "Diarrhea", "Fatigue", "Dizziness", "Palpitations"]
            )
            
            duration = st.selectbox("Symptom Duration", 
                                  ["Less than 24 hours", "1-3 days", "1 week", "2-4 weeks", "More than 1 month"])
            
            pain_scale = st.slider("Pain Level (0-10)", 0, 10, 0)
        
        # Submit button
        submitted = st.form_submit_button("🔍 Generate AI Diagnosis", use_container_width=True)
        
        if submitted:
            # Simulate AI processing
            with st.spinner('🤖 AI is analyzing patient data...'):
                time.sleep(2)  # Simulate processing time
            
            st.success("✅ AI Analysis Complete!")
            
            # Enhanced diagnosis results based on inputs
            diagnosis_result = generate_smart_diagnosis({
                'age': age,
                'gender': gender,
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'heart_rate': heart_rate,
                'temperature': temperature,
                'chief_complaint': chief_complaint,
                'symptoms': symptoms,
                'duration': duration,
                'pain_scale': pain_scale
            })
            
            # Display results
            st.subheader("🎯 AI Diagnosis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔍 Differential Diagnosis")
                for i, diag in enumerate(diagnosis_result['diagnoses']):
                    urgency_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                    confidence_bar = "█" * int(diag['confidence'] * 10) + "░" * (10 - int(diag['confidence'] * 10))
                    
                    st.markdown(f"""
                    <div class="diagnosis-card" style="border-left: 4px solid {urgency_color[diag['urgency']].replace('🟢', 'green').replace('🟡', 'orange').replace('🔴', 'red')}">
                        <strong>#{i+1}: {diag['condition']}</strong><br>
                        <small>Confidence: {diag['confidence']:.1%}</small><br>
                        <div style="font-family: monospace; font-size: 12px; color: #666;">|{confidence_bar}|</div>
                        <small>Urgency: {urgency_color[diag['urgency']]} {diag['urgency']}</small>
                        {f'<br><small>🔍 <em>{diag["reasoning"]}</em></small>' if diag.get('reasoning') else ''}
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📋 Clinical Recommendations")
                recommendations = diagnosis_result['recommendations']
                
                if recommendations.get('immediate'):
                    st.markdown("**🚨 Immediate Actions:**")
                    for action in recommendations['immediate']:
                        st.markdown(f"• {action}")
                
                if recommendations.get('follow_up'):
                    st.markdown("**📅 Follow-up Tests:**")
                    for test in recommendations['follow_up']:
                        st.markdown(f"• {test}")
                
                if recommendations.get('lifestyle'):
                    st.markdown("**🏃‍♂️ Lifestyle Recommendations:**")
                    for lifestyle in recommendations['lifestyle']:
                        st.markdown(f"• {lifestyle}")
                
                if recommendations.get('monitoring'):
                    st.markdown("**👁️ Monitoring:**")
                    for monitor in recommendations['monitoring']:
                        st.markdown(f"• {monitor}")
            
            # Risk Assessment
            if diagnosis_result['risk_assessment']:
                st.markdown("---")
                st.subheader("⚠️ Risk Assessment")
                
                risk = diagnosis_result['risk_assessment']
                risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴", "Critical": "⚫"}
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Overall Risk", f"{risk_color[risk['level']]} {risk['level']}", 
                             delta=f"Score: {risk['score']:.2f}")
                
                with col2:
                    st.metric("Urgency Level", risk['urgency'], delta=risk['timeframe'])
                
                with col3:
                    if risk['factors']:
                        st.markdown("**Risk Factors:**")
                        for factor in risk['factors']:
                            st.markdown(f"• {factor}")
            
            # Add disclaimer
            st.markdown("---")
            st.warning("⚠️ **Medical Disclaimer:** This AI analysis is for educational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider.")

def render_analytics_reports():
    """Render analytics and reports page"""
    st.header("📊 Analytics & Reports")
    
    # Time period selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    st.markdown("---")
    
    # Analytics sections
    tab1, tab2, tab3 = st.tabs(["📈 Performance Metrics", "🏥 Clinical Insights", "📋 System Reports"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # AI Model Performance
            st.subheader("🎯 AI Model Performance")
            performance_data = {
                'Model': ['XGBoost', 'LightGBM', 'Random Forest', 'Neural Network'],
                'Accuracy': [0.94, 0.92, 0.89, 0.91],
                'Precision': [0.93, 0.91, 0.88, 0.90],
                'Recall': [0.95, 0.93, 0.90, 0.92]
            }
            
            fig = px.bar(performance_data, x='Model', y=['Accuracy', 'Precision', 'Recall'],
                        title="Model Performance Comparison", barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Diagnosis Distribution
            st.subheader("🔬 Diagnosis Distribution")
            diagnosis_counts = {
                'Hypertension': 45,
                'Diabetes': 32,
                'Anxiety': 28,
                'Cardiovascular': 23,
                'Respiratory': 18
            }
            
            fig = px.bar(x=list(diagnosis_counts.keys()), y=list(diagnosis_counts.values()),
                        title="Top Diagnosed Conditions")
            fig.update_layout(xaxis_title="Condition", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🏥 Clinical Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Average Patient Age", "52.3", delta="1.2 years")
            st.metric("Male/Female Ratio", "48/52", delta="2% F increase")
            st.metric("Critical Cases", "3", delta="1 this week")
        
        with col2:
            st.metric("Average Diagnosis Time", "12.5 min", delta="-2.3 min")
            st.metric("Follow-up Compliance", "87%", delta="3%")
            st.metric("Patient Satisfaction", "4.6/5", delta="0.1")
        
        # Risk factors analysis
        st.subheader("⚠️ Risk Factors Analysis")
        risk_factors = pd.DataFrame({
            'Risk Factor': ['Smoking', 'Obesity', 'Family History', 'Age >50', 'High BP'],
            'Prevalence': [23, 45, 67, 78, 34],
            'Impact Score': [8.5, 7.2, 6.8, 5.9, 9.1]
        })
        
        fig = px.scatter(risk_factors, x='Prevalence', y='Impact Score', size='Prevalence',
                        hover_name='Risk Factor', title="Risk Factor Analysis")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📋 System Reports")
        
        # Generate report buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Generate Performance Report"):
                st.success("Performance report generated and saved to downloads/")
        
        with col2:
            if st.button("🏥 Generate Clinical Summary"):
                st.success("Clinical summary report generated and saved to downloads/")
        
        with col3:
            if st.button("📈 Generate Trend Analysis"):
                st.success("Trend analysis report generated and saved to downloads/")
        
        # Recent reports table
        st.subheader("Recent Reports")
        reports_data = {
            'Report Name': ['Weekly Performance', 'Monthly Clinical Summary', 'Quarterly Analysis'],
            'Generated': ['2024-08-19', '2024-08-15', '2024-08-01'],
            'Status': ['Complete', 'Complete', 'Complete'],
            'Download': ['📥 Download', '📥 Download', '📥 Download']
        }
        
        reports_df = pd.DataFrame(reports_data)
        st.dataframe(reports_df, use_container_width=True)

def render_system_status():
    """Render system status page"""
    st.header("⚙️ System Status")
    
    # System health indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("System Uptime", "99.8%", delta="0.1%")
    with col2:
        st.metric("Active Sessions", "142", delta="23")
    with col3:
        st.metric("Response Time", "1.2s", delta="-0.3s")
    
    st.markdown("---")
    
    # Service status
    st.subheader("🔧 Service Status")
    
    services = [
        {"name": "Web Dashboard", "status": "Operational", "uptime": "99.9%"},
        {"name": "AI Diagnosis Engine", "status": "Operational", "uptime": "99.7%"},
        {"name": "Database", "status": "Operational", "uptime": "100%"},
        {"name": "Authentication Service", "status": "Operational", "uptime": "99.8%"},
        {"name": "Notification System", "status": "Maintenance", "uptime": "97.2%"}
    ]
    
    for service in services:
        status_emoji = "🟢" if service["status"] == "Operational" else "🟡"
        st.markdown(f"{status_emoji} **{service['name']}** - {service['status']} ({service['uptime']} uptime)")
    
    st.markdown("---")
    
    # System logs (mock)
    st.subheader("📋 Recent System Logs")
    
    logs = [
        {"time": "2024-08-19 14:30:15", "level": "INFO", "message": "AI model prediction completed successfully"},
        {"time": "2024-08-19 14:28:42", "level": "INFO", "message": "New patient registered: DEMO_004"},
        {"time": "2024-08-19 14:25:33", "level": "WARN", "message": "High database connection pool usage (85%)"},
        {"time": "2024-08-19 14:20:11", "level": "INFO", "message": "Scheduled model retraining initiated"},
        {"time": "2024-08-19 14:18:55", "level": "ERROR", "message": "Notification service temporarily unavailable"}
    ]
    
    logs_df = pd.DataFrame(logs)
    st.dataframe(logs_df, use_container_width=True)

def render_model_performance():
    """Render model performance page"""
    st.header("🎯 Model Performance")
    
    # Model selector
    selected_model = st.selectbox("Select Model", ["XGBoost Classifier", "LightGBM", "Random Forest", "Neural Network"])
    
    st.markdown(f"### Performance Metrics for {selected_model}")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "94.2%", delta="2.1%")
    with col2:
        st.metric("Precision", "93.8%", delta="1.9%")
    with col3:
        st.metric("Recall", "94.6%", delta="2.3%")
    with col4:
        st.metric("F1-Score", "94.2%", delta="2.1%")
    
    st.markdown("---")
    
    # Performance over time
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Performance Over Time")
        dates = pd.date_range(start='2024-07-01', end='2024-08-19', freq='W')
        accuracy_scores = [0.92 + random.uniform(-0.02, 0.03) for _ in range(len(dates))]
        
        fig = px.line(x=dates, y=accuracy_scores, title="Model Accuracy Trend")
        fig.update_layout(xaxis_title="Date", yaxis_title="Accuracy")
        fig.update_yaxis(range=[0.88, 0.96])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Confusion Matrix")
        # Mock confusion matrix data
        confusion_data = np.array([[85, 5, 3], [4, 78, 2], [2, 3, 89]])
        fig = px.imshow(confusion_data, 
                       labels=dict(x="Predicted", y="Actual"),
                       x=['Low Risk', 'Medium Risk', 'High Risk'],
                       y=['Low Risk', 'Medium Risk', 'High Risk'],
                       title="Confusion Matrix",
                       color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("🔍 Feature Importance")
    features = ['Age', 'Blood Pressure', 'Heart Rate', 'BMI', 'Family History', 'Symptoms', 'Lab Results']
    importance = [0.18, 0.22, 0.15, 0.14, 0.12, 0.10, 0.09]
    
    fig = px.bar(x=features, y=importance, title="Feature Importance in Model Predictions")
    fig.update_layout(xaxis_title="Features", yaxis_title="Importance Score")
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function"""
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.sample_patients = generate_sample_data()
    
    # Render header
    render_header()
    
    # Render sidebar and get selected page
    selected_page = render_sidebar()
    
    # Route to appropriate page
    if selected_page == "🏠 Dashboard Overview":
        render_dashboard_overview()
    elif selected_page == "👤 Patient Management":
        render_patient_management()
    elif selected_page == "🔬 AI Diagnosis":
        render_ai_diagnosis()
    elif selected_page == "📊 Analytics & Reports":
        render_analytics_reports()
    elif selected_page == "⚙️ System Status":
        render_system_status()
    elif selected_page == "🎯 Model Performance":
        render_model_performance()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <small>MADA v1.0 | Medical AI Diagnosis Assistant | Built with Streamlit</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
