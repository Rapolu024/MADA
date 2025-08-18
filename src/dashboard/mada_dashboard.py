"""
MADA Dashboard
Streamlit-based web interface for medical professionals
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
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import MADA components
try:
    from diagnosis_agent.diagnosis_agent import DiagnosisAgent
    from patient_intake.models import PatientIntakeForm
    from data_storage.database_manager import DatabaseManager
except ImportError:
    st.error("MADA components not found. Please ensure the system is properly installed.")
    st.stop()

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
</style>
""", unsafe_allow_html=True)

class MADADashboard:
    """Main dashboard class for MADA system"""
    
    def __init__(self):
        # Initialize MADA components
        if 'diagnosis_agent' not in st.session_state:
            with st.spinner('Initializing MADA AI System...'):
                st.session_state.diagnosis_agent = DiagnosisAgent(load_models=False)
        
        # Initialize sample data if not exists
        if 'sample_patients' not in st.session_state:
            st.session_state.sample_patients = self._generate_sample_data()
        
        # Initialize diagnosis history
        if 'diagnosis_history' not in st.session_state:
            st.session_state.diagnosis_history = []
    
    def _generate_sample_data(self) -> List[Dict]:
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
                "status": "Pending Review"
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
                "status": "Under Investigation"
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
                "status": "Scheduled Follow-up"
            }
        ]
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown("""
        <div class="main-header">
            <h1 style="color: white; margin: 0;">🏥 MADA - Medical AI Diagnosis Assistant</h1>
            <p style="color: #e0e0e0; margin: 0;">AI-Powered Early Disease Detection System</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
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
    
    def render_dashboard_overview(self):
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
                label="Average Confidence",
                value="87.3%",
                delta="2.1%"
            )
        
        # Patient queue and alerts
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🚨 Priority Patient Queue")
            
            # Create patient queue table
            queue_df = pd.DataFrame(st.session_state.sample_patients)
            
            # Color-code by risk level
            def highlight_risk(row):
                if row['risk_level'] == 'High':
                    return ['background-color: #ffe6e6'] * len(row)
                elif row['risk_level'] == 'Medium':
                    return ['background-color: #fff3cd'] * len(row)
                else:
                    return ['background-color: #e6ffe6'] * len(row)
            
            styled_df = queue_df.style.apply(highlight_risk, axis=1)
            st.dataframe(styled_df, use_container_width=True)
        
        with col2:
            st.subheader("⚠️ Active Alerts")
            
            # Mock alerts
            alerts = [
                {"type": "🔴 Critical", "message": "Patient DEMO_001: Possible diabetes detected"},
                {"type": "🟠 Warning", "message": "Patient DEMO_002: Cardiovascular risk factors"},
                {"type": "🟡 Info", "message": "Model retaining completed successfully"}
            ]
            
            for alert in alerts:
                st.markdown(f"""
                <div class="urgent-alert">
                    <strong>{alert['type']}</strong><br>
                    {alert['message']}
                </div>
                """, unsafe_allow_html=True)
        
        # Recent trends chart
        st.subheader("📈 Recent Diagnosis Trends")
        
        # Generate mock trend data
        dates = pd.date_range(start='2024-08-01', end='2024-08-18', freq='D')
        diagnoses = np.random.poisson(15, len(dates))
        high_risk = np.random.poisson(3, len(dates))
        
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(
            x=dates, y=diagnoses,
            mode='lines+markers',
            name='Total Diagnoses',
            line=dict(color='#2a5298')
        ))
        trend_fig.add_trace(go.Scatter(
            x=dates, y=high_risk,
            mode='lines+markers',
            name='High-Risk Cases',
            line=dict(color='#dc3545')
        ))
        
        trend_fig.update_layout(
            title="Daily Diagnosis Volume",
            xaxis_title="Date",
            yaxis_title="Number of Cases",
            height=400
        )
        
        st.plotly_chart(trend_fig, use_container_width=True)
    
    def render_patient_management(self):
        """Render patient management interface"""
        st.header("👤 Patient Management")
        
        tab1, tab2, tab3 = st.tabs(["Patient Search", "New Patient Intake", "Patient History"])
        
        with tab1:
            st.subheader("🔍 Search Patients")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                search_name = st.text_input("Patient Name")
            with col2:
                search_id = st.text_input("Patient ID")
            with col3:
                search_date = st.date_input("Visit Date")
            
            if st.button("Search Patients"):
                # Mock search results
                results_df = pd.DataFrame(st.session_state.sample_patients)
                st.dataframe(results_df, use_container_width=True)
        
        with tab2:
            st.subheader("➕ New Patient Intake")
            self.render_patient_intake_form()
        
        with tab3:
            st.subheader("📋 Patient History")
            
            selected_patient = st.selectbox(
                "Select Patient",
                ["DEMO_001 - John Smith", "DEMO_002 - Sarah Johnson", "DEMO_003 - Michael Brown"]
            )
            
            if selected_patient:
                # Mock patient history
                history_data = {
                    "Date": ["2024-08-15", "2024-07-20", "2024-06-15"],
                    "Diagnosis": ["Diabetes Type 2 (Suspected)", "Prediabetes", "Regular Checkup"],
                    "Confidence": ["89%", "76%", "N/A"],
                    "Status": ["Under Treatment", "Monitoring", "Resolved"]
                }
                
                history_df = pd.DataFrame(history_data)
                st.dataframe(history_df, use_container_width=True)
    
    def render_patient_intake_form(self):
        """Render new patient intake form"""
        st.markdown("### Patient Information")
        
        with st.form("patient_intake_form"):
            # Demographics
            col1, col2 = st.columns(2)
            with col1:
                first_name = st.text_input("First Name *")
                age = st.number_input("Age", min_value=0, max_value=120, value=30)
                height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
            
            with col2:
                last_name = st.text_input("Last Name *")
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                weight = st.number_input("Weight (kg)", min_value=20, max_value=500, value=70)
            
            # Vital Signs
            st.markdown("### Vital Signs")
            col1, col2, col3 = st.columns(3)
            with col1:
                systolic_bp = st.number_input("Systolic BP", min_value=60, max_value=250, value=120)
                heart_rate = st.number_input("Heart Rate", min_value=30, max_value=200, value=72)
            
            with col2:
                diastolic_bp = st.number_input("Diastolic BP", min_value=30, max_value=150, value=80)
                temperature = st.number_input("Temperature (°F)", min_value=95.0, max_value=110.0, value=98.6)
            
            with col3:
                respiratory_rate = st.number_input("Respiratory Rate", min_value=8, max_value=40, value=16)
                oxygen_sat = st.number_input("Oxygen Saturation (%)", min_value=70, max_value=100, value=98)
            
            # Chief Complaint
            st.markdown("### Chief Complaint")
            primary_complaint = st.text_area("Primary Complaint *", placeholder="Describe the main symptoms or concerns...")
            pain_scale = st.slider("Pain Scale (0-10)", 0, 10, 0)
            
            # Medical History
            st.markdown("### Medical History")
            smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
            alcohol = st.selectbox("Alcohol Consumption", ["None", "Occasional", "Moderate", "Heavy"])
            allergies = st.text_area("Allergies", placeholder="List any known allergies...")
            medications = st.text_area("Current Medications", placeholder="List current medications...")
            
            # Submit button
            submitted = st.form_submit_button("Submit Patient Data", type="primary")
            
            if submitted:
                if first_name and last_name and primary_complaint:
                    # Create patient data structure
                    patient_data = {
                        "demographics": {
                            "first_name": first_name,
                            "last_name": last_name,
                            "age": age,
                            "gender": gender.lower()
                        },
                        "vital_signs": {
                            "systolic_bp": systolic_bp,
                            "diastolic_bp": diastolic_bp,
                            "heart_rate": heart_rate,
                            "temperature": temperature,
                            "respiratory_rate": respiratory_rate,
                            "oxygen_saturation": oxygen_sat,
                            "height": height,
                            "weight": weight
                        },
                        "chief_complaint": {
                            "primary_complaint": primary_complaint,
                            "pain_scale": pain_scale
                        },
                        "medical_history": {
                            "smoking_status": smoking_status.lower(),
                            "alcohol_consumption": alcohol.lower(),
                            "allergies": allergies.split(",") if allergies else [],
                            "current_medications": medications.split(",") if medications else []
                        }
                    }
                    
                    st.success(f"✅ Patient {first_name} {last_name} added successfully!")
                    st.json(patient_data)
                    
                    # Option to generate AI diagnosis
                    if st.button("🤖 Generate AI Diagnosis"):
                        with st.spinner("Generating AI diagnosis..."):
                            # This would call the actual diagnosis agent
                            st.info("AI diagnosis would be generated here with trained models.")
                
                else:
                    st.error("Please fill in all required fields (*)")
    
    def render_ai_diagnosis(self):
        """Render AI diagnosis interface"""
        st.header("🔬 AI Diagnosis Interface")
        
        # Patient selection
        st.subheader("Select Patient for Diagnosis")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_patient = st.selectbox(
                "Choose Patient",
                ["DEMO_001 - John Smith", "DEMO_002 - Sarah Johnson", "DEMO_003 - Michael Brown"]
            )
        
        with col2:
            include_reasoning = st.checkbox("Include Clinical Reasoning", value=True)
        
        if st.button("🤖 Generate AI Diagnosis", type="primary"):
            with st.spinner("Running AI diagnosis models..."):
                # Mock diagnosis results
                diagnosis_results = {
                    "patient_id": "DEMO_001",
                    "timestamp": datetime.now(),
                    "predictions": [
                        {"condition": "Diabetes Type 2", "probability": 0.87, "confidence_level": "high"},
                        {"condition": "Metabolic Syndrome", "probability": 0.64, "confidence_level": "medium"},
                        {"condition": "Hypertension", "probability": 0.43, "confidence_level": "low"}
                    ],
                    "risk_assessment": {
                        "overall_risk": "high",
                        "risk_score": 0.78,
                        "risk_factors": ["age_over_50", "elevated_glucose", "family_history"]
                    },
                    "recommendations": [
                        "Immediate endocrinology consultation recommended",
                        "HbA1c and glucose tolerance test needed",
                        "Lifestyle modification counseling"
                    ],
                    "urgent_flags": ["possible_diabetes"]
                }
                
                # Display results
                st.success("✅ AI Diagnosis Generated Successfully!")
                
                # Top predictions
                st.subheader("🎯 Top Diagnosis Predictions")
                
                for i, pred in enumerate(diagnosis_results["predictions"], 1):
                    confidence_color = {
                        "high": "#28a745",
                        "medium": "#ffc107", 
                        "low": "#6c757d"
                    }[pred["confidence_level"]]
                    
                    st.markdown(f"""
                    <div class="diagnosis-card">
                        <h4>{i}. {pred['condition']}</h4>
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <div style="background: {confidence_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em;">
                                {pred['confidence_level'].upper()}
                            </div>
                            <div style="font-weight: bold; font-size: 1.2em;">
                                {pred['probability']:.1%}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Risk assessment
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("⚠️ Risk Assessment")
                    risk = diagnosis_results["risk_assessment"]
                    
                    risk_color = {
                        "low": "#28a745",
                        "medium": "#ffc107",
                        "high": "#dc3545"
                    }[risk["overall_risk"]]
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: {risk_color}; color: white; border-radius: 8px;">
                        <h2 style="margin: 0;">{risk['overall_risk'].upper()} RISK</h2>
                        <p style="margin: 0; font-size: 1.2em;">Score: {risk['risk_score']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("**Risk Factors:**")
                    for factor in risk["risk_factors"]:
                        st.write(f"• {factor.replace('_', ' ').title()}")
                
                with col2:
                    st.subheader("💡 Clinical Recommendations")
                    for i, rec in enumerate(diagnosis_results["recommendations"], 1):
                        st.write(f"{i}. {rec}")
                
                # Urgent flags
                if diagnosis_results["urgent_flags"]:
                    st.subheader("🚨 Urgent Flags")
                    for flag in diagnosis_results["urgent_flags"]:
                        st.error(f"⚠️ {flag.replace('_', ' ').title()}")
    
    def render_analytics(self):
        """Render analytics and reports"""
        st.header("📊 Analytics & Reports")
        
        tab1, tab2, tab3 = st.tabs(["📈 Performance Metrics", "🏥 Clinic Statistics", "📋 Reports"])
        
        with tab1:
            st.subheader("AI Model Performance")
            
            # Model performance metrics
            models = ["XGBoost", "LightGBM", "Random Forest", "Neural Network", "Ensemble"]
            accuracy = [0.89, 0.87, 0.85, 0.91, 0.94]
            precision = [0.88, 0.86, 0.84, 0.90, 0.93]
            recall = [0.87, 0.85, 0.83, 0.89, 0.92]
            
            performance_df = pd.DataFrame({
                "Model": models,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall
            })
            
            # Performance chart
            fig = px.bar(
                performance_df.melt(id_vars=["Model"], var_name="Metric", value_name="Score"),
                x="Model",
                y="Score", 
                color="Metric",
                title="Model Performance Comparison",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance table
            st.dataframe(performance_df, use_container_width=True)
        
        with tab2:
            st.subheader("Clinic Statistics")
            
            # Disease distribution
            diseases = ["Diabetes", "Hypertension", "Heart Disease", "Respiratory Issues", "Other"]
            counts = [25, 18, 12, 8, 15]
            
            fig_pie = px.pie(
                values=counts,
                names=diseases,
                title="Disease Distribution (Last 30 Days)"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Monthly trends
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
            patients = [120, 135, 142, 128, 156, 167, 189, 203]
            
            fig_trend = px.line(
                x=months,
                y=patients,
                title="Monthly Patient Volume",
                labels={"x": "Month", "y": "Number of Patients"}
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with tab3:
            st.subheader("Generate Reports")
            
            col1, col2 = st.columns(2)
            with col1:
                report_type = st.selectbox(
                    "Report Type",
                    ["Daily Summary", "Weekly Analysis", "Monthly Report", "Performance Review"]
                )
            
            with col2:
                date_range = st.date_input(
                    "Date Range",
                    value=[date.today() - timedelta(days=7), date.today()]
                )
            
            if st.button("Generate Report"):
                st.success("✅ Report generated successfully!")
                
                # Mock report data
                report_data = {
                    "report_type": report_type,
                    "date_range": f"{date_range[0]} to {date_range[1]}",
                    "total_patients": 45,
                    "high_risk_cases": 12,
                    "ai_diagnoses": 38,
                    "average_confidence": 0.87
                }
                
                st.json(report_data)
    
    def render_system_status(self):
        """Render system status and health monitoring"""
        st.header("⚙️ System Status")
        
        # System health indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🤖 AI Models Status",
                value="5/5 Active",
                delta="All Operational"
            )
        
        with col2:
            st.metric(
                label="💾 Database Health",
                value="Excellent",
                delta="0.2s avg response"
            )
        
        with col3:
            st.metric(
                label="🔒 Security Status", 
                value="Secure",
                delta="No threats"
            )
        
        with col4:
            st.metric(
                label="📡 API Status",
                value="Online",
                delta="99.9% uptime"
            )
        
        # Detailed system information
        st.subheader("📋 System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔧 System Configuration**
            - Python Version: 3.9.7
            - Framework: Streamlit 1.25.0
            - Database: PostgreSQL 13
            - ML Models: 5 Active
            - Last Backup: 2 hours ago
            """)
        
        with col2:
            st.markdown("""
            **📊 Performance Metrics**
            - CPU Usage: 23%
            - Memory Usage: 67%
            - Disk Space: 78% used
            - Network Status: Normal
            - Active Sessions: 12
            """)
        
        # System logs
        st.subheader("📝 Recent System Logs")
        
        logs = [
            {"timestamp": "2024-08-18 03:30:15", "level": "INFO", "message": "Patient DEMO_001 diagnosis completed successfully"},
            {"timestamp": "2024-08-18 03:28:42", "level": "INFO", "message": "Model ensemble prediction accuracy: 94.2%"},
            {"timestamp": "2024-08-18 03:25:18", "level": "WARNING", "message": "High memory usage detected (78%)"},
            {"timestamp": "2024-08-18 03:22:03", "level": "INFO", "message": "Database backup completed successfully"},
            {"timestamp": "2024-08-18 03:15:30", "level": "INFO", "message": "New patient intake processed: DEMO_003"}
        ]
        
        logs_df = pd.DataFrame(logs)
        st.dataframe(logs_df, use_container_width=True)
    
    def render_model_performance(self):
        """Render detailed model performance analysis"""
        st.header("🎯 Model Performance Analysis")
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model for Analysis",
            ["Ensemble (All Models)", "XGBoost", "LightGBM", "Random Forest", "Neural Network"]
        )
        
        # Performance over time
        st.subheader("📈 Performance Trends")
        
        dates = pd.date_range(start='2024-08-01', end='2024-08-18', freq='D')
        accuracy = 0.85 + 0.1 * np.random.random(len(dates))
        precision = 0.83 + 0.1 * np.random.random(len(dates))
        recall = 0.82 + 0.1 * np.random.random(len(dates))
        
        performance_fig = go.Figure()
        performance_fig.add_trace(go.Scatter(x=dates, y=accuracy, name='Accuracy', line=dict(color='#28a745')))
        performance_fig.add_trace(go.Scatter(x=dates, y=precision, name='Precision', line=dict(color='#007bff')))
        performance_fig.add_trace(go.Scatter(x=dates, y=recall, name='Recall', line=dict(color='#dc3545')))
        
        performance_fig.update_layout(
            title=f"{selected_model} - Performance Over Time",
            xaxis_title="Date",
            yaxis_title="Score",
            height=400
        )
        
        st.plotly_chart(performance_fig, use_container_width=True)
        
        # Confusion matrix visualization
        st.subheader("🎯 Confusion Matrix")
        
        # Mock confusion matrix data
        conf_matrix = np.array([[45, 3, 2], [4, 38, 1], [2, 1, 42]])
        diseases = ["Diabetes", "Hypertension", "Heart Disease"]
        
        fig_conf = px.imshow(
            conf_matrix,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix - Disease Classification",
            labels=dict(x="Predicted", y="Actual"),
            x=diseases,
            y=diseases
        )
        
        st.plotly_chart(fig_conf, use_container_width=True)
        
        # Feature importance
        st.subheader("🔍 Feature Importance")
        
        features = ["Age", "BMI", "Blood Pressure", "Glucose Level", "Family History", "Smoking Status", "Exercise Frequency"]
        importance = [0.23, 0.18, 0.16, 0.14, 0.12, 0.09, 0.08]
        
        importance_df = pd.DataFrame({"Feature": features, "Importance": importance})
        
        fig_importance = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation='h',
            title="Top Feature Importance",
            height=400
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
    
    def run(self):
        """Main dashboard runner"""
        self.render_header()
        
        # Sidebar navigation
        page = self.render_sidebar()
        
        # Route to selected page
        if page == "🏠 Dashboard Overview":
            self.render_dashboard_overview()
        elif page == "👤 Patient Management":
            self.render_patient_management()
        elif page == "🔬 AI Diagnosis":
            self.render_ai_diagnosis()
        elif page == "📊 Analytics & Reports":
            self.render_analytics()
        elif page == "⚙️ System Status":
            self.render_system_status()
        elif page == "🎯 Model Performance":
            self.render_model_performance()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666;">
            <p>MADA - Medical AI Early Diagnosis Assistant v1.0 | 
            <a href="#" style="color: #2a5298;">Documentation</a> | 
            <a href="#" style="color: #2a5298;">Support</a> | 
            <a href="#" style="color: #2a5298;">Privacy Policy</a></p>
        </div>
        """, unsafe_allow_html=True)

# Main app execution
if __name__ == "__main__":
    dashboard = MADADashboard()
    dashboard.run()
