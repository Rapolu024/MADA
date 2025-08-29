"""
ECG Analyzer
Main interface for ECG analysis combining preprocessing, modeling, and diagnosis
"""

import numpy as np
import pandas as pd
import cv2
from typing import Dict, List, Any, Tuple, Optional, Union
import logging
import os
from pathlib import Path
from datetime import datetime
import json

from .ecg_processor import ECGImageProcessor
from .ecg_models import ECGClassifier, ECGFeatureExtractor

logger = logging.getLogger(__name__)


class ECGAnalyzer:
    """
    Complete ECG analysis system combining preprocessing, feature extraction, and classification
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ECG analyzer with configuration"""
        
        # Default configuration
        default_config = {
            'image_size': (512, 512),
            'num_classes': 6,
            'model_type': 'cnn',
            'preprocessing': {
                'remove_grid': True,
                'enhance_contrast': True,
                'denoise': True
            },
            'batch_size': 32,
            'validation_split': 0.2
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Initialize components
        self.processor = ECGImageProcessor(target_size=self.config['image_size'])
        self.classifier = ECGClassifier(
            input_shape=(*self.config['image_size'], 1),
            num_classes=self.config['num_classes']
        )
        self.feature_extractor = ECGFeatureExtractor(
            input_shape=(*self.config['image_size'], 1)
        )
        
        # Analysis state
        self.is_trained = False
        self.training_history = None
        self.model_metrics = {}
        
    def load_ecg_dataset(self, image_dir: str, 
                        labels_file: Optional[str] = None,
                        label_column: str = 'condition',
                        filename_column: str = 'filename') -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load ECG dataset from directory and labels file
        """
        logger.info(f"Loading ECG dataset from {image_dir}")
        
        # Get all image files
        image_files = []
        for ext in self.processor.supported_formats:
            image_files.extend(Path(image_dir).glob(f"*{ext}"))
            image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No ECG images found in {image_dir}")
        
        # Load labels if provided
        labels_df = None
        if labels_file and os.path.exists(labels_file):
            labels_df = pd.read_csv(labels_file)
            logger.info(f"Loaded labels from {labels_file}")
        
        # Process images and extract labels
        images = []
        labels = []
        metadata = []
        
        for image_file in image_files:
            try:
                # Validate and preprocess image
                validation = self.processor.validate_ecg_image(str(image_file))
                
                if not validation['is_valid']:
                    logger.warning(f"Skipping invalid image: {image_file.name}")
                    continue
                
                # Preprocess image
                processed_image = self.processor.preprocess_ecg_image(
                    str(image_file),
                    **self.config['preprocessing']
                )
                
                images.append(processed_image)
                
                # Extract label
                if labels_df is not None:
                    # Find label in dataframe
                    label_row = labels_df[labels_df[filename_column] == image_file.name]
                    if not label_row.empty:
                        label = label_row.iloc[0][label_column]
                    else:
                        # Try without extension
                        stem_name = image_file.stem
                        label_row = labels_df[labels_df[filename_column] == stem_name]
                        if not label_row.empty:
                            label = label_row.iloc[0][label_column]
                        else:
                            logger.warning(f"No label found for {image_file.name}, using 'unknown'")
                            label = 'unknown'
                else:
                    # Extract label from directory structure or filename
                    label = self._extract_label_from_path(image_file)
                
                labels.append(label)
                
                # Store metadata
                metadata.append({
                    'filename': image_file.name,
                    'file_path': str(image_file),
                    'validation_score': validation['signal_quality'],
                    'has_grid': validation['has_grid']
                })
                
            except Exception as e:
                logger.error(f"Failed to process {image_file.name}: {e}")
                continue
        
        if not images:
            raise ValueError("No valid ECG images could be processed")
        
        # Convert to numpy arrays
        images_array = np.array(images)
        labels_array = np.array(labels)
        metadata_df = pd.DataFrame(metadata)
        
        logger.info(f"Loaded {len(images_array)} ECG images with {len(np.unique(labels_array))} unique labels")
        
        return images_array, labels_array, metadata_df
    
    def train_ecg_classifier(self, images: np.ndarray, labels: np.ndarray,
                           epochs: int = 50,
                           save_model_path: Optional[str] = None) -> Dict[str, Any]:
        """Train ECG classification model"""
        
        logger.info("Starting ECG classifier training")
        
        # Train the classifier
        training_summary = self.classifier.train(
            images=images,
            labels=labels,
            validation_split=self.config['validation_split'],
            epochs=epochs,
            batch_size=self.config['batch_size'],
            model_type=self.config['model_type'],
            save_path=save_model_path
        )
        
        # Update training state
        self.is_trained = True
        self.training_history = training_summary
        
        # Store model metrics
        self.model_metrics = {
            'accuracy': training_summary['final_val_accuracy'],
            'loss': training_summary['final_val_loss'],
            'training_samples': training_summary['num_samples'],
            'num_classes': training_summary['num_classes'],
            'model_type': training_summary['model_type']
        }
        
        logger.info(f"Training completed - Validation Accuracy: {training_summary['final_val_accuracy']:.3f}")
        
        return training_summary
    
    def analyze_ecg_image(self, image_path: str, 
                         include_features: bool = True,
                         include_preprocessing_steps: bool = False) -> Dict[str, Any]:
        """
        Comprehensive ECG image analysis
        """
        logger.info(f"Analyzing ECG image: {os.path.basename(image_path)}")
        
        analysis_result = {
            'image_path': image_path,
            'filename': os.path.basename(image_path),
            'timestamp': datetime.now().isoformat(),
            'validation': {},
            'preprocessing': {},
            'classification': {},
            'features': {},
            'recommendations': []
        }
        
        try:
            # Step 1: Validate image
            validation = self.processor.validate_ecg_image(image_path)
            analysis_result['validation'] = validation
            
            if not validation['is_valid']:
                analysis_result['status'] = 'invalid'
                analysis_result['error'] = f"Invalid ECG image: {', '.join(validation['issues'])}"
                return analysis_result
            
            # Step 2: Preprocess image
            processed_image = self.processor.preprocess_ecg_image(
                image_path,
                **self.config['preprocessing']
            )
            
            analysis_result['preprocessing'] = {
                'target_size': self.config['image_size'],
                'steps_applied': list(self.config['preprocessing'].keys()),
                'success': True
            }
            
            # Step 3: Extract features if requested
            if include_features:
                features = self.processor.extract_ecg_features(processed_image)
                analysis_result['features'] = features
            
            # Step 4: Classification (if model is trained)
            if self.is_trained:
                # Prepare image for prediction
                image_batch = np.expand_dims(processed_image, axis=0)
                
                # Get predictions
                predictions = self.classifier.predict(image_batch, return_probabilities=True)
                prediction = predictions[0]  # Single image prediction
                
                analysis_result['classification'] = {
                    'predicted_condition': prediction['predicted_condition'],
                    'confidence': prediction['confidence'],
                    'predicted_class': prediction['predicted_class']
                }
                
                # Add detailed predictions for multi-class
                if 'top_predictions' in prediction:
                    analysis_result['classification']['top_predictions'] = prediction['top_predictions']
                
                if 'all_probabilities' in prediction:
                    analysis_result['classification']['class_probabilities'] = prediction['all_probabilities']
                
                # Generate clinical recommendations
                analysis_result['recommendations'] = self._generate_clinical_recommendations(
                    prediction, validation, features if include_features else {}
                )
                
            else:
                analysis_result['classification'] = {
                    'status': 'model_not_trained',
                    'message': 'ECG classifier model has not been trained yet'
                }
            
            analysis_result['status'] = 'success'
            
        except Exception as e:
            analysis_result['status'] = 'error'
            analysis_result['error'] = str(e)
            logger.error(f"ECG analysis failed for {image_path}: {e}")
        
        return analysis_result
    
    def batch_analyze_ecgs(self, image_dir: str,
                          output_file: Optional[str] = None) -> pd.DataFrame:
        """Analyze multiple ECG images in batch"""
        
        logger.info(f"Starting batch ECG analysis for directory: {image_dir}")
        
        # Get all image files
        image_files = []
        for ext in self.processor.supported_formats:
            image_files.extend(Path(image_dir).glob(f"*{ext}"))
            image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
        
        # Analyze each image
        results = []
        for image_file in image_files:
            try:
                result = self.analyze_ecg_image(str(image_file))
                results.append(result)
                logger.info(f"Analyzed {image_file.name}: {result['status']}")
            except Exception as e:
                logger.error(f"Failed to analyze {image_file.name}: {e}")
        
        # Create results DataFrame
        results_df = self._create_results_dataframe(results)
        
        # Save results if requested
        if output_file:
            results_df.to_csv(output_file, index=False)
            logger.info(f"Batch analysis results saved to {output_file}")
        
        return results_df
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get trained model performance metrics"""
        
        if not self.is_trained:
            return {'status': 'model_not_trained'}
        
        return {
            'training_history': self.training_history,
            'metrics': self.model_metrics,
            'is_trained': self.is_trained
        }
    
    def save_model(self, filepath: str):
        """Save trained ECG model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        self.classifier.save_model(filepath)
        
        # Save configuration
        config_path = f"{filepath}_analyzer_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"ECG analyzer saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load pre-trained ECG model"""
        
        self.classifier.load_model(filepath)
        
        # Load configuration if exists
        config_path = f"{filepath}_analyzer_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
        
        self.is_trained = True
        logger.info(f"ECG analyzer loaded from {filepath}")
    
    def _extract_label_from_path(self, image_path: Path) -> str:
        """Extract label from image path or filename"""
        
        # Try to extract from parent directory name
        parent_name = image_path.parent.name.lower()
        
        # Common ECG condition patterns
        condition_patterns = {
            'normal': 'Normal',
            'mi': 'Myocardial Infarction',
            'myocardial': 'Myocardial Infarction',
            'lbbb': 'Left Bundle Branch Block',
            'rbbb': 'Right Bundle Branch Block',
            'pvc': 'Premature Ventricular Contraction',
            'af': 'Atrial Fibrillation',
            'atrial': 'Atrial Fibrillation',
            'arrhythmia': 'Atrial Fibrillation'
        }
        
        # Check filename
        filename_lower = image_path.stem.lower()
        
        # Search for patterns
        for pattern, condition in condition_patterns.items():
            if pattern in parent_name or pattern in filename_lower:
                return condition
        
        # Default to unknown
        return 'Unknown'
    
    def _generate_clinical_recommendations(self, prediction: Dict[str, Any],
                                         validation: Dict[str, Any],
                                         features: Dict[str, float]) -> List[str]:
        """Generate clinical recommendations based on analysis results"""
        
        recommendations = []
        
        condition = prediction['predicted_condition']
        confidence = prediction['confidence']
        
        # Base recommendations on predicted condition
        if condition == 'Myocardial Infarction':
            recommendations.extend([
                "⚠️ URGENT: Suspected myocardial infarction detected",
                "Immediate cardiology consultation recommended",
                "Consider emergency cardiac catheterization",
                "Monitor cardiac enzymes and vital signs"
            ])
        
        elif condition == 'Atrial Fibrillation':
            recommendations.extend([
                "Atrial fibrillation detected - assess stroke risk",
                "Consider anticoagulation therapy evaluation",
                "Monitor heart rate and rhythm",
                "Evaluate for underlying causes"
            ])
        
        elif condition in ['Left Bundle Branch Block', 'Right Bundle Branch Block']:
            recommendations.extend([
                f"{condition} detected - evaluate cardiac function",
                "Consider echocardiography",
                "Monitor for progression",
                "Assess for underlying heart disease"
            ])
        
        elif condition == 'Premature Ventricular Contraction':
            recommendations.extend([
                "PVCs detected - assess frequency and clinical context",
                "Consider 24-hour Holter monitoring",
                "Evaluate for structural heart disease",
                "Monitor symptoms and exercise tolerance"
            ])
        
        elif condition == 'Normal':
            recommendations.append("ECG appears normal - continue routine monitoring")
        
        # Add confidence-based recommendations
        if confidence < 0.7:
            recommendations.append("⚠️ Low diagnostic confidence - consider repeat ECG or expert review")
        
        # Add image quality recommendations
        if validation['signal_quality'] < 0.5:
            recommendations.append("Poor ECG quality detected - consider repeat with better lead placement")
        
        return recommendations
    
    def _create_results_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create organized DataFrame from batch analysis results"""
        
        records = []
        
        for result in results:
            record = {
                'filename': result['filename'],
                'status': result['status'],
                'signal_quality': result.get('validation', {}).get('signal_quality', 0),
                'has_grid': result.get('validation', {}).get('has_grid', False),
                'predicted_condition': result.get('classification', {}).get('predicted_condition', 'Unknown'),
                'confidence': result.get('classification', {}).get('confidence', 0),
                'timestamp': result['timestamp']
            }
            
            # Add top features if available
            features = result.get('features', {})
            if features:
                record.update({
                    'mean_intensity': features.get('mean_intensity', 0),
                    'contrast': features.get('contrast', 0),
                    'edge_density': features.get('edge_density', 0),
                    'snr_estimate': features.get('snr_estimate', 0)
                })
            
            # Add error information
            if result['status'] == 'error':
                record['error_message'] = result.get('error', 'Unknown error')
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def generate_analysis_report(self, results_df: pd.DataFrame,
                               output_path: Optional[str] = None) -> str:
        """Generate comprehensive analysis report"""
        
        report = []
        report.append("🏥 ECG Analysis Report")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Images Analyzed: {len(results_df)}")
        
        # Success statistics
        successful = len(results_df[results_df['status'] == 'success'])
        report.append(f"Successfully Processed: {successful} ({successful/len(results_df)*100:.1f}%)")
        
        # Condition distribution
        if successful > 0:
            report.append("\n📊 Detected Conditions:")
            condition_counts = results_df['predicted_condition'].value_counts()
            for condition, count in condition_counts.items():
                percentage = count / len(results_df) * 100
                report.append(f"  • {condition}: {count} cases ({percentage:.1f}%)")
        
        # Quality statistics
        report.append("\n📈 Image Quality Summary:")
        avg_quality = results_df['signal_quality'].mean()
        report.append(f"  • Average Signal Quality: {avg_quality:.3f}")
        poor_quality = len(results_df[results_df['signal_quality'] < 0.5])
        report.append(f"  • Poor Quality Images: {poor_quality} ({poor_quality/len(results_df)*100:.1f}%)")
        
        # Confidence analysis
        if successful > 0:
            avg_confidence = results_df[results_df['status'] == 'success']['confidence'].mean()
            low_confidence = len(results_df[results_df['confidence'] < 0.7])
            report.append(f"\n🎯 Prediction Confidence:")
            report.append(f"  • Average Confidence: {avg_confidence:.3f}")
            report.append(f"  • Low Confidence Cases: {low_confidence}")
        
        # Errors
        errors = len(results_df[results_df['status'] == 'error'])
        if errors > 0:
            report.append(f"\n❌ Processing Errors: {errors}")
        
        report_text = "\n".join(report)
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Analysis report saved to {output_path}")
        
        return report_text
