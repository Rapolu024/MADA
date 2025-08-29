"""
Diagnosis Models
Machine learning models for medical diagnosis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import joblib
import os

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# External ML libraries (optional)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

from config.config import MODEL_CONFIG, MODELS_DIR

logger = logging.getLogger(__name__)


class ModelPerformanceTracker:
    """Track model performance over time"""
    
    def __init__(self):
        self.performance_history = []
    
    def add_performance(self, model_name: str, metrics: Dict[str, float]):
        """Add performance metrics for a model"""
        record = {
            'model_name': model_name,
            'metrics': metrics,
            'timestamp': datetime.now()
        }
        self.performance_history.append(record)
    
    def get_best_model(self, metric: str = 'f1_score') -> Optional[str]:
        """Get the name of the best performing model"""
        if not self.performance_history:
            return None
        
        best_score = -1
        best_model = None
        
        for record in self.performance_history:
            if metric in record['metrics'] and record['metrics'][metric] > best_score:
                best_score = record['metrics'][metric]
                best_model = record['model_name']
        
        return best_model
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model performances"""
        if not self.performance_history:
            return {}
        
        models = {}
        for record in self.performance_history:
            model_name = record['model_name']
            if model_name not in models:
                models[model_name] = {'performances': [], 'latest': None}
            
            models[model_name]['performances'].append(record)
            if (models[model_name]['latest'] is None or 
                record['timestamp'] > models[model_name]['latest']['timestamp']):
                models[model_name]['latest'] = record
        
        summary = {}
        for model_name, data in models.items():
            latest_metrics = data['latest']['metrics']
            summary[model_name] = {
                'latest_performance': latest_metrics,
                'num_evaluations': len(data['performances']),
                'last_updated': data['latest']['timestamp'].isoformat()
            }
        
        return summary


class DiagnosisModelEnsemble:
    """
    Ensemble of multiple ML models for robust disease prediction
    Includes self-learning and online adaptation capabilities
    """
    
    def __init__(self, enable_deep_learning: bool = True):
        self.models = {}
        self.model_weights = {}
        self.is_fitted = False
        self.feature_names = None
        self.class_names = None
        self.performance_tracker = ModelPerformanceTracker()
        self.enable_deep_learning = enable_deep_learning and HAS_TENSORFLOW
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all ML models with optimized configurations"""
        
        # Random Forest (always available)
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Neural Network
        self.models['mlp'] = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        # XGBoost (if available)
        if HAS_XGBOOST:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
        # LightGBM (if available)
        if HAS_LIGHTGBM:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        
        # Initialize model weights (equal initially)
        for model_name in self.models.keys():
            self.model_weights[model_name] = 1.0 / len(self.models)
        
        # Deep Learning model (if available)
        if self.enable_deep_learning:
            self.models['deep_neural_network'] = None  # Will be built during training
            self.model_weights['deep_neural_network'] = 1.0 / (len(self.models))
        
        logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    def _build_deep_model(self, input_dim: int, num_classes: int):
        """Build deep neural network model"""
        if not self.enable_deep_learning or not HAS_TENSORFLOW:
            return None
        
        try:
            model = Sequential([
                Dense(256, activation='relu', input_shape=(input_dim,)),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(128, activation='relu'),
                BatchNormalization(), 
                Dropout(0.3),
                
                Dense(64, activation='relu'),
                Dropout(0.2),
                
                Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
            ])
            
            optimizer = Adam(learning_rate=0.001)
            loss = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
            
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=['accuracy']
            )
            
            return model
        except Exception as e:
            logger.warning(f"Could not build deep learning model: {e}")
            return None
    
    def fit(self, X, y, validation_split: float = 0.2):
        """Fit all models on training data"""
        logger.info(f"Training ensemble on {len(X)} samples")
        
        # Convert to numpy arrays if needed
        if hasattr(X, 'values'):
            X_array = X.values
            self.feature_names = X.columns.tolist()
        else:
            X_array = np.array(X)
            self.feature_names = [f'feature_{i}' for i in range(X_array.shape[1])]
        
        if hasattr(y, 'values'):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # Get unique classes
        self.class_names = sorted(np.unique(y_array).tolist())
        
        # Encode labels for multi-class
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_array)
        
        # Split for validation
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_array, y_encoded, test_size=validation_split, random_state=42, 
                stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None
            )
        except ValueError:
            # If stratification fails, don't stratify
            X_train, X_val, y_train, y_val = train_test_split(
                X_array, y_encoded, test_size=validation_split, random_state=42
            )
        
        # Train traditional ML models
        for model_name, model in self.models.items():
            if model_name == 'deep_neural_network':
                continue
            
            logger.info(f"Training {model_name}...")
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                val_predictions = model.predict(X_val)
                val_accuracy = accuracy_score(y_val, val_predictions)
                
                # Get detailed metrics
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_val, val_predictions, average='weighted', zero_division=0
                )
                
                metrics = {
                    'accuracy': val_accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                self.performance_tracker.add_performance(model_name, metrics)
                logger.info(f"{model_name} - Accuracy: {val_accuracy:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                # Remove failed model
                if model_name in self.models:
                    del self.models[model_name]
                if model_name in self.model_weights:
                    del self.model_weights[model_name]
        
        # Train deep learning model
        if self.enable_deep_learning and 'deep_neural_network' in self.models:
            logger.info("Training deep neural network...")
            try:
                # Build model
                deep_model = self._build_deep_model(X_train.shape[1], len(self.class_names))
                
                if deep_model is not None:
                    # Callbacks
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
                    ]
                    
                    # Train
                    history = deep_model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50,
                        batch_size=32,
                        callbacks=callbacks,
                        verbose=0
                    )
                    
                    # Evaluate
                    val_loss, val_accuracy = deep_model.evaluate(X_val, y_val, verbose=0)
                    val_predictions = np.argmax(deep_model.predict(X_val, verbose=0), axis=1)
                    
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_val, val_predictions, average='weighted', zero_division=0
                    )
                    
                    metrics = {
                        'accuracy': val_accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'val_loss': val_loss
                    }
                    
                    self.models['deep_neural_network'] = deep_model
                    self.performance_tracker.add_performance('deep_neural_network', metrics)
                    
                    logger.info(f"Deep Neural Network - Accuracy: {val_accuracy:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train deep neural network: {e}")
                if 'deep_neural_network' in self.models:
                    del self.models['deep_neural_network']
                if 'deep_neural_network' in self.model_weights:
                    del self.model_weights['deep_neural_network']
        
        # Update model weights based on performance
        self._update_model_weights()
        
        self.is_fitted = True
        logger.info(f"Ensemble training completed. Active models: {list(self.models.keys())}")
        
        return self
    
    def predict(self, X) -> np.ndarray:
        """Make predictions using the ensemble"""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
        
        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        # Get predictions from all models
        predictions = {}
        for model_name, model in self.models.items():
            try:
                if model_name == 'deep_neural_network' and model is not None:
                    pred_proba = model.predict(X_array, verbose=0)
                    if pred_proba.shape[1] > 1:
                        pred = np.argmax(pred_proba, axis=1)
                    else:
                        pred = (pred_proba.flatten() > 0.5).astype(int)
                else:
                    pred = model.predict(X_array)
                
                predictions[model_name] = pred
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")
        
        if not predictions:
            raise RuntimeError("All models failed to make predictions")
        
        # Simple majority voting for now
        if len(predictions) == 1:
            return list(predictions.values())[0]
        
        # Stack predictions and use majority vote
        pred_stack = np.column_stack(list(predictions.values()))
        final_predictions = []
        for i in range(pred_stack.shape[0]):
            # Get most common prediction
            unique, counts = np.unique(pred_stack[i], return_counts=True)
            final_predictions.append(unique[np.argmax(counts)])
        
        return np.array(final_predictions)
    
    def predict_proba(self, X) -> np.ndarray:
        """Get prediction probabilities from the ensemble"""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
        
        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        # Get probabilities from all models
        probabilities = {}
        for model_name, model in self.models.items():
            try:
                if model_name == 'deep_neural_network' and model is not None:
                    proba = model.predict(X_array, verbose=0)
                else:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_array)
                    else:
                        # For models without predict_proba, create dummy probabilities
                        predictions = model.predict(X_array)
                        proba = np.zeros((len(X_array), len(self.class_names)))
                        for i, pred in enumerate(predictions):
                            proba[i, pred] = 1.0
                
                probabilities[model_name] = proba
            except Exception as e:
                logger.warning(f"Probability prediction failed for {model_name}: {e}")
        
        if not probabilities:
            raise RuntimeError("All models failed to predict probabilities")
        
        # Average probabilities
        if len(probabilities) == 1:
            return list(probabilities.values())[0]
        
        # Simple averaging
        prob_arrays = list(probabilities.values())
        return np.mean(prob_arrays, axis=0)
    
    def _update_model_weights(self):
        """Update model weights based on performance"""
        performance_summary = self.performance_tracker.get_performance_summary()
        
        total_f1 = 0
        f1_scores = {}
        
        # Get F1 scores for weight calculation
        for model_name in self.models.keys():
            if model_name in performance_summary:
                f1_score = performance_summary[model_name]['latest_performance'].get('f1_score', 0)
                f1_scores[model_name] = f1_score
                total_f1 += f1_score
            else:
                f1_scores[model_name] = 0
        
        # Update weights based on F1 scores
        if total_f1 > 0:
            for model_name in self.models.keys():
                self.model_weights[model_name] = f1_scores[model_name] / total_f1
        else:
            # Equal weights if no performance data
            for model_name in self.models.keys():
                self.model_weights[model_name] = 1.0 / len(self.models)
        
        logger.info(f"Updated model weights: {self.model_weights}")
    
    def save_models(self, filepath: str = None) -> bool:
        """Save all trained models"""
        if not self.is_fitted:
            logger.warning("Cannot save unfitted models")
            return False
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(MODELS_DIR, f'diagnosis_ensemble_{timestamp}.pkl')
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save traditional ML models
            model_data = {
                'models': {name: model for name, model in self.models.items() 
                          if name != 'deep_neural_network'},
                'model_weights': self.model_weights,
                'feature_names': self.feature_names,
                'class_names': self.class_names,
                'label_encoder': self.label_encoder,
                'performance_tracker': self.performance_tracker,
                'created_at': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, filepath)
            
            # Save deep learning model separately
            if 'deep_neural_network' in self.models and self.models['deep_neural_network'] is not None:
                dl_filepath = filepath.replace('.pkl', '_deep_model.h5')
                self.models['deep_neural_network'].save(dl_filepath)
            
            logger.info(f"Models saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            return False
    
    def load_models(self, filepath: str = None) -> bool:
        """Load trained models"""
        if filepath is None:
            # Find the most recent model file
            try:
                model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('diagnosis_ensemble_') and f.endswith('.pkl')]
                if not model_files:
                    logger.warning("No saved models found")
                    return False
                
                latest_file = sorted(model_files)[-1]
                filepath = os.path.join(MODELS_DIR, latest_file)
            except FileNotFoundError:
                logger.warning("Models directory not found")
                return False
        
        try:
            model_data = joblib.load(filepath)
            
            self.models.update(model_data['models'])
            self.model_weights = model_data['model_weights']
            self.feature_names = model_data['feature_names']
            self.class_names = model_data['class_names']
            self.label_encoder = model_data['label_encoder']
            self.performance_tracker = model_data['performance_tracker']
            
            # Load deep learning model if available
            dl_filepath = filepath.replace('.pkl', '_deep_model.h5')
            if os.path.exists(dl_filepath) and self.enable_deep_learning:
                import tensorflow as tf
                self.models['deep_neural_network'] = tf.keras.models.load_model(dl_filepath)
            
            self.is_fitted = True
            logger.info(f"Models loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of the ensemble"""
        return {
            'is_fitted': self.is_fitted,
            'active_models': list(self.models.keys()),
            'model_weights': self.model_weights,
            'num_features': len(self.feature_names) if self.feature_names else 0,
            'num_classes': len(self.class_names) if self.class_names else 0,
            'class_names': self.class_names,
            'performance_summary': self.performance_tracker.get_performance_summary(),
            'best_model': self.performance_tracker.get_best_model()
        }
