"""
Diagnosis Models
Implementation of multiple ML models for disease prediction
Includes XGBoost, LightGBM, Random Forest, and Neural Networks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Models
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

# Deep Learning (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logging.warning("TensorFlow not available. Deep learning models disabled.")

from config.config import MODEL_CONFIG, DISEASE_CATEGORIES, MODELS_DIR

logger = logging.getLogger(__name__)


class ModelPerformanceTracker:
    """Track and analyze model performance metrics"""
    
    def __init__(self):
        self.performance_history = []
    
    def add_performance(self, model_name: str, metrics: Dict[str, float], 
                       timestamp: datetime = None):
        """Add performance metrics for a model"""
        if timestamp is None:
            timestamp = datetime.now()
        
        performance_record = {
            'model_name': model_name,
            'timestamp': timestamp,
            'metrics': metrics
        }
        
        self.performance_history.append(performance_record)
    
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
        
        # XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(**MODEL_CONFIG['xgboost'])
        
        # LightGBM
        self.models['lightgbm'] = lgb.LGBMClassifier(
            **MODEL_CONFIG['lightgbm'],
            verbose=-1  # Suppress warnings
        )
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(**MODEL_CONFIG['random_forest'])
        
        # Neural Network
        self.models['mlp'] = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        
        # Initialize model weights (equal initially)
        for model_name in self.models.keys():
            self.model_weights[model_name] = 1.0 / len(self.models)
        
        # Deep Learning model (if available)
        if self.enable_deep_learning:
            self.models['deep_neural_network'] = None  # Will be built during training
            self.model_weights['deep_neural_network'] = 1.0 / (len(self.models))
        
        logger.info(f\"Initialized {len(self.models)} models: {list(self.models.keys())}\")\n    \n    def _build_deep_model(self, input_dim: int, num_classes: int) -> tf.keras.Model:\n        \"\"\"Build deep neural network model\"\"\"\n        if not self.enable_deep_learning:\n            return None\n        \n        model = Sequential([\n            Dense(512, activation='relu', input_shape=(input_dim,)),\n            BatchNormalization(),\n            Dropout(0.3),\n            \n            Dense(256, activation='relu'),\n            BatchNormalization(), \n            Dropout(0.3),\n            \n            Dense(128, activation='relu'),\n            BatchNormalization(),\n            Dropout(0.2),\n            \n            Dense(64, activation='relu'),\n            Dropout(0.2),\n            \n            Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')\n        ])\n        \n        optimizer = Adam(learning_rate=0.001)\n        loss = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'\n        \n        model.compile(\n            optimizer=optimizer,\n            loss=loss,\n            metrics=['accuracy']\n        )\n        \n        return model\n    \n    def fit(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> 'DiagnosisModelEnsemble':\n        \"\"\"Fit all models on training data\"\"\"\n        logger.info(f\"Training ensemble on {len(X)} samples with {X.shape[1]} features\")\n        \n        self.feature_names = X.columns.tolist()\n        self.class_names = sorted(y.unique().tolist())\n        \n        # Convert to numpy for compatibility\n        X_array = X.values\n        y_array = y.values\n        \n        # Encode labels for multi-class\n        from sklearn.preprocessing import LabelEncoder\n        label_encoder = LabelEncoder()\n        y_encoded = label_encoder.fit_transform(y_array)\n        self.label_encoder = label_encoder\n        \n        # Split for validation\n        from sklearn.model_selection import train_test_split\n        X_train, X_val, y_train, y_val = train_test_split(\n            X_array, y_encoded, test_size=validation_split, random_state=42, stratify=y_encoded\n        )\n        \n        # Train traditional ML models\n        for model_name, model in self.models.items():\n            if model_name == 'deep_neural_network':\n                continue\n            \n            logger.info(f\"Training {model_name}...\")\n            try:\n                # Train model\n                model.fit(X_train, y_train)\n                \n                # Evaluate on validation set\n                val_predictions = model.predict(X_val)\n                val_accuracy = accuracy_score(y_val, val_predictions)\n                \n                # Get detailed metrics\n                precision, recall, f1, _ = precision_recall_fscore_support(\n                    y_val, val_predictions, average='weighted'\n                )\n                \n                metrics = {\n                    'accuracy': val_accuracy,\n                    'precision': precision,\n                    'recall': recall,\n                    'f1_score': f1\n                }\n                \n                # Try to get AUC for binary/multiclass\n                try:\n                    if len(self.class_names) == 2:\n                        val_proba = model.predict_proba(X_val)[:, 1]\n                        metrics['auc'] = roc_auc_score(y_val, val_proba)\n                    else:\n                        val_proba = model.predict_proba(X_val)\n                        metrics['auc'] = roc_auc_score(y_val, val_proba, multi_class='ovr')\n                except:\n                    metrics['auc'] = 0.0\n                \n                self.performance_tracker.add_performance(model_name, metrics)\n                \n                logger.info(f\"{model_name} - Accuracy: {val_accuracy:.4f}, F1: {f1:.4f}\")\n                \n            except Exception as e:\n                logger.error(f\"Failed to train {model_name}: {e}\")\n                # Remove failed model\n                del self.models[model_name]\n                if model_name in self.model_weights:\n                    del self.model_weights[model_name]\n        \n        # Train deep learning model\n        if self.enable_deep_learning and 'deep_neural_network' in self.models:\n            logger.info(\"Training deep neural network...\")\n            try:\n                # Build model\n                deep_model = self._build_deep_model(X_train.shape[1], len(self.class_names))\n                \n                # Callbacks\n                callbacks = [\n                    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),\n                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)\n                ]\n                \n                # Train\n                history = deep_model.fit(\n                    X_train, y_train,\n                    validation_data=(X_val, y_val),\n                    epochs=200,\n                    batch_size=32,\n                    callbacks=callbacks,\n                    verbose=0\n                )\n                \n                # Evaluate\n                val_loss, val_accuracy = deep_model.evaluate(X_val, y_val, verbose=0)\n                val_predictions = np.argmax(deep_model.predict(X_val), axis=1)\n                \n                precision, recall, f1, _ = precision_recall_fscore_support(\n                    y_val, val_predictions, average='weighted'\n                )\n                \n                metrics = {\n                    'accuracy': val_accuracy,\n                    'precision': precision,\n                    'recall': recall,\n                    'f1_score': f1,\n                    'val_loss': val_loss\n                }\n                \n                self.models['deep_neural_network'] = deep_model\n                self.performance_tracker.add_performance('deep_neural_network', metrics)\n                \n                logger.info(f\"Deep Neural Network - Accuracy: {val_accuracy:.4f}, F1: {f1:.4f}\")\n                \n            except Exception as e:\n                logger.error(f\"Failed to train deep neural network: {e}\")\n                if 'deep_neural_network' in self.models:\n                    del self.models['deep_neural_network']\n                if 'deep_neural_network' in self.model_weights:\n                    del self.model_weights['deep_neural_network']\n        \n        # Update model weights based on performance\n        self._update_model_weights()\n        \n        self.is_fitted = True\n        logger.info(f\"Ensemble training completed. Active models: {list(self.models.keys())}\")\n        \n        return self\n    \n    def predict(self, X: pd.DataFrame) -> np.ndarray:\n        \"\"\"Make predictions using the ensemble\"\"\"\n        if not self.is_fitted:\n            raise ValueError(\"Models must be fitted before prediction\")\n        \n        # Ensure features match training data\n        X = self._align_features(X)\n        X_array = X.values\n        \n        # Get predictions from all models\n        predictions = {}\n        for model_name, model in self.models.items():\n            try:\n                if model_name == 'deep_neural_network' and self.enable_deep_learning:\n                    pred_proba = model.predict(X_array)\n                    if pred_proba.shape[1] > 1:\n                        pred = np.argmax(pred_proba, axis=1)\n                    else:\n                        pred = (pred_proba.flatten() > 0.5).astype(int)\n                else:\n                    pred = model.predict(X_array)\n                \n                predictions[model_name] = pred\n            except Exception as e:\n                logger.warning(f\"Prediction failed for {model_name}: {e}\")\n        \n        if not predictions:\n            raise RuntimeError(\"All models failed to make predictions\")\n        \n        # Weighted ensemble prediction\n        final_predictions = self._combine_predictions(predictions)\n        \n        return final_predictions\n    \n    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:\n        \"\"\"Get prediction probabilities from the ensemble\"\"\"\n        if not self.is_fitted:\n            raise ValueError(\"Models must be fitted before prediction\")\n        \n        X = self._align_features(X)\n        X_array = X.values\n        \n        # Get probabilities from all models\n        probabilities = {}\n        for model_name, model in self.models.items():\n            try:\n                if model_name == 'deep_neural_network' and self.enable_deep_learning:\n                    proba = model.predict(X_array)\n                else:\n                    proba = model.predict_proba(X_array)\n                \n                probabilities[model_name] = proba\n            except Exception as e:\n                logger.warning(f\"Probability prediction failed for {model_name}: {e}\")\n        \n        if not probabilities:\n            raise RuntimeError(\"All models failed to predict probabilities\")\n        \n        # Weighted ensemble probabilities\n        final_probabilities = self._combine_probabilities(probabilities)\n        \n        return final_probabilities\n    \n    def _combine_predictions(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:\n        \"\"\"Combine predictions from multiple models using weighted voting\"\"\"\n        if len(predictions) == 1:\n            return list(predictions.values())[0]\n        \n        # Stack predictions\n        pred_stack = np.column_stack(list(predictions.values()))\n        model_names = list(predictions.keys())\n        \n        # Get weights for active models\n        weights = np.array([self.model_weights.get(name, 0) for name in model_names])\n        weights = weights / weights.sum()  # Normalize\n        \n        # Weighted voting\n        final_predictions = []\n        for i in range(pred_stack.shape[0]):\n            # Count votes for each class\n            unique_classes, indices = np.unique(pred_stack[i], return_inverse=True)\n            vote_counts = np.bincount(indices, weights=weights)\n            \n            # Choose class with highest weighted vote\n            winner_idx = np.argmax(vote_counts)\n            final_predictions.append(unique_classes[winner_idx])\n        \n        return np.array(final_predictions)\n    \n    def _combine_probabilities(self, probabilities: Dict[str, np.ndarray]) -> np.ndarray:\n        \"\"\"Combine probabilities from multiple models using weighted averaging\"\"\"\n        if len(probabilities) == 1:\n            return list(probabilities.values())[0]\n        \n        model_names = list(probabilities.keys())\n        weights = np.array([self.model_weights.get(name, 0) for name in model_names])\n        weights = weights / weights.sum()  # Normalize\n        \n        # Weighted average of probabilities\n        weighted_proba = None\n        for i, (model_name, proba) in enumerate(probabilities.items()):\n            if weighted_proba is None:\n                weighted_proba = weights[i] * proba\n            else:\n                weighted_proba += weights[i] * proba\n        \n        return weighted_proba\n    \n    def _update_model_weights(self):\n        \"\"\"Update model weights based on performance\"\"\"\n        performance_summary = self.performance_tracker.get_performance_summary()\n        \n        total_f1 = 0\n        f1_scores = {}\n        \n        # Get F1 scores for weight calculation\n        for model_name in self.models.keys():\n            if model_name in performance_summary:\n                f1_score = performance_summary[model_name]['latest_performance'].get('f1_score', 0)\n                f1_scores[model_name] = f1_score\n                total_f1 += f1_score\n            else:\n                f1_scores[model_name] = 0\n        \n        # Update weights based on F1 scores\n        if total_f1 > 0:\n            for model_name in self.models.keys():\n                self.model_weights[model_name] = f1_scores[model_name] / total_f1\n        else:\n            # Equal weights if no performance data\n            for model_name in self.models.keys():\n                self.model_weights[model_name] = 1.0 / len(self.models)\n        \n        logger.info(f\"Updated model weights: {self.model_weights}\")\n    \n    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:\n        \"\"\"Ensure feature DataFrame matches training features\"\"\"\n        if self.feature_names is None:\n            return X\n        \n        # Add missing columns\n        for feature in self.feature_names:\n            if feature not in X.columns:\n                X[feature] = 0\n        \n        # Select and reorder columns\n        X = X[self.feature_names]\n        \n        return X\n    \n    def online_learning_update(self, X_new: pd.DataFrame, y_new: pd.Series):\n        \"\"\"Update models with new data (online learning)\"\"\"\n        if not self.is_fitted:\n            logger.warning(\"Cannot perform online learning on unfitted models\")\n            return\n        \n        logger.info(f\"Performing online learning with {len(X_new)} new samples\")\n        \n        X_new = self._align_features(X_new)\n        X_array = X_new.values\n        y_encoded = self.label_encoder.transform(y_new.values)\n        \n        # Update models that support incremental learning\n        updated_models = []\n        \n        for model_name, model in self.models.items():\n            if model_name == 'deep_neural_network':\n                # Retrain deep model with new data\n                try:\n                    model.fit(X_array, y_encoded, epochs=10, verbose=0)\n                    updated_models.append(model_name)\n                except Exception as e:\n                    logger.error(f\"Online learning failed for {model_name}: {e}\")\n            \n            # For tree-based models, we would need to retrain\n            # In a production system, consider using models with partial_fit\n        \n        logger.info(f\"Online learning completed for models: {updated_models}\")\n    \n    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:\n        \"\"\"Get feature importance from all models\"\"\"\n        importance_data = {}\n        \n        for model_name, model in self.models.items():\n            try:\n                if model_name == 'deep_neural_network':\n                    # Skip deep learning model (feature importance is complex)\n                    continue\n                \n                if hasattr(model, 'feature_importances_'):\n                    importances = model.feature_importances_\n                elif hasattr(model, 'coef_'):\n                    importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)\n                else:\n                    continue\n                \n                importance_dict = dict(zip(self.feature_names, importances))\n                importance_data[model_name] = importance_dict\n                \n            except Exception as e:\n                logger.error(f\"Failed to get feature importance for {model_name}: {e}\")\n        \n        return importance_data\n    \n    def save_models(self, filepath: str = None) -> bool:\n        \"\"\"Save all trained models\"\"\"\n        if not self.is_fitted:\n            logger.warning(\"Cannot save unfitted models\")\n            return False\n        \n        if filepath is None:\n            timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n            filepath = os.path.join(MODELS_DIR, f'diagnosis_ensemble_{timestamp}.pkl')\n        \n        try:\n            os.makedirs(os.path.dirname(filepath), exist_ok=True)\n            \n            # Save traditional ML models\n            model_data = {\n                'models': {name: model for name, model in self.models.items() \n                          if name != 'deep_neural_network'},\n                'model_weights': self.model_weights,\n                'feature_names': self.feature_names,\n                'class_names': self.class_names,\n                'label_encoder': self.label_encoder,\n                'performance_tracker': self.performance_tracker,\n                'created_at': datetime.now().isoformat()\n            }\n            \n            joblib.dump(model_data, filepath)\n            \n            # Save deep learning model separately\n            if 'deep_neural_network' in self.models and self.models['deep_neural_network'] is not None:\n                dl_filepath = filepath.replace('.pkl', '_deep_model.h5')\n                self.models['deep_neural_network'].save(dl_filepath)\n            \n            logger.info(f\"Models saved to {filepath}\")\n            return True\n            \n        except Exception as e:\n            logger.error(f\"Failed to save models: {e}\")\n            return False\n    \n    def load_models(self, filepath: str = None) -> bool:\n        \"\"\"Load trained models\"\"\"\n        if filepath is None:\n            # Find the most recent model file\n            model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('diagnosis_ensemble_') and f.endswith('.pkl')]\n            if not model_files:\n                logger.warning(\"No saved models found\")\n                return False\n            \n            latest_file = sorted(model_files)[-1]\n            filepath = os.path.join(MODELS_DIR, latest_file)\n        \n        try:\n            model_data = joblib.load(filepath)\n            \n            self.models = model_data['models']\n            self.model_weights = model_data['model_weights']\n            self.feature_names = model_data['feature_names']\n            self.class_names = model_data['class_names']\n            self.label_encoder = model_data['label_encoder']\n            self.performance_tracker = model_data['performance_tracker']\n            \n            # Load deep learning model if available\n            dl_filepath = filepath.replace('.pkl', '_deep_model.h5')\n            if os.path.exists(dl_filepath) and self.enable_deep_learning:\n                self.models['deep_neural_network'] = tf.keras.models.load_model(dl_filepath)\n            \n            self.is_fitted = True\n            logger.info(f\"Models loaded from {filepath}\")\n            return True\n            \n        except Exception as e:\n            logger.error(f\"Failed to load models: {e}\")\n            return False\n    \n    def get_model_summary(self) -> Dict[str, Any]:\n        \"\"\"Get summary of the ensemble\"\"\"\n        return {\n            'is_fitted': self.is_fitted,\n            'active_models': list(self.models.keys()),\n            'model_weights': self.model_weights,\n            'num_features': len(self.feature_names) if self.feature_names else 0,\n            'num_classes': len(self.class_names) if self.class_names else 0,\n            'class_names': self.class_names,\n            'performance_summary': self.performance_tracker.get_performance_summary(),\n            'best_model': self.performance_tracker.get_best_model()\n        }
