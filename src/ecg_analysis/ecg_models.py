"""
ECG Classification Models
Deep learning models for ECG image analysis and cardiac condition detection
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Any, Tuple, Optional
import logging
import os
import joblib
from pathlib import Path
import cv2

logger = logging.getLogger(__name__)


class ECGFeatureExtractor:
    """
    Extract features from ECG images using CNN backbone
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (512, 512, 1)):
        """Initialize feature extractor"""
        self.input_shape = input_shape
        self.model = None
        
    def build_feature_extractor(self) -> keras.Model:
        """Build CNN feature extraction model"""
        
        inputs = keras.Input(shape=self.input_shape, name='ecg_input')
        
        # First convolutional block
        x = layers.Conv2D(32, (7, 7), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.1)(x)
        
        # Second convolutional block
        x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.1)(x)
        
        # Third convolutional block
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        # Fourth convolutional block
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        # Global average pooling for feature extraction
        features = layers.GlobalAveragePooling2D(name='ecg_features')(x)
        
        model = keras.Model(inputs=inputs, outputs=features, name='ecg_feature_extractor')
        return model
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """Extract features from ECG images"""
        if self.model is None:
            raise ValueError("Feature extractor model not built or loaded")
        
        # Preprocess images
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1)
        
        # Normalize
        images = images.astype(np.float32) / 255.0
        
        # Extract features
        features = self.model.predict(images)
        return features


class ECGClassifier:
    """
    ECG Classification model for cardiac condition detection
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (512, 512, 1),
                 num_classes: int = 5):
        """
        Initialize ECG classifier
        
        Common cardiac conditions:
        - Normal
        - Myocardial Infarction (MI)
        - Left Bundle Branch Block (LBBB) 
        - Right Bundle Branch Block (RBBB)
        - Premature Ventricular Contraction (PVC)
        - Atrial Fibrillation (AF)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None
        
        # Cardiac condition mapping
        self.condition_mapping = {
            0: 'Normal',
            1: 'Myocardial Infarction',
            2: 'Left Bundle Branch Block',
            3: 'Right Bundle Branch Block', 
            4: 'Premature Ventricular Contraction',
            5: 'Atrial Fibrillation'
        }
    
    def build_cnn_model(self) -> keras.Model:
        """Build CNN model for ECG classification"""
        
        inputs = keras.Input(shape=self.input_shape, name='ecg_input')
        
        # First convolutional block - capture fine details
        x = layers.Conv2D(32, (7, 7), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.1)(x)
        
        # Second block - detect patterns
        x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.1)(x)
        
        # Third block - complex pattern recognition
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        # Fourth block - high-level features
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        # Fifth block - abstract representations
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        
        # Dense layers for classification
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name='classification')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax', name='classification')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='ecg_classifier')
        return model
    
    def build_resnet_model(self) -> keras.Model:
        """Build ResNet-style model for ECG classification"""
        
        def residual_block(x, filters, kernel_size=3, stride=1):
            """ResNet residual block"""
            shortcut = x
            
            # First conv
            x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            
            # Second conv
            x = layers.Conv2D(filters, kernel_size, padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Adjust shortcut if needed
            if stride != 1 or shortcut.shape[-1] != filters:
                shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            
            # Add shortcut
            x = layers.Add()([x, shortcut])
            x = layers.ReLU()(x)
            
            return x
        
        inputs = keras.Input(shape=self.input_shape)
        
        # Initial conv
        x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Residual blocks
        x = residual_block(x, 64)
        x = residual_block(x, 64)
        
        x = residual_block(x, 128, stride=2)
        x = residual_block(x, 128)
        
        x = residual_block(x, 256, stride=2)
        x = residual_block(x, 256)
        
        x = residual_block(x, 512, stride=2)
        x = residual_block(x, 512)
        
        # Global pooling and classification
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        
        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='ecg_resnet')
        return model
    
    def compile_model(self, model: keras.Model, learning_rate: float = 0.001):
        """Compile model with appropriate loss and optimizer"""
        
        if self.num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy', 'precision', 'recall']
        else:
            loss = 'categorical_crossentropy'
            metrics = ['accuracy', 'top_2_accuracy']
        
        optimizer = optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def prepare_data(self, images: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        
        # Preprocess images
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1)
        
        # Normalize pixel values
        images = images.astype(np.float32) / 255.0
        
        # Encode labels
        if self.num_classes == 2:
            # Binary classification
            labels_encoded = self.label_encoder.fit_transform(labels).astype(np.float32)
        else:
            # Multi-class classification
            labels_encoded = self.label_encoder.fit_transform(labels)
            labels_encoded = keras.utils.to_categorical(labels_encoded, self.num_classes)
        
        return images, labels_encoded
    
    def train(self, images: np.ndarray, labels: np.ndarray,
              validation_split: float = 0.2,
              epochs: int = 50,
              batch_size: int = 32,
              model_type: str = 'cnn',
              save_path: Optional[str] = None) -> Dict[str, Any]:
        """Train ECG classification model"""
        
        logger.info(f"Training ECG classifier with {len(images)} samples")
        
        # Prepare data
        X, y = self.prepare_data(images, labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=labels
        )
        
        # Build model
        if model_type.lower() == 'resnet':
            self.model = self.build_resnet_model()
        else:
            self.model = self.build_cnn_model()
        
        # Compile model
        self.model = self.compile_model(self.model)
        
        logger.info(f"Model architecture: {self.model.name}")
        logger.info(f"Total parameters: {self.model.count_params():,}")
        
        # Setup callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        if save_path:
            callback_list.append(
                callbacks.ModelCheckpoint(
                    save_path,
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False
                )
            )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        # Evaluate final model
        train_loss, train_acc = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
        
        # Generate predictions for detailed evaluation
        y_pred = self.model.predict(X_val)
        
        if self.num_classes == 2:
            y_pred_classes = (y_pred > 0.5).astype(int)
            y_true_classes = y_val.astype(int)
        else:
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_val, axis=1)
        
        # Create training summary
        training_summary = {
            'model_type': model_type,
            'num_samples': len(images),
            'num_classes': self.num_classes,
            'epochs_trained': len(self.history.history['loss']),
            'final_train_loss': train_loss,
            'final_train_accuracy': train_acc,
            'final_val_loss': val_loss,
            'final_val_accuracy': val_acc,
            'classification_report': classification_report(
                y_true_classes, y_pred_classes, 
                target_names=[self.condition_mapping.get(i, f'Class_{i}') for i in range(self.num_classes)],
                output_dict=True
            ),
            'label_mapping': dict(zip(self.label_encoder.classes_, 
                                    range(len(self.label_encoder.classes_))))
        }
        
        logger.info(f"Training completed - Val Accuracy: {val_acc:.3f}")
        return training_summary
    
    def predict(self, images: np.ndarray, return_probabilities: bool = True) -> Dict[str, Any]:
        """Make predictions on ECG images"""
        
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Preprocess images
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1)
        
        images = images.astype(np.float32) / 255.0
        
        # Get predictions
        predictions = self.model.predict(images)
        
        results = []
        
        for i, pred in enumerate(predictions):
            if self.num_classes == 2:
                # Binary classification
                probability = float(pred[0])
                predicted_class = int(probability > 0.5)
                confidence = max(probability, 1 - probability)
                
                result = {
                    'predicted_class': predicted_class,
                    'predicted_condition': self.condition_mapping.get(predicted_class, 'Unknown'),
                    'confidence': confidence,
                    'probability_positive': probability
                }
            else:
                # Multi-class classification
                predicted_class = int(np.argmax(pred))
                probabilities = pred.astype(float)
                confidence = float(np.max(probabilities))
                
                # Get top predictions
                top_indices = np.argsort(probabilities)[::-1][:3]
                top_predictions = [
                    {
                        'class': int(idx),
                        'condition': self.condition_mapping.get(idx, f'Class_{idx}'),
                        'probability': float(probabilities[idx])
                    }
                    for idx in top_indices
                ]
                
                result = {
                    'predicted_class': predicted_class,
                    'predicted_condition': self.condition_mapping.get(predicted_class, 'Unknown'),
                    'confidence': confidence,
                    'top_predictions': top_predictions
                }
                
                if return_probabilities:
                    result['all_probabilities'] = probabilities.tolist()
            
            results.append(result)
        
        return results
    
    def save_model(self, filepath: str):
        """Save trained model and label encoder"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model
        model_path = f"{filepath}_model.h5"
        self.model.save(model_path)
        
        # Save label encoder
        encoder_path = f"{filepath}_label_encoder.pkl"
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save configuration
        config = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'condition_mapping': self.condition_mapping
        }
        config_path = f"{filepath}_config.pkl"
        joblib.dump(config, config_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, filepath: str):
        """Load trained model and label encoder"""
        
        # Load model
        model_path = f"{filepath}_model.h5"
        if os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load label encoder
        encoder_path = f"{filepath}_label_encoder.pkl"
        if os.path.exists(encoder_path):
            self.label_encoder = joblib.load(encoder_path)
        
        # Load configuration
        config_path = f"{filepath}_config.pkl"
        if os.path.exists(config_path):
            config = joblib.load(config_path)
            self.input_shape = config.get('input_shape', self.input_shape)
            self.num_classes = config.get('num_classes', self.num_classes)
            self.condition_mapping = config.get('condition_mapping', self.condition_mapping)
        
        logger.info(f"Model loaded from {model_path}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        if self.history is None:
            logger.warning("No training history available")
            return
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
