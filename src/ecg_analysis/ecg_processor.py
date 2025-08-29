"""
ECG Image Processor
Handles ECG image preprocessing, quality assessment, and feature extraction
"""

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure
from scipy import signal, ndimage
from typing import Dict, List, Any, Tuple, Optional
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ECGImageProcessor:
    """
    Comprehensive ECG image preprocessing and analysis
    """
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        """Initialize ECG processor with target image dimensions"""
        self.target_size = target_size
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # ECG signal characteristics
        self.ecg_frequency_range = (0.5, 150)  # Hz
        self.grid_line_threshold = 0.8
        self.noise_threshold = 0.1
        
        # Cardiac rhythm patterns (for validation)
        self.normal_hr_range = (60, 100)  # bpm
        self.qrs_width_range = (0.06, 0.12)  # seconds
        
    def validate_ecg_image(self, image_path: str) -> Dict[str, Any]:
        """
        Validate ECG image quality and format
        """
        validation_result = {
            'is_valid': False,
            'file_exists': False,
            'format_supported': False,
            'image_readable': False,
            'has_grid': False,
            'signal_quality': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check file existence
            if not os.path.exists(image_path):
                validation_result['issues'].append("File does not exist")
                return validation_result
            
            validation_result['file_exists'] = True
            
            # Check file format
            file_ext = Path(image_path).suffix.lower()
            if file_ext not in self.supported_formats:
                validation_result['issues'].append(f"Unsupported format: {file_ext}")
                validation_result['recommendations'].append("Convert to JPG, PNG, or TIFF format")
                return validation_result
            
            validation_result['format_supported'] = True
            
            # Try to read image
            image = cv2.imread(image_path)
            if image is None:
                validation_result['issues'].append("Cannot read image file")
                return validation_result
            
            validation_result['image_readable'] = True
            
            # Check image dimensions
            height, width = image.shape[:2]
            if width < 200 or height < 200:
                validation_result['issues'].append("Image resolution too low")
                validation_result['recommendations'].append("Use higher resolution ECG images (>= 200x200)")
            
            # Detect ECG grid pattern
            grid_score = self._detect_ecg_grid(image)
            validation_result['has_grid'] = grid_score > self.grid_line_threshold
            
            # Assess signal quality
            signal_quality = self._assess_signal_quality(image)
            validation_result['signal_quality'] = signal_quality
            
            if signal_quality < 0.3:
                validation_result['issues'].append("Poor signal quality detected")
                validation_result['recommendations'].append("Use clearer ECG images with visible waveforms")
            
            # Overall validation
            validation_result['is_valid'] = (
                validation_result['file_exists'] and
                validation_result['format_supported'] and
                validation_result['image_readable'] and
                signal_quality > 0.2
            )
            
        except Exception as e:
            validation_result['issues'].append(f"Validation error: {str(e)}")
            logger.error(f"ECG validation failed for {image_path}: {e}")
        
        return validation_result
    
    def preprocess_ecg_image(self, image_path: str, 
                           remove_grid: bool = True,
                           enhance_contrast: bool = True,
                           denoise: bool = True) -> np.ndarray:
        """
        Comprehensive ECG image preprocessing
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Step 1: Remove grid lines if requested
            if remove_grid:
                gray = self._remove_grid_lines(gray)
            
            # Step 2: Enhance contrast
            if enhance_contrast:
                gray = self._enhance_contrast(gray)
            
            # Step 3: Denoise
            if denoise:
                gray = self._denoise_image(gray)
            
            # Step 4: Normalize intensity
            gray = self._normalize_intensity(gray)
            
            # Step 5: Resize to target size
            processed = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            
            return processed
            
        except Exception as e:
            logger.error(f"ECG preprocessing failed for {image_path}: {e}")
            raise
    
    def extract_ecg_features(self, processed_image: np.ndarray) -> Dict[str, float]:
        """
        Extract quantitative features from preprocessed ECG image
        """
        features = {}
        
        try:
            # Basic image statistics
            features['mean_intensity'] = np.mean(processed_image)
            features['std_intensity'] = np.std(processed_image)
            features['contrast'] = np.std(processed_image) / (np.mean(processed_image) + 1e-6)
            
            # Edge density (indicator of signal complexity)
            edges = cv2.Canny(processed_image.astype(np.uint8), 50, 150)
            features['edge_density'] = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Horizontal line density (ECG waveform indicator)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            features['horizontal_line_density'] = np.sum(horizontal_lines > 0) / (edges.shape[0] * edges.shape[1])
            
            # Vertical variations (QRS complexity)
            vertical_gradient = np.gradient(processed_image, axis=0)
            features['vertical_variation'] = np.std(vertical_gradient)
            
            # Horizontal variations (RR interval variability)
            horizontal_gradient = np.gradient(processed_image, axis=1)
            features['horizontal_variation'] = np.std(horizontal_gradient)
            
            # Signal-to-noise ratio estimation
            signal_power = np.var(processed_image)
            noise_estimate = np.var(processed_image - ndimage.gaussian_filter(processed_image, sigma=1))
            features['snr_estimate'] = signal_power / (noise_estimate + 1e-6)
            
            # Frequency domain features
            fft_signal = np.fft.fft2(processed_image)
            fft_magnitude = np.abs(fft_signal)
            features['dominant_frequency'] = self._estimate_dominant_frequency(fft_magnitude)
            features['frequency_spread'] = np.std(fft_magnitude)
            
            # Morphological features
            features.update(self._extract_morphological_features(processed_image))
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return basic features if extraction fails
            features = {
                'mean_intensity': np.mean(processed_image),
                'std_intensity': np.std(processed_image),
                'contrast': 1.0,
                'edge_density': 0.1,
                'horizontal_line_density': 0.1,
                'vertical_variation': 1.0,
                'horizontal_variation': 1.0,
                'snr_estimate': 1.0,
                'dominant_frequency': 1.0,
                'frequency_spread': 1.0
            }
        
        return features
    
    def batch_process_ecg_images(self, image_dir: str, output_dir: str,
                               metadata_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Process multiple ECG images in batch
        """
        results = {
            'processed_count': 0,
            'failed_count': 0,
            'validation_results': {},
            'feature_summary': {},
            'processing_time': 0
        }
        
        import time
        start_time = time.time()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(Path(image_dir).glob(f"*{ext}"))
            image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
        
        all_features = []
        
        for image_file in image_files:
            try:
                logger.info(f"Processing {image_file.name}")
                
                # Validate image
                validation = self.validate_ecg_image(str(image_file))
                results['validation_results'][image_file.name] = validation
                
                if not validation['is_valid']:
                    results['failed_count'] += 1
                    logger.warning(f"Skipping invalid image: {image_file.name}")
                    continue
                
                # Preprocess image
                processed_image = self.preprocess_ecg_image(str(image_file))
                
                # Extract features
                features = self.extract_ecg_features(processed_image)
                features['filename'] = image_file.name
                all_features.append(features)
                
                # Save processed image
                output_path = Path(output_dir) / f"processed_{image_file.name}"
                cv2.imwrite(str(output_path), processed_image)
                
                results['processed_count'] += 1
                
            except Exception as e:
                results['failed_count'] += 1
                logger.error(f"Failed to process {image_file.name}: {e}")
        
        # Create feature summary
        if all_features:
            features_df = pd.DataFrame(all_features)
            
            # Save features to CSV
            if metadata_file:
                features_df.to_csv(metadata_file, index=False)
            
            # Create summary statistics
            numeric_features = features_df.select_dtypes(include=[np.number])
            results['feature_summary'] = {
                'mean_features': numeric_features.mean().to_dict(),
                'std_features': numeric_features.std().to_dict(),
                'feature_correlation': numeric_features.corr().to_dict()
            }
        
        results['processing_time'] = time.time() - start_time
        logger.info(f"Batch processing completed: {results['processed_count']} success, {results['failed_count']} failed")
        
        return results
    
    def _detect_ecg_grid(self, image: np.ndarray) -> float:
        """Detect ECG grid pattern in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines  
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # Calculate grid score
        h_score = np.sum(horizontal_lines > 0) / (gray.shape[0] * gray.shape[1])
        v_score = np.sum(vertical_lines > 0) / (gray.shape[0] * gray.shape[1])
        
        return (h_score + v_score) / 2
    
    def _assess_signal_quality(self, image: np.ndarray) -> float:
        """Assess ECG signal quality"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Calculate signal-to-noise ratio
        signal_var = np.var(gray)
        noise_var = np.var(gray - cv2.GaussianBlur(gray, (5, 5), 0))
        
        if noise_var == 0:
            return 1.0
        
        snr = signal_var / noise_var
        quality_score = min(snr / 10.0, 1.0)  # Normalize to 0-1
        
        return quality_score
    
    def _remove_grid_lines(self, image: np.ndarray) -> np.ndarray:
        """Remove ECG grid lines while preserving waveforms"""
        # Detect and remove horizontal grid lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        horizontal_mask = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect and remove vertical grid lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        vertical_mask = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine masks
        grid_mask = cv2.bitwise_or(horizontal_mask, vertical_mask)
        
        # Remove grid while preserving strong signals
        result = image.copy()
        grid_pixels = grid_mask > 0
        result[grid_pixels] = cv2.inpaint(image, (grid_mask > 0).astype(np.uint8), 3, cv2.INPAINT_TELEA)[grid_pixels]
        
        return result
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance ECG image contrast"""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # Additional gamma correction
        gamma = 1.2
        lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        enhanced = cv2.LUT(enhanced, lookup_table)
        
        return enhanced
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Remove noise while preserving ECG waveform details"""
        # Use bilateral filter to preserve edges
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Apply gentle Gaussian blur for additional smoothing
        denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
        
        return denoised
    
    def _normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """Normalize image intensity to standard range"""
        # Normalize to 0-255 range
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized.astype(np.uint8)
    
    def _estimate_dominant_frequency(self, fft_magnitude: np.ndarray) -> float:
        """Estimate dominant frequency from FFT magnitude"""
        # Find peak frequency
        flat_fft = fft_magnitude.flatten()
        peak_idx = np.argmax(flat_fft)
        
        # Convert to frequency estimate (normalized)
        return peak_idx / len(flat_fft)
    
    def _extract_morphological_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract morphological features from ECG"""
        features = {}
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Opening (erosion followed by dilation)
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        features['opening_response'] = np.mean(opened)
        
        # Closing (dilation followed by erosion)
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        features['closing_response'] = np.mean(closed)
        
        # Gradient (difference between dilation and erosion)
        gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        features['morphological_gradient'] = np.mean(gradient)
        
        return features
    
    def visualize_processing_steps(self, image_path: str, save_path: Optional[str] = None):
        """Visualize ECG preprocessing steps"""
        # Load original image
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Processing steps
        step1 = self._remove_grid_lines(original)
        step2 = self._enhance_contrast(step1)
        step3 = self._denoise_image(step2)
        final = self._normalize_intensity(step3)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        images = [original, step1, step2, step3, final]
        titles = ['Original', 'Grid Removed', 'Contrast Enhanced', 'Denoised', 'Final Processed']
        
        for i, (img, title) in enumerate(zip(images, titles)):
            row, col = divmod(i, 3)
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
        
        # Hide empty subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
