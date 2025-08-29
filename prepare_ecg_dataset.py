#!/usr/bin/env python3
"""
ECG Dataset Preparation Script
Helps organize and prepare ECG image datasets for training
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.ecg_analysis import ECGImageProcessor, ECGAnalyzer


def organize_ecg_dataset(source_dir: str, output_dir: str, 
                        create_labels: bool = True) -> Dict[str, Any]:
    """
    Organize ECG images and create labels file
    """
    logger.info(f"Organizing ECG dataset from {source_dir} to {output_dir}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize processor for validation
    processor = ECGImageProcessor()
    
    # Find all image files
    image_files = []
    for ext in processor.supported_formats:
        image_files.extend(Path(source_dir).rglob(f"*{ext}"))
        image_files.extend(Path(source_dir).rglob(f"*{ext.upper()}"))
    
    logger.info(f"Found {len(image_files)} potential ECG images")
    
    # Process and organize files
    organized_data = []
    processed_count = 0
    
    for image_file in image_files:
        try:
            # Validate image
            validation = processor.validate_ecg_image(str(image_file))
            
            if validation['is_valid']:
                # Determine condition from path/filename
                condition = extract_condition_from_path(image_file)
                
                # Create organized filename
                new_filename = f"{condition.lower().replace(' ', '_')}_{processed_count:04d}{image_file.suffix}"
                new_path = Path(output_dir) / new_filename
                
                # Copy file to organized location (or create symlink)
                import shutil
                shutil.copy2(str(image_file), str(new_path))
                
                # Store metadata
                organized_data.append({
                    'filename': new_filename,
                    'original_path': str(image_file),
                    'condition': condition,
                    'signal_quality': validation['signal_quality'],
                    'has_grid': validation['has_grid'],
                    'file_size': image_file.stat().st_size,
                    'valid': True
                })
                
                processed_count += 1
                
            else:
                logger.warning(f"Invalid ECG image: {image_file.name} - {', '.join(validation['issues'])}")
                organized_data.append({
                    'filename': image_file.name,
                    'original_path': str(image_file),
                    'condition': 'Invalid',
                    'signal_quality': validation['signal_quality'],
                    'has_grid': False,
                    'file_size': image_file.stat().st_size,
                    'valid': False,
                    'issues': '; '.join(validation['issues'])
                })
                
        except Exception as e:
            logger.error(f"Failed to process {image_file}: {e}")
    
    # Create labels file
    if create_labels and organized_data:
        labels_df = pd.DataFrame(organized_data)
        labels_file = Path(output_dir) / 'ecg_labels.csv'
        labels_df.to_csv(labels_file, index=False)
        logger.info(f"Created labels file: {labels_file}")
    
    # Generate summary
    summary = {
        'total_files': len(image_files),
        'valid_files': processed_count,
        'invalid_files': len(image_files) - processed_count,
        'output_directory': output_dir,
        'conditions_found': list(set([item['condition'] for item in organized_data if item['valid']]))
    }
    
    logger.info(f"Dataset organization complete:")
    logger.info(f"  - Total files processed: {summary['total_files']}")
    logger.info(f"  - Valid ECG images: {summary['valid_files']}")
    logger.info(f"  - Invalid images: {summary['invalid_files']}")
    logger.info(f"  - Conditions found: {summary['conditions_found']}")
    
    return summary


def extract_condition_from_path(image_path: Path) -> str:
    """
    Extract cardiac condition from image path or filename
    """
    # Convert to string for pattern matching
    path_str = str(image_path).lower()
    filename = image_path.name.lower()
    parent_dir = image_path.parent.name.lower()
    
    # Common ECG condition patterns
    condition_patterns = {
        # Normal
        'normal': 'Normal',
        'healthy': 'Normal',
        'sinus': 'Normal',
        
        # Myocardial Infarction
        'mi': 'Myocardial Infarction',
        'myocardial_infarction': 'Myocardial Infarction',
        'heart_attack': 'Myocardial Infarction',
        'stemi': 'Myocardial Infarction',
        'nstemi': 'Myocardial Infarction',
        
        # Bundle Branch Blocks
        'lbbb': 'Left Bundle Branch Block',
        'left_bundle': 'Left Bundle Branch Block',
        'rbbb': 'Right Bundle Branch Block',
        'right_bundle': 'Right Bundle Branch Block',
        
        # Arrhythmias
        'af': 'Atrial Fibrillation',
        'afib': 'Atrial Fibrillation',
        'atrial_fib': 'Atrial Fibrillation',
        'atrial_fibrillation': 'Atrial Fibrillation',
        
        'pvc': 'Premature Ventricular Contraction',
        'premature_ventricular': 'Premature Ventricular Contraction',
        'ventricular_ectopy': 'Premature Ventricular Contraction',
        
        # Other conditions
        'vt': 'Ventricular Tachycardia',
        'ventricular_tach': 'Ventricular Tachycardia',
        'svt': 'Supraventricular Tachycardia',
        'bradycardia': 'Bradycardia',
        'tachycardia': 'Tachycardia'
    }
    
    # Search for patterns in filename and path
    for pattern, condition in condition_patterns.items():
        if pattern in filename or pattern in parent_dir or pattern in path_str:
            return condition
    
    # Try to extract from common filename formats
    if '_' in filename:
        parts = filename.split('_')
        for part in parts:
            if part in condition_patterns:
                return condition_patterns[part]
    
    # Default to Unknown
    return 'Unknown'


def create_sample_labels_file(output_path: str):
    """
    Create a sample labels file template
    """
    sample_data = [
        {'filename': 'normal_0001.jpg', 'condition': 'Normal', 'patient_id': 'P001', 'age': 45, 'gender': 'M'},
        {'filename': 'mi_0001.jpg', 'condition': 'Myocardial Infarction', 'patient_id': 'P002', 'age': 62, 'gender': 'F'},
        {'filename': 'af_0001.jpg', 'condition': 'Atrial Fibrillation', 'patient_id': 'P003', 'age': 58, 'gender': 'M'},
        {'filename': 'lbbb_0001.jpg', 'condition': 'Left Bundle Branch Block', 'patient_id': 'P004', 'age': 71, 'gender': 'F'},
        {'filename': 'pvc_0001.jpg', 'condition': 'Premature Ventricular Contraction', 'patient_id': 'P005', 'age': 39, 'gender': 'M'}
    ]
    
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)
    
    print(f"Sample labels file created at: {output_path}")
    print("Columns: filename, condition, patient_id, age, gender")
    print("You can modify this file to match your ECG dataset structure.")


def validate_dataset(dataset_dir: str, labels_file: str) -> Dict[str, Any]:
    """
    Validate ECG dataset completeness and quality
    """
    logger.info(f"Validating ECG dataset in {dataset_dir}")
    
    # Initialize analyzer
    analyzer = ECGAnalyzer()
    
    try:
        # Load dataset
        images, labels, metadata = analyzer.load_ecg_dataset(
            image_dir=dataset_dir,
            labels_file=labels_file
        )
        
        # Dataset statistics
        unique_conditions = np.unique(labels)
        condition_counts = pd.Series(labels).value_counts()
        
        validation_result = {
            'valid': True,
            'total_images': len(images),
            'image_shape': images[0].shape,
            'unique_conditions': unique_conditions.tolist(),
            'condition_distribution': condition_counts.to_dict(),
            'average_signal_quality': metadata['validation_score'].mean(),
            'low_quality_images': len(metadata[metadata['validation_score'] < 0.5])
        }
        
        logger.info("Dataset validation successful:")
        logger.info(f"  - Total images: {validation_result['total_images']}")
        logger.info(f"  - Unique conditions: {len(unique_conditions)}")
        logger.info(f"  - Average quality: {validation_result['average_signal_quality']:.3f}")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        return {'valid': False, 'error': str(e)}


def main():
    """Main function for CLI usage"""
    
    parser = argparse.ArgumentParser(description="ECG Dataset Preparation Utilities")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Organize command
    organize_parser = subparsers.add_parser('organize', help='Organize ECG images')
    organize_parser.add_argument('source_dir', help='Source directory with ECG images')
    organize_parser.add_argument('output_dir', help='Output directory for organized dataset')
    organize_parser.add_argument('--no-labels', action='store_true', help='Skip creating labels file')
    
    # Sample labels command
    sample_parser = subparsers.add_parser('sample-labels', help='Create sample labels file')
    sample_parser.add_argument('output_file', help='Output path for sample labels file')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate ECG dataset')
    validate_parser.add_argument('dataset_dir', help='Dataset directory')
    validate_parser.add_argument('labels_file', help='Labels CSV file')
    
    args = parser.parse_args()
    
    if args.command == 'organize':
        summary = organize_ecg_dataset(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            create_labels=not args.no_labels
        )
        print(f"Organization complete. Processed {summary['valid_files']} valid ECG images.")
        
    elif args.command == 'sample-labels':
        create_sample_labels_file(args.output_file)
        
    elif args.command == 'validate':
        result = validate_dataset(args.dataset_dir, args.labels_file)
        if result['valid']:
            print("✅ Dataset validation successful")
            print(f"Total images: {result['total_images']}")
            print(f"Conditions: {result['unique_conditions']}")
        else:
            print(f"❌ Dataset validation failed: {result['error']}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
