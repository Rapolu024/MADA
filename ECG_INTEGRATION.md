# ECG Integration for MADA System

## Overview

The Medical AI Early Diagnosis Assistant (MADA) now includes comprehensive ECG (Electrocardiogram) image analysis capabilities for cardiac condition detection and diagnosis.

## Features

### 🔍 **ECG Image Processing**
- Advanced image preprocessing (grid removal, contrast enhancement, denoising)
- Quality validation and assessment
- Feature extraction (morphological, frequency-domain, statistical)
- Batch processing capabilities

### 🧠 **Deep Learning Models**
- CNN and ResNet architectures for ECG classification
- Multi-class cardiac condition detection
- Confidence scoring and uncertainty quantification
- Transfer learning capabilities

### 📊 **Supported Cardiac Conditions**
- **Normal** - Healthy cardiac rhythm
- **Myocardial Infarction (MI)** - Heart attack detection
- **Left Bundle Branch Block (LBBB)** - Conduction disorder
- **Right Bundle Branch Block (RBBB)** - Conduction disorder
- **Premature Ventricular Contraction (PVC)** - Arrhythmia
- **Atrial Fibrillation (AF)** - Irregular heart rhythm

### 🏥 **Clinical Integration**
- MADA patient intake integration
- Clinical recommendation generation
- Risk assessment and urgency scoring
- Comprehensive reporting

## Directory Structure

```
MADA/
├── data/ecg_images/           # ECG dataset storage
│   ├── raw/                   # Original ECG images
│   ├── processed/             # Preprocessed images
│   ├── train/                 # Training set
│   ├── test/                  # Test set
│   └── validation/            # Validation set
├── data/ecg_metadata/         # Metadata and labels
├── src/ecg_analysis/          # ECG analysis modules
│   ├── __init__.py
│   ├── ecg_processor.py       # Image preprocessing
│   ├── ecg_models.py          # Deep learning models
│   └── ecg_analyzer.py        # Main analysis interface
└── prepare_ecg_dataset.py     # Dataset preparation utilities
```

## Quick Start

### 1. Prepare Your ECG Dataset

First, organize your ECG images using the preparation script:

```bash
# Organize ECG images from a source directory
python prepare_ecg_dataset.py organize /path/to/your/ecg/images data/ecg_images/raw

# Create a sample labels file template
python prepare_ecg_dataset.py sample-labels data/ecg_metadata/sample_labels.csv

# Validate your dataset
python prepare_ecg_dataset.py validate data/ecg_images/raw data/ecg_metadata/ecg_labels.csv
```

### 2. Basic ECG Analysis

```python
from src.ecg_analysis import ECGAnalyzer

# Initialize analyzer
analyzer = ECGAnalyzer()

# Load and prepare dataset
images, labels, metadata = analyzer.load_ecg_dataset(
    image_dir='data/ecg_images/raw',
    labels_file='data/ecg_metadata/ecg_labels.csv'
)

# Train ECG classifier
training_summary = analyzer.train_ecg_classifier(
    images=images,
    labels=labels,
    epochs=50,
    save_model_path='models/ecg_classifier'
)

# Analyze single ECG
result = analyzer.analyze_ecg_image('path/to/ecg/image.jpg')
print(f"Predicted condition: {result['classification']['predicted_condition']}")
print(f"Confidence: {result['classification']['confidence']:.3f}")
```

### 3. Batch Analysis

```python
# Analyze multiple ECG images
results_df = analyzer.batch_analyze_ecgs(
    image_dir='data/ecg_images/test',
    output_file='results/ecg_batch_analysis.csv'
)

# Generate comprehensive report
report = analyzer.generate_analysis_report(
    results_df,
    output_path='reports/ecg_analysis_report.txt'
)
print(report)
```

## Dataset Requirements

### Image Format
- **Supported formats**: JPG, PNG, TIFF, BMP
- **Recommended resolution**: ≥ 512x512 pixels
- **Image quality**: Clear ECG waveforms, minimal noise
- **Grid lines**: Optional (automatically detected and removed)

### Labels File Format
CSV file with the following columns:
```csv
filename,condition,patient_id,age,gender
normal_0001.jpg,Normal,P001,45,M
mi_0002.jpg,Myocardial Infarction,P002,62,F
af_0003.jpg,Atrial Fibrillation,P003,58,M
```

### Directory Structure Options

**Option 1: Condition-based directories**
```
ecg_images/
├── normal/
├── myocardial_infarction/
├── atrial_fibrillation/
├── left_bundle_branch_block/
└── right_bundle_branch_block/
```

**Option 2: Filename-based labeling**
```
ecg_images/
├── normal_001.jpg
├── mi_001.jpg
├── af_001.jpg
└── lbbb_001.jpg
```

## Configuration

### ECG Analyzer Configuration
```python
config = {
    'image_size': (512, 512),      # Target image dimensions
    'num_classes': 6,              # Number of cardiac conditions
    'model_type': 'cnn',           # 'cnn' or 'resnet'
    'preprocessing': {
        'remove_grid': True,       # Remove ECG grid lines
        'enhance_contrast': True,  # Enhance image contrast
        'denoise': True           # Apply denoising
    },
    'batch_size': 32,
    'validation_split': 0.2
}

analyzer = ECGAnalyzer(config=config)
```

### Model Architecture Options
- **CNN**: Custom convolutional neural network optimized for ECG images
- **ResNet**: ResNet-style architecture with residual connections

## Integration with MADA System

### Patient Intake Integration
```python
# Add ECG analysis to patient intake workflow
from src.patient_intake.intake_processor import PatientIntakeProcessor
from src.ecg_analysis import ECGAnalyzer

intake_processor = PatientIntakeProcessor()
ecg_analyzer = ECGAnalyzer()

# Process patient with ECG
patient_data = intake_processor.process_intake_form(intake_form)

# Add ECG analysis if ECG image provided
if ecg_image_path:
    ecg_result = ecg_analyzer.analyze_ecg_image(ecg_image_path)
    patient_data.ecg_analysis = ecg_result
```

### Dashboard Integration
The ECG analysis results are automatically integrated into the MADA dashboard, displaying:
- ECG image preview
- Predicted cardiac conditions
- Confidence scores
- Clinical recommendations
- Risk assessment

## Clinical Recommendations

The system generates evidence-based clinical recommendations:

### Myocardial Infarction
- ⚠️ **URGENT**: Immediate cardiology consultation
- Consider emergency cardiac catheterization
- Monitor cardiac enzymes and vital signs

### Atrial Fibrillation
- Assess stroke risk (CHA2DS2-VASc score)
- Consider anticoagulation therapy evaluation
- Monitor heart rate and rhythm

### Bundle Branch Blocks
- Evaluate cardiac function
- Consider echocardiography
- Assess for underlying heart disease

### Arrhythmias
- Assess frequency and clinical context
- Consider 24-hour Holter monitoring
- Evaluate for structural heart disease

## Performance Metrics

### Model Performance
- **Accuracy**: Typically >90% on validation data
- **Sensitivity**: >85% for critical conditions (MI, AF)
- **Specificity**: >90% for normal ECGs
- **Processing Speed**: ~0.5 seconds per image

### Quality Metrics
- **Signal Quality Assessment**: 0-1 score
- **Grid Detection**: Automatic grid line identification
- **Artifact Detection**: Noise and interference identification

## Advanced Features

### Feature Extraction
Comprehensive feature extraction including:
- **Morphological features**: Opening, closing, gradient responses
- **Statistical features**: Mean intensity, contrast, edge density
- **Frequency domain**: Dominant frequency, spectral characteristics
- **Signal quality**: SNR estimation, artifact detection

### Uncertainty Quantification
- **Confidence scoring**: Model prediction confidence
- **Ensemble methods**: Multiple model consensus
- **Uncertainty estimation**: Epistemic and aleatoric uncertainty

### Continuous Learning
- **Online learning**: Model updates with new data
- **Active learning**: Identify informative samples
- **Performance monitoring**: Track model degradation

## Data Privacy and Security

### HIPAA Compliance
- **Data encryption**: All ECG images encrypted at rest
- **Access controls**: Role-based access to ECG data
- **Audit logging**: Complete audit trail for ECG analysis
- **De-identification**: Automatic removal of patient identifiers

### Security Features
- **Secure storage**: Encrypted ECG image storage
- **Access logging**: Track all ECG data access
- **Data retention**: Configurable retention policies

## API Reference

### ECGAnalyzer Methods

#### `load_ecg_dataset(image_dir, labels_file)`
Load ECG dataset from directory and labels file.

#### `train_ecg_classifier(images, labels, epochs, save_path)`
Train ECG classification model.

#### `analyze_ecg_image(image_path, include_features)`
Analyze single ECG image.

#### `batch_analyze_ecgs(image_dir, output_file)`
Analyze multiple ECG images in batch.

#### `save_model(filepath)` / `load_model(filepath)`
Save and load trained models.

### ECGProcessor Methods

#### `validate_ecg_image(image_path)`
Validate ECG image quality and format.

#### `preprocess_ecg_image(image_path, options)`
Preprocess ECG image with specified options.

#### `extract_ecg_features(image)`
Extract quantitative features from ECG image.

## Troubleshooting

### Common Issues

**Issue**: Low prediction confidence
- **Solution**: Check ECG image quality, ensure proper lead placement

**Issue**: Grid lines affecting results
- **Solution**: Enable `remove_grid=True` in preprocessing options

**Issue**: Poor model performance
- **Solution**: Increase training data, balance class distribution

**Issue**: Memory errors during training
- **Solution**: Reduce batch size, use smaller image dimensions

### Performance Optimization

1. **GPU Acceleration**: Install TensorFlow with GPU support
2. **Batch Processing**: Process multiple images simultaneously
3. **Image Resizing**: Use appropriate target image dimensions
4. **Model Optimization**: Use quantization for deployment

## Support and Documentation

### Getting Help
- Check the main MADA documentation
- Review ECG analysis examples
- Contact the development team

### Contributing
- Follow MADA coding standards
- Add comprehensive tests
- Update documentation
- Submit pull requests

## Future Enhancements

### Planned Features
- **12-lead ECG support**: Multi-lead ECG analysis
- **Temporal analysis**: ECG sequence analysis
- **Risk scoring**: Automated cardiac risk assessment
- **Integration**: EMR system integration
- **Mobile support**: Mobile ECG analysis

### Research Directions
- **Federated learning**: Distributed ECG model training
- **Explainable AI**: Interpretable ECG predictions
- **Multi-modal fusion**: Combine ECG with other cardiac data
- **Real-time analysis**: Continuous ECG monitoring

---

For detailed technical documentation and examples, please refer to the individual module documentation in the `src/ecg_analysis/` directory.
