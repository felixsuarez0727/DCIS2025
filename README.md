# Enhanced Radar Signal Classification

This project implements an advanced radar signal classification system with a specific focus on resolving confusion between AM combined signals and PULSED Air-Ground-MTI signals.

## Key Features

- **Attention-based CNN architecture**: Specialized neural network with channel and spatial attention mechanisms
- **Advanced feature extraction**: Enhanced signal processing with frequency domain and wavelet features
- **Targeted class weighting**: Custom weighting to improve discrimination between confusable classes
- **Focal loss implementation**: Modified loss function to focus on hard-to-classify examples
- **Flexible model selection**: Choice between TensorFlow-based CNN and Random Forest classifiers
- **Comprehensive evaluation**: Specialized metrics to analyze specific class confusions

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/radar_classification_enhanced.git
   cd radar_classification_enhanced
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Quick Start

To train a model with enhanced attention mechanisms and 10,000 samples per class:

```bash
python main.py --train_dataset /path/to/dataset.hdf5 --combine_am --samples_per_class 10000 --batch_size 32 --epochs 30
```

## Command Line Arguments

### Dataset Options
- `--train_dataset`: Path to training HDF5 dataset (required)
- `--test_dataset`: Path to test HDF5 dataset (optional, will split from training if not provided)
- `--data_percentage`: Percentage of data to use (default: 1.0)
- `--samples_per_class`: Number of samples per class (default: 10000)

### Model Options
- `--model_type`: Model type to use (choices: 'tf' or 'rf', default: 'tf')
  - 'tf': TensorFlow-based CNN with attention mechanisms
  - 'rf': Random Forest classifier with enhanced features

### Training Options
- `--epochs`: Number of training epochs (default: 30)
- `--batch_size`: Training batch size (default: 32)
- `--cross_validation`: Enable cross-validation
- `--cv_splits`: Number of cross-validation splits (default: 5)

### Feature Options
- `--combine_am`: Combine AM-related signals into one class (AM-DSB, AM-SSB, ASK)
- `--no_frequency_features`: Disable frequency domain feature extraction
- `--no_wavelet`: Disable wavelet transform features

### Regularization Options
- `--no_class_weights`: Disable class weights
- `--no_early_stopping`: Disable early stopping

## Model Architecture

### CNN with Attention (TensorFlow)
The CNN model incorporates multiple attention mechanisms:

1. **Channel Attention**: Focuses on important frequency channels
2. **Spatial Attention**: Highlights relevant time-frequency regions
3. **Residual Connections**: Facilitates gradient flow and feature reuse
4. **Focal Loss**: Custom loss function to focus on hard examples, especially between AM and PULSED classes

### Enhanced Random Forest (scikit-learn)
The Random Forest model includes:

1. **Advanced Feature Extraction**: Time-frequency domain features specifically designed to discriminate AM from PULSED
2. **Balanced Sampling**: Weighted sampling to handle class imbalance
3. **Optimized Hyperparameters**: Fine-tuned for better AM vs PULSED discrimination

## Results Visualization

The system automatically generates several visualizations in the `results/plots/` directory:

- **Confusion Matrix**: With highlighted areas showing AM vs PULSED confusions
- **Training History**: Accuracy and loss curves
- **Problem Classes Analysis**: Specific metrics for the confusable classes
- **Feature Importance**: For Random Forest models

## Evaluating Confusion Between Classes

The system includes specialized metrics to evaluate confusion between AM_combined and PULSED_Air-Ground-MTI classes:

- **Discriminative Power**: Measures the model's ability to distinguish between these classes
- **Confusion Rates**: Percentage of samples confused between classes
- **Decision Confidence**: Confidence scores for correctly and incorrectly classified examples

## Example Usage Scenarios

### Basic Training with TensorFlow
```bash
python main.py --train_dataset /path/to/dataset.hdf5 --combine_am --samples_per_class 10000
```

### Training with Random Forest
```bash
python main.py --train_dataset /path/to/dataset.hdf5 --combine_am --samples_per_class 10000 --model_type rf
```

### Cross-Validation
```bash
python main.py --train_dataset /path/to/dataset.hdf5 --combine_am --samples_per_class 10000 --cross_validation --cv_splits 5
```

### Separate Train/Test Datasets
```bash
python main.py --train_dataset /path/to/train.hdf5 --test_dataset /path/to/test.hdf5 --combine_am --samples_per_class 10000
```

## License

MIT License