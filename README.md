# Enhanced Radar Signal Classification V2

This project implements an advanced radar signal classification system with a specific focus on resolving confusion between AM combined signals and PULSED Air-Ground-MTI signals.

## Key Features

- **Dense Neural Network architecture**: Specialized neural network with batch normalization and dropout for robust generalization
- **Advanced feature extraction**: Enhanced signal processing with spectrograms, wavelet features, and frequency domain analysis
- **Targeted class weighting**: Custom weighting to improve discrimination between confusable classes
- **Multi-domain analysis**: Combined time, frequency, and time-frequency domain features for improved classification
- **Flexible model selection**: Choice between TensorFlow-based Dense Network and Random Forest classifiers
- **Comprehensive evaluation**: Specialized metrics to analyze specific class confusions
- **Improved data splitting**: 70% training, 15% validation and 15% test splits for more robust performance assessment
- **Interactive neural architecture visualization**: React-based visualization for better understanding of the model structure

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/radar_classification_enhanced_V2.git
   cd radar_classification_enhanced_V2
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Quick Start

To train a model with 20,000 samples per class:

```bash
python main.py --train_dataset /path/to/dataset.hdf5 --combine_am --samples_per_class 20000 --batch_size 64 --epochs 30
```

## Command Line Arguments

### Dataset Options
- `--train_dataset`: Path to training HDF5 dataset (required)
- `--test_dataset`: Path to test HDF5 dataset (optional, will split from training if not provided)
- `--data_percentage`: Percentage of data to use (default: 1.0)
- `--samples_per_class`: Number of samples per class (default: 20000)

### Model Options
- `--model_type`: Model type to use (choices: 'tf' or 'rf', default: 'tf')
  - 'tf': TensorFlow-based Dense Neural Network
  - 'rf': Random Forest classifier (alternative implementation)

### Training Options
- `--epochs`: Number of training epochs (default: 30)
- `--batch_size`: Training batch size (default: 64)
- `--cross_validation`: Enable cross-validation
- `--cv_splits`: Number of cross-validation splits (default: 5)

### Feature Options
- `--combine_am`: Combine AM-related signals into one class (AM-DSB, AM-SSB, ASK)

### Regularization Options
- `--no_class_weights`: Disable class weights
- `--no_early_stopping`: Disable early stopping

## Model Architecture

### Dense Neural Network (TensorFlow)
The Dense Neural Network model incorporates several design features:

1. **Input Layer**: Accepts 513-dimensional feature vectors
2. **Hidden Layers**: Two dense layers (256 and 128 neurons) with ReLU activation
3. **Batch Normalization**: After each dense layer to stabilize learning
4. **Dropout (50%)**: Strong regularization to prevent overfitting
5. **Output Layer**: 7 neurons with softmax activation for class probabilities

### Alternative Random Forest (scikit-learn)
The alternative Random Forest model includes:

1. **Advanced Feature Processing**: Same 513-dimensional feature vectors as the neural network
2. **Balanced Class Weights**: Weighted sampling to handle class imbalance
3. **Optimized Hyperparameters**: Fine-tuned for better discrimination between classes

## Interactive Model Architecture Visualization

The project includes a modern React-based visualization of the neural network architecture:

### Viewing the Visualization

1. Navigate to the Neural_model_architecture directory:
   ```bash
   cd Neural_model_architecture
   ```

2. Open the visualization in your browser:
   ```bash
   # If you have Node.js installed, you can run:
   npm install  # Only needed the first time
   npm start
   
   # Alternatively, simply open the index.html file in any browser:
   open index.html  # On macOS
   # Or just double-click the file in your file explorer
   ```

3. The visualization provides:
   - Detailed layer-by-layer breakdown of the neural network
   - Shape information for each layer (dimensions)
   - Parameter counts for each layer
   - Activation functions and dropout rates
   - Visual representation of connections between layers
   - Statistical summary of trainable vs. non-trainable parameters
   - Color-coded layers by type (dense, batch normalization, dropout)

This interactive visualization helps to understand:
- The flow of data through the network
- How the 513-dimensional input transforms through each layer
- Where the majority of parameters are concentrated
- The regularization strategy (dropout placement)
- The complete structure of the classification pipeline

## Advanced Data Splitting Strategy

Version 2.0 implements an improved data splitting strategy:

- **Training set (70%)**: 126,000 samples used for model training
- **Validation set (15%)**: 27,000 samples for parameter tuning during training
- **Test set (15%)**: 27,000 samples for final evaluation

This separation provides more reliable performance metrics compared to the previous version's combined validation/test approach.

## Feature Vector Construction

The 513-dimensional input vector is constructed through three complementary techniques:
- **Spectrograms**: Time-frequency representation (449 features)
- **Wavelet Transform**: Multi-scale decomposition using Daubechies 4 (58 features)
- **FFT Statistics**: Key frequency domain metrics (6 features)

## Results Visualization

The system automatically generates several visualizations in the `results/plots/` directory:

- **Confusion Matrix**: With detailed class-wise accuracy and error rates
- **Training History**: Accuracy and loss curves showing model convergence
- **Train Test Distribution**: Visualization of class distribution across datasets

## Performance Analysis

The system achieved the following metrics in Version 2.0:

- **Overall Accuracy**: 97.72% on the test set
- **AM Signal Accuracy**: 96.99% specific to AM_combined class
- **Training Time**: 80.44 seconds for model training
- **Total Processing**: 20.4 minutes including data loading and analysis
- **Notable Confusions**: 279 PULSED_Air-Ground-MTI signals misclassified as AM_combined

## Example Usage Scenarios

### Basic Training
```bash
python main.py --train_dataset /path/to/dataset.hdf5 --combine_am --samples_per_class 20000
```

### Training with Random Forest (Alternative)
```bash
python main.py --train_dataset /path/to/dataset.hdf5 --combine_am --samples_per_class 20000 --model_type rf
```

### Cross-Validation
```bash
python main.py --train_dataset /path/to/dataset.hdf5 --combine_am --samples_per_class 20000 --cross_validation --cv_splits 5
```

### Analyzing HDF5 Dataset Structure
```bash
python scripts/analyze_hdf5.py --file /path/to/dataset.hdf5 --output results/analysis
```

### Visualizing Neural Network Architecture
```bash
# Navigate to the visualization directory
cd Neural_model_architecture

# Open in browser
open index.html
```

## Future Work

Planned improvements include:
- Characteristic-specific feature extraction for AM vs PULSED discrimination
- More advanced architectures like CNNs and transformers
- Data augmentation techniques for improved class balance
- Ensemble methods combining neural and traditional models
- Neural Architecture Search for optimized model design
- Enhanced visualization tools for model interpretability

## License

MIT License
