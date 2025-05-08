import os
import sys
import argparse
import logging
import logging.config
import time
import traceback

from src.data_loader import DataLoader
from src.train import ModelTrainer
from src.utils import ResultsVisualizer

def setup_logging():
    """Set up logging configuration"""
    if os.path.exists('logging.conf'):
        logging.config.fileConfig('logging.conf')
    else:
        # Basic configuration if no config file is found
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Radar Signal Classification')
    
    # Dataset arguments
    parser.add_argument('--train_dataset', type=str, required=True,
                        help='Path to training HDF5 dataset')
    parser.add_argument('--test_dataset', type=str, default=None,
                        help='Path to test HDF5 dataset (if not provided, will split train dataset)')
    parser.add_argument('--data_percentage', type=float, default=1.0,
                        help='Percentage of data to use (0.0 to 1.0)')
    parser.add_argument('--samples_per_class', type=int, default=25,
                        help='Number of samples per class')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, choices=['tf', 'rf'], default='tf',
                        help='Model type: tf (TensorFlow) or rf (Random Forest)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--cross_validation', action='store_true',
                        help='Use cross-validation')
    parser.add_argument('--cv_splits', type=int, default=5,
                        help='Number of cross-validation splits')
    
    # Feature arguments
    parser.add_argument('--combine_am', action='store_true',
                        help='Combine AM-related signals (AM-DSB, AM-SSB, ASK) into one class')
    
    # Output arguments
    parser.add_argument('--no_class_weights', action='store_true',
                        help='Disable class weights')
    parser.add_argument('--no_early_stopping', action='store_true',
                        help='Disable early stopping')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    # Set up logging
    setup_logging()
    logger = logging.getLogger('main')
    
    # Parse arguments
    args = parse_arguments()
    
    # Start timing
    start_time = time.time()
    
    # Print configuration
    logger.info("Radar Signal Classification")
    logger.info("=" * 50)
    logger.info(f"Training Dataset: {args.train_dataset}")
    logger.info(f"Testing Dataset: {args.test_dataset if args.test_dataset else 'Split from training'}")
    logger.info(f"Data Percentage: {args.data_percentage}")
    logger.info(f"Samples Per Class: {args.samples_per_class}")
    logger.info(f"Combine AM Signals: {args.combine_am}")
    logger.info(f"Model Type: {args.model_type}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Cross Validation: {args.cross_validation}")
    if args.cross_validation:
        logger.info(f"CV Splits: {args.cv_splits}")
    logger.info(f"Use Class Weights: {not args.no_class_weights}")
    logger.info(f"Use Early Stopping: {not args.no_early_stopping}")
    logger.info("=" * 50)
    
    try:
        # Initialize data loader
        logger.info("Initializing data loader...")
        data_loader = DataLoader(
            train_dataset_path=args.train_dataset,
            test_dataset_path=args.test_dataset,
            data_percentage=args.data_percentage,
            samples_per_class=args.samples_per_class,
            combine_am=args.combine_am
        )
        
        # Load data
        logger.info("Loading and preprocessing data...")
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_data()
        
        # Print data information
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_val shape: {X_val.shape}")
        logger.info(f"X_test shape: {X_test.shape}")
        logger.info(f"Number of classes: {len(data_loader.get_class_names())}")
        
        # Initialize results visualizer
        visualizer = ResultsVisualizer()
        
        # Plot class distribution in train and test sets
        logger.info("Plotting class distributions...")
        visualizer.compare_train_test_distribution(
            data_loader.y_train,
            data_loader.y_test,
            data_loader.get_class_names()
        )
        
        # Initialize model trainer
        logger.info("Initializing model trainer...")
        trainer = ModelTrainer(data_loader)
        
        # Train model with or without cross-validation
        if args.cross_validation:
            logger.info(f"Starting {args.cv_splits}-fold cross-validation...")
            results = trainer.train_with_cross_validation(
                epochs=args.epochs,
                batch_size=args.batch_size,
                cv_splits=args.cv_splits,
                use_class_weights=not args.no_class_weights,
                use_early_stopping=not args.no_early_stopping
            )
            
            # Log cross-validation results
            logger.info(f"Cross-validation accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
            
            # Save detailed metrics
            visualizer.save_metrics(results, 'cv_results.json')
            
        else:
            logger.info("Training single model...")
            results = trainer.train_model(
                epochs=args.epochs,
                batch_size=args.batch_size,
                use_class_weights=not args.no_class_weights,
                use_early_stopping=not args.no_early_stopping
            )
            
            # Log training results
            logger.info(f"Test accuracy: {results['accuracy']:.4f}")
            
            # Save metrics
            visualizer.save_metrics(results, 'training_results.json')
        
        # Analyze AM signals specifically
        if args.combine_am:
            logger.info("Analyzing AM signals specifically...")
            am_results = trainer.analyze_am_signals()
            
            if am_results:
                logger.info(f"AM accuracy: {am_results['accuracy']:.4f}")
                logger.info(f"AM recall: {am_results['recall']:.4f}")
                
                # Save AM metrics
                visualizer.save_metrics(am_results, 'am_results.json')

            # Analizar confusiones PULSED->AM
            logger.info("Analizando confusiones PULSED_Air-Ground-MTI -> AM_combined...")
            trainer.analyze_confusion_am_pulsed(num_to_plot=10)
        
        # Calculate total execution time
        execution_time = time.time() - start_time
        logger.info(f"Total execution time: {execution_time:.2f} seconds")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == '__main__':
    sys.exit(main())