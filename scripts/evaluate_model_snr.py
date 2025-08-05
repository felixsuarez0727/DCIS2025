#!/usr/bin/env python3
"""
SNR Model Evaluator Script
Evaluates trained model performance across different SNR levels
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import logging
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader

class SNREvaluator:
    def __init__(self, model_path, dataset_path, output_dir='results/snr_analysis'):
        """
        Initialize SNR Evaluator
        
        Args:
            model_path (str): Path to trained Keras model
            dataset_path (str): Path to HDF5 dataset
            output_dir (str): Directory to save results
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load model
        self.model = self._load_model()
        
        # Initialize data loader
        self.data_loader = DataLoader(
            train_dataset_path=dataset_path,
            data_percentage=1.0,
            samples_per_class=100,  # Will be overridden per SNR level
            combine_am=True
        )
        
        # Define SNR levels to evaluate
        self.snr_levels = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        
        # Results storage
        self.results = []
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger('snr_evaluator')
    
    def _load_model(self):
        """Load the trained Keras model"""
        try:
            self.logger.info(f"Loading model from {self.model_path}")
            model = keras.models.load_model(self.model_path)
            self.logger.info("Model loaded successfully")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def evaluate_snr_level(self, snr_level, max_samples_per_class=50):
        """
        Evaluate model performance for a specific SNR level
        
        Args:
            snr_level (int): SNR level in dB
            max_samples_per_class (int): Maximum samples per class to evaluate
        
        Returns:
            dict: Performance metrics for the SNR level
        """
        self.logger.info(f"Evaluating SNR level: {snr_level} dB")
        
        try:
            # Load data for this SNR level
            X, y = self.data_loader.load_data_by_snr(snr_level, max_samples_per_class)
            
            if len(X) == 0:
                self.logger.warning(f"No data available for SNR {snr_level} dB")
                return {
                    'snr_level': snr_level,
                    'num_samples': 0,
                    'accuracy': None,
                    'precision': None,
                    'recall': None,
                    'f1_score': None,
                    'loss': None,
                    'am_accuracy': None,
                    'am_recall': None,
                    'error': 'No data available'
                }
            
            # Encode labels
            label_encoder = LabelEncoder()
            if not hasattr(label_encoder, 'classes_'):
                label_encoder.fit(y)
            y_encoded = label_encoder.transform(y)
            
            # Make predictions
            try:
                predictions = self.model.predict(X, verbose=0)
                predicted_classes = np.argmax(predictions, axis=1)
                
                # Calculate metrics
                accuracy = accuracy_score(y_encoded, predicted_classes)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_encoded, predicted_classes, average='weighted'
                )
                
                # Calculate loss
                loss = self.model.evaluate(X, y_encoded, verbose=0)[0]
                
                # Calculate AM-specific metrics
                am_accuracy, am_recall = self._calculate_am_metrics(y, predicted_classes, label_encoder)
                
                metrics = {
                    'snr_level': snr_level,
                    'num_samples': len(X),
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'loss': float(loss),
                    'am_accuracy': float(am_accuracy) if am_accuracy is not None else None,
                    'am_recall': float(am_recall) if am_recall is not None else None,
                    'error': None
                }
                
                am_accuracy_str = f"{am_accuracy:.4f}" if am_accuracy is not None else "N/A"
                self.logger.info(f"SNR {snr_level} dB - Accuracy: {accuracy:.4f}, AM Accuracy: {am_accuracy_str}")
                
                return metrics
                
            except Exception as e:
                self.logger.error(f"Error during model evaluation for SNR {snr_level} dB: {str(e)}")
                return {
                    'snr_level': snr_level,
                    'num_samples': len(X),
                    'accuracy': None,
                    'precision': None,
                    'recall': None,
                    'f1_score': None,
                    'loss': None,
                    'am_accuracy': None,
                    'am_recall': None,
                    'error': str(e)
                }
                
        except Exception as e:
            self.logger.error(f"Error loading data for SNR {snr_level} dB: {str(e)}")
            return {
                'snr_level': snr_level,
                'num_samples': 0,
                'accuracy': None,
                'precision': None,
                'recall': None,
                'f1_score': None,
                'loss': None,
                'am_accuracy': None,
                'am_recall': None,
                'error': str(e)
            }
    
    def _calculate_am_metrics(self, true_labels, predicted_classes, label_encoder):
        """
        Calculate AM-specific metrics
        
        Args:
            true_labels (array): True labels
            predicted_classes (array): Predicted class indices
            label_encoder (LabelEncoder): Fitted label encoder
        
        Returns:
            tuple: (am_accuracy, am_recall)
        """
        try:
            # Find AM_combined class index
            am_class_name = 'AM_combined'
            if am_class_name in label_encoder.classes_:
                am_class_idx = label_encoder.transform([am_class_name])[0]
                
                # Get AM samples
                am_mask = label_encoder.transform(true_labels) == am_class_idx
                
                if np.sum(am_mask) > 0:
                    am_true = label_encoder.transform(true_labels)[am_mask]
                    am_pred = predicted_classes[am_mask]
                    
                    am_accuracy = accuracy_score(am_true, am_pred)
                    am_recall = np.sum((am_true == am_class_idx) & (am_pred == am_class_idx)) / np.sum(am_true == am_class_idx)
                    
                    return am_accuracy, am_recall
            
            return None, None
            
        except Exception as e:
            self.logger.warning(f"Error calculating AM metrics: {str(e)}")
            return None, None
    
    def evaluate_all_snr_levels(self, max_samples_per_class=50):
        """
        Evaluate model performance across all SNR levels
        
        Args:
            max_samples_per_class (int): Maximum samples per class to evaluate
        """
        self.logger.info("Starting evaluation across all SNR levels")
        
        for snr_level in self.snr_levels:
            result = self.evaluate_snr_level(snr_level, max_samples_per_class)
            self.results.append(result)
        
        # Save results
        self._save_results()
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Generate LaTeX table
        self._generate_latex_table()
        
        self.logger.info("SNR evaluation completed")
    
    def _save_results(self):
        """Save results to CSV and JSON files"""
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save CSV
        csv_path = os.path.join(self.output_dir, 'snr_performance.csv')
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Results saved to {csv_path}")
        
        # Save JSON
        json_path = os.path.join(self.output_dir, 'snr_performance.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.logger.info(f"Results saved to {json_path}")
    
    def _generate_visualizations(self):
        """Generate SNR vs Accuracy visualization"""
        # Filter out results with errors
        valid_results = [r for r in self.results if r['error'] is None and r['accuracy'] is not None]
        
        if not valid_results:
            self.logger.warning("No valid results for visualization")
            return
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data
        snr_levels = [r['snr_level'] for r in valid_results]
        accuracies = [r['accuracy'] for r in valid_results]
        am_accuracies = [r['am_accuracy'] for r in valid_results if r['am_accuracy'] is not None]
        am_snr_levels = [r['snr_level'] for r in valid_results if r['am_accuracy'] is not None]
        
        # Plot overall accuracy
        ax1.plot(snr_levels, accuracies, 'bo-', linewidth=2, markersize=8, label='Overall Accuracy')
        ax1.set_xlabel('SNR (dB)', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Model Performance vs SNR', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add trend line
        if len(snr_levels) > 1:
            z = np.polyfit(snr_levels, accuracies, 1)
            p = np.poly1d(z)
            ax1.plot(snr_levels, p(snr_levels), "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.4f})')
            ax1.legend()
        
        # Plot AM accuracy if available
        if am_accuracies:
            ax2.plot(am_snr_levels, am_accuracies, 'go-', linewidth=2, markersize=8, label='AM Accuracy')
            ax2.set_xlabel('SNR (dB)', fontsize=12)
            ax2.set_ylabel('Accuracy', fontsize=12)
            ax2.set_title('AM Signal Performance vs SNR', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add trend line for AM
            if len(am_snr_levels) > 1:
                z_am = np.polyfit(am_snr_levels, am_accuracies, 1)
                p_am = np.poly1d(z_am)
                ax2.plot(am_snr_levels, p_am(am_snr_levels), "r--", alpha=0.8, label=f'Trend (slope: {z_am[0]:.4f})')
                ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'snr_vs_accuracy.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Visualization saved to {plot_path}")
        
        plt.close()
    
    def _generate_latex_table(self):
        """Generate LaTeX table for paper"""
        # Filter out results with errors
        valid_results = [r for r in self.results if r['error'] is None and r['accuracy'] is not None]
        
        if not valid_results:
            self.logger.warning("No valid results for LaTeX table")
            return
        
        latex_content = []
        latex_content.append("\\begin{table}[h]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Model Performance Across SNR Levels}")
        latex_content.append("\\label{tab:snr_performance}")
        latex_content.append("\\begin{tabular}{|c|c|c|c|c|c|}")
        latex_content.append("\\hline")
        latex_content.append("SNR (dB) & Accuracy & Precision & Recall & F1-Score & AM Accuracy \\\\")
        latex_content.append("\\hline")
        
        for result in valid_results:
            snr = result['snr_level']
            acc = result['accuracy']
            prec = result['precision']
            rec = result['recall']
            f1 = result['f1_score']
            am_acc = result['am_accuracy'] if result['am_accuracy'] is not None else 'N/A'
            
            if am_acc == 'N/A':
                latex_content.append(f"{snr:3d} & {acc:.4f} & {prec:.4f} & {rec:.4f} & {f1:.4f} & {am_acc} \\\\")
            else:
                latex_content.append(f"{snr:3d} & {acc:.4f} & {prec:.4f} & {rec:.4f} & {f1:.4f} & {am_acc:.4f} \\\\")
        
        latex_content.append("\\hline")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        
        # Save LaTeX table
        latex_path = os.path.join(self.output_dir, 'latex_table.txt')
        with open(latex_path, 'w') as f:
            f.write('\n'.join(latex_content))
        
        self.logger.info(f"LaTeX table saved to {latex_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Evaluate model performance across SNR levels')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained Keras model')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to HDF5 dataset')
    parser.add_argument('--output', type=str, default='results/snr_analysis',
                        help='Output directory for results')
    parser.add_argument('--samples_per_class', type=int, default=50,
                        help='Maximum samples per class to evaluate')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = SNREvaluator(args.model, args.dataset, args.output)
    
    # Run evaluation
    evaluator.evaluate_all_snr_levels(args.samples_per_class)
    
    print("\n" + "="*50)
    print("SNR EVALUATION COMPLETED SUCCESSFULLY")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Results saved to: {args.output}")
    print("="*50)

if __name__ == '__main__':
    main() 