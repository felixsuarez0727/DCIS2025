#!/usr/bin/env python3
"""
SNR Analysis Pipeline
Automated pipeline for analyzing model performance across SNR levels
"""

import os
import sys
import subprocess
import time
import logging
import argparse
from pathlib import Path

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('snr_pipeline')

def run_command(command, description):
    """
    Run a command and handle errors
    
    Args:
        command (str): Command to run
        description (str): Description of what the command does
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger = logging.getLogger('snr_pipeline')
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✓ {description} completed successfully")
        if result.stdout:
            logger.debug(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed")
        logger.error(f"Error: {e.stderr}")
        return False

def check_output_files(output_dir):
    """
    Check that all expected output files were created
    
    Args:
        output_dir (str): Output directory to check
    
    Returns:
        bool: True if all files exist, False otherwise
    """
    logger = logging.getLogger('snr_pipeline')
    
    expected_files = [
        'snr_summary.txt',
        'snr_analysis.json',
        'snr_performance.csv',
        'snr_performance.json',
        'snr_vs_accuracy.png',
        'latex_table.txt'
    ]
    
    missing_files = []
    for file in expected_files:
        file_path = os.path.join(output_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"Missing output files: {missing_files}")
        return False
    else:
        logger.info("✓ All expected output files created")
        return True

def print_summary(output_dir):
    """
    Print a summary of the analysis results
    
    Args:
        output_dir (str): Output directory with results
    """
    logger = logging.getLogger('snr_pipeline')
    
    # Read SNR summary
    summary_path = os.path.join(output_dir, 'snr_summary.txt')
    if os.path.exists(summary_path):
        logger.info("=" * 60)
        logger.info("SNR ANALYSIS SUMMARY")
        logger.info("=" * 60)
        
        with open(summary_path, 'r') as f:
            content = f.read()
            # Print first 20 lines of summary
            lines = content.split('\n')[:20]
            for line in lines:
                logger.info(line)
        
        logger.info("=" * 60)
    
    # Read performance results
    csv_path = os.path.join(output_dir, 'snr_performance.csv')
    if os.path.exists(csv_path):
        import pandas as pd
        try:
            df = pd.read_csv(csv_path)
            valid_results = df[df['accuracy'].notna()]
            
            if not valid_results.empty:
                logger.info("PERFORMANCE SUMMARY:")
                logger.info(f"  SNR levels evaluated: {len(valid_results)}")
                logger.info(f"  Best accuracy: {valid_results['accuracy'].max():.4f}")
                logger.info(f"  Worst accuracy: {valid_results['accuracy'].min():.4f}")
                logger.info(f"  Average accuracy: {valid_results['accuracy'].mean():.4f}")
                
                # Find best and worst SNR levels
                best_snr = valid_results.loc[valid_results['accuracy'].idxmax(), 'snr_level']
                worst_snr = valid_results.loc[valid_results['accuracy'].idxmin(), 'snr_level']
                logger.info(f"  Best SNR level: {best_snr} dB")
                logger.info(f"  Worst SNR level: {worst_snr} dB")
                
                # AM-specific analysis
                if 'am_accuracy' in valid_results.columns:
                    am_results = valid_results[valid_results['am_accuracy'].notna()]
                    if not am_results.empty:
                        logger.info(f"  AM signals - Best accuracy: {am_results['am_accuracy'].max():.4f}")
                        logger.info(f"  AM signals - Worst accuracy: {am_results['am_accuracy'].min():.4f}")
        except Exception as e:
            logger.warning(f"Could not read performance results: {e}")

def main():
    """Main pipeline function"""
    parser = argparse.ArgumentParser(description='SNR Analysis Pipeline')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to HDF5 dataset')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained Keras model')
    parser.add_argument('--output', type=str, default='results/snr_analysis',
                        help='Output directory for results')
    parser.add_argument('--samples_per_class', type=int, default=50,
                        help='Maximum samples per class to evaluate')
    parser.add_argument('--skip_structure_analysis', action='store_true',
                        help='Skip SNR structure analysis (if already done)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Start timing
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("SNR ANALYSIS PIPELINE STARTED")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Samples per class: {args.samples_per_class}")
    logger.info("=" * 60)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # STEP 1: Analyze SNR structure (unless skipped)
    if not args.skip_structure_analysis:
        logger.info("STEP 1: Analyzing SNR structure...")
        structure_cmd = f"python scripts/snr_structure_analysis.py --dataset {args.dataset} --output {args.output}"
        if not run_command(structure_cmd, "SNR structure analysis"):
            logger.error("Pipeline failed at step 1")
            return 1
    else:
        logger.info("STEP 1: Skipping SNR structure analysis")
    
    # STEP 2: Evaluate model performance across SNR levels
    logger.info("STEP 2: Evaluating model performance across SNR levels...")
    evaluation_cmd = f"python scripts/evaluate_model_snr.py --model {args.model} --dataset {args.dataset} --output {args.output} --samples_per_class {args.samples_per_class}"
    if not run_command(evaluation_cmd, "SNR model evaluation"):
        logger.error("Pipeline failed at step 2")
        return 1
    
    # STEP 3: Verify outputs
    logger.info("STEP 3: Verifying output files...")
    if not check_output_files(args.output):
        logger.warning("Some output files are missing, but continuing...")
    
    # STEP 4: Print summary
    logger.info("STEP 4: Generating summary...")
    print_summary(args.output)
    
    # Calculate total execution time
    execution_time = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info("SNR ANALYSIS PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.1f} minutes)")
    logger.info(f"Results saved to: {args.output}")
    logger.info("=" * 60)
    
    # List output files
    logger.info("Generated files:")
    for file in os.listdir(args.output):
        file_path = os.path.join(args.output, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            logger.info(f"  {file} ({size} bytes)")
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 