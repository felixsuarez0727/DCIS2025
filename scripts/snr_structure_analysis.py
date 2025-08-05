#!/usr/bin/env python3
"""
SNR Structure Analysis Script
Analyzes the HDF5 dataset structure to understand SNR encoding in keys
"""

import h5py
import numpy as np
import ast
import logging
import json
import os
from collections import defaultdict, Counter
import argparse

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('snr_analysis')

def extract_snr_from_key(key):
    """
    Extract SNR from a key string that represents a tuple
    
    Args:
        key (str): Key string like "('MODULATION', 'SIGNAL_TYPE', 'PARAM1', 'SNR_INDEX')"
    
    Returns:
        tuple: (snr_index, snr_dB) or (None, None) if parsing fails
    """
    try:
        # Parse the string as a tuple
        key_tuple = ast.literal_eval(key)
        
        if len(key_tuple) >= 4:
            snr_index_str = key_tuple[3]
            # Convert string to integer
            try:
                snr_index = int(snr_index_str)
                # Convert SNR index to dB value using cyclic mapping
                # Range: -20 to 18 dB in steps of 2
                snr_dB = -20 + (snr_index % 20) * 2
                return snr_index, snr_dB
            except ValueError:
                return None, None
        else:
            return None, None
    except (ValueError, SyntaxError, IndexError) as e:
        return None, None

def analyze_snr_structure(dataset_path, output_dir='results/snr_analysis'):
    """
    Analyze the SNR structure of the HDF5 dataset
    
    Args:
        dataset_path (str): Path to the HDF5 dataset
        output_dir (str): Directory to save results
    """
    logger = setup_logging()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Analyzing SNR structure of dataset: {dataset_path}")
    
    # Statistics containers
    snr_indices = []
    snr_dB_values = []
    modulation_types = []
    signal_types = []
    key_parsing_stats = {'successful': 0, 'failed': 0}
    
    try:
        with h5py.File(dataset_path, 'r') as f:
            # Get all keys
            all_keys = list(f.keys())
            logger.info(f"Total keys in dataset: {len(all_keys)}")
            
            # Analyze each key
            for i, key in enumerate(all_keys):
                if i % 10000 == 0:
                    logger.info(f"Processed {i}/{len(all_keys)} keys...")
                
                snr_index, snr_dB = extract_snr_from_key(key)
                
                if snr_index is not None:
                    snr_indices.append(snr_index)
                    snr_dB_values.append(snr_dB)
                    key_parsing_stats['successful'] += 1
                    
                    # Extract modulation and signal type
                    try:
                        key_tuple = ast.literal_eval(key)
                        if len(key_tuple) >= 2:
                            modulation_types.append(key_tuple[0])
                            signal_types.append(key_tuple[1])
                    except:
                        pass
                else:
                    key_parsing_stats['failed'] += 1
            
            logger.info(f"Key parsing completed: {key_parsing_stats['successful']} successful, {key_parsing_stats['failed']} failed")
    
    except Exception as e:
        logger.error(f"Error reading dataset: {str(e)}")
        raise
    
    # Analyze SNR distribution
    if snr_indices:
        unique_snr_indices = sorted(set(snr_indices))
        unique_snr_dB = sorted(set(snr_dB_values))
        snr_index_counts = Counter(snr_indices)
        snr_dB_counts = Counter(snr_dB_values)
        
        logger.info(f"Unique SNR indices found: {unique_snr_indices}")
        logger.info(f"Unique SNR dB values: {unique_snr_dB}")
        logger.info(f"SNR index distribution: {dict(snr_index_counts)}")
        logger.info(f"SNR dB distribution: {dict(snr_dB_counts)}")
        
        # Create SNR mapping
        snr_mapping = {}
        for snr_index in unique_snr_indices:
            snr_dB = -20 + (snr_index % 20) * 2
            snr_mapping[snr_index] = snr_dB
        
        # Generate comprehensive report
        report = {
            'dataset_path': dataset_path,
            'total_keys': len(all_keys),
            'successful_parsing': key_parsing_stats['successful'],
            'failed_parsing': key_parsing_stats['failed'],
            'snr_mapping': snr_mapping,
            'snr_index_distribution': dict(snr_index_counts),
            'snr_dB_distribution': dict(snr_dB_counts),
            'unique_snr_indices': unique_snr_indices,
            'unique_snr_dB': unique_snr_dB,
            'modulation_types': list(set(modulation_types)),
            'signal_types': list(set(signal_types)),
            'analysis_timestamp': str(np.datetime64('now'))
        }
        
        # Save detailed report
        report_path = os.path.join(output_dir, 'snr_summary.txt')
        with open(report_path, 'w') as f:
            f.write("SNR STRUCTURE ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset: {dataset_path}\n")
            f.write(f"Total keys analyzed: {len(all_keys)}\n")
            f.write(f"Successfully parsed: {key_parsing_stats['successful']}\n")
            f.write(f"Failed to parse: {key_parsing_stats['failed']}\n\n")
            
            f.write("SNR MAPPING (Index -> dB):\n")
            f.write("-" * 30 + "\n")
            for idx, db in sorted(snr_mapping.items()):
                f.write(f"  {idx:2d} -> {db:3d} dB\n")
            f.write("\n")
            
            f.write("SNR DISTRIBUTION:\n")
            f.write("-" * 20 + "\n")
            for db in sorted(snr_dB_counts.keys()):
                f.write(f"  {db:3d} dB: {snr_dB_counts[db]:6d} samples\n")
            f.write("\n")
            
            f.write("MODULATION TYPES:\n")
            f.write("-" * 18 + "\n")
            for mod in sorted(set(modulation_types)):
                f.write(f"  {mod}\n")
            f.write("\n")
            
            f.write("SIGNAL TYPES:\n")
            f.write("-" * 13 + "\n")
            for sig in sorted(set(signal_types)):
                f.write(f"  {sig}\n")
        
        # Save JSON report
        json_path = os.path.join(output_dir, 'snr_analysis.json')
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Analysis complete. Reports saved to {output_dir}")
        logger.info(f"SNR mapping: {snr_mapping}")
        
        return report
    
    else:
        logger.error("No valid SNR data found in the dataset")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Analyze SNR structure of HDF5 dataset')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to HDF5 dataset')
    parser.add_argument('--output', type=str, default='results/snr_analysis',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run analysis
    report = analyze_snr_structure(args.dataset, args.output)
    
    if report:
        print("\n" + "="*50)
        print("SNR ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Dataset: {args.dataset}")
        print(f"Total keys: {report['total_keys']}")
        print(f"SNR range: {min(report['unique_snr_dB'])} to {max(report['unique_snr_dB'])} dB")
        print(f"Reports saved to: {args.output}")
        print("="*50)
    else:
        print("SNR analysis failed. Check logs for details.")

if __name__ == '__main__':
    main() 