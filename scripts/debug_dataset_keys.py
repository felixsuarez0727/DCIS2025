#!/usr/bin/env python3
"""
Debug Dataset Keys Script
Examine the actual structure of keys in the HDF5 dataset
"""

import h5py
import ast
import logging

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('debug_keys')

def debug_dataset_keys(dataset_path):
    """
    Debug the structure of keys in the HDF5 dataset
    
    Args:
        dataset_path (str): Path to the HDF5 dataset
    """
    logger = setup_logging()
    
    logger.info(f"Debugging keys in dataset: {dataset_path}")
    
    try:
        with h5py.File(dataset_path, 'r') as f:
            # Get all keys
            all_keys = list(f.keys())
            logger.info(f"Total keys in dataset: {len(all_keys)}")
            
            # Examine first few keys
            logger.info("First 10 keys:")
            for i, key in enumerate(all_keys[:10]):
                logger.info(f"  Key {i}: {key} (type: {type(key)})")
                
                # Try to parse as tuple
                try:
                    if isinstance(key, str):
                        parsed = ast.literal_eval(key)
                        logger.info(f"    Parsed as: {parsed} (type: {type(parsed)})")
                        
                        if isinstance(parsed, tuple):
                            logger.info(f"    Tuple length: {len(parsed)}")
                            for j, item in enumerate(parsed):
                                logger.info(f"      Item {j}: {item} (type: {type(item)})")
                    else:
                        logger.info(f"    Not a string, cannot parse with ast.literal_eval")
                except Exception as e:
                    logger.info(f"    Parse error: {e}")
            
            # Look for patterns
            logger.info("\nAnalyzing key patterns...")
            
            # Check if keys are strings
            string_keys = [k for k in all_keys if isinstance(k, str)]
            logger.info(f"String keys: {len(string_keys)}")
            
            # Check if keys contain parentheses
            paren_keys = [k for k in string_keys if '(' in k and ')' in k]
            logger.info(f"Keys with parentheses: {len(paren_keys)}")
            
            if paren_keys:
                logger.info("First 5 keys with parentheses:")
                for i, key in enumerate(paren_keys[:5]):
                    logger.info(f"  {i}: {key}")
                    
                    try:
                        parsed = ast.literal_eval(key)
                        logger.info(f"    Parsed: {parsed}")
                        if isinstance(parsed, tuple) and len(parsed) >= 4:
                            logger.info(f"    SNR index (4th element): {parsed[3]} (type: {type(parsed[3])})")
                    except Exception as e:
                        logger.info(f"    Parse error: {e}")
            
            # Check for non-string keys
            non_string_keys = [k for k in all_keys if not isinstance(k, str)]
            if non_string_keys:
                logger.info(f"Non-string keys found: {len(non_string_keys)}")
                logger.info(f"First 5 non-string keys: {non_string_keys[:5]}")
    
    except Exception as e:
        logger.error(f"Error reading dataset: {str(e)}")
        raise

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python debug_dataset_keys.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    debug_dataset_keys(dataset_path)

if __name__ == '__main__':
    main() 