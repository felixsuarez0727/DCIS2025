import h5py
import numpy as np
import logging
import os
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import scipy.signal
import pywt

class DataLoader:
    def __init__(self, train_dataset_path, test_dataset_path=None, data_percentage=1.0, 
                 stratified=True, samples_per_class=25, combine_am=True, random_state=42):
        """
        Initialize DataLoader with separate train and test datasets
        
        Args:
            train_dataset_path (str): Path to training HDF5 dataset
            test_dataset_path (str): Path to testing HDF5 dataset (if None, will split train_dataset)
            data_percentage (float): Percentage of data to use (0.0 to 1.0)
            stratified (bool): Whether to use stratified sampling
            samples_per_class (int): Number of samples per class to select
            combine_am (bool): Whether to combine AM-related modulations (AM-DSB, AM-SSB, ASK) into one class
            random_state (int): Random seed for reproducibility
        """
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.data_percentage = data_percentage
        self.stratified = stratified
        self.samples_per_class = samples_per_class
        self.combine_am = combine_am
        self.random_state = random_state
        
        # Initialize data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_encoded = None
        self.y_test_encoded = None
        self.class_names = None
        self.label_encoder = LabelEncoder()
        
        # SNR-related attributes
        self.snr_mapping = self._create_snr_mapping()
        
        # Configure logging
        self.logger = logging.getLogger('src.data_loader')
        
    def _create_snr_mapping(self):
        """
        Create SNR mapping from index to dB values
        Range: -20 to 18 dB in steps of 2
        
        Returns:
            dict: Mapping from SNR index to dB value
        """
        snr_mapping = {}
        for i in range(20):  # 0 to 19
            snr_dB = -20 + (i % 20) * 2
            snr_mapping[i] = snr_dB
        return snr_mapping
    
    def _extract_snr_from_key(self, key):
        """
        Extract SNR from a key string that represents a tuple
        
        Args:
            key (str): Key string like "('MODULATION', 'SIGNAL_TYPE', 'PARAM1', 'SNR_INDEX')"
        
        Returns:
            int: SNR dB value or None if parsing fails
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
                    snr_dB = -20 + (snr_index % 20) * 2
                    return snr_dB
                except ValueError:
                    return None
            else:
                return None
        except (ValueError, SyntaxError, IndexError) as e:
            return None
    
    def get_available_snr_levels(self):
        """
        Get list of available SNR levels in the dataset
        
        Returns:
            list: Available SNR levels in dB
        """
        if not hasattr(self, '_available_snr_levels'):
            self._available_snr_levels = sorted(set(self.snr_mapping.values()))
        return self._available_snr_levels
        
    def _group_modulations(self, labels):
        """
        Group AM signals if combine_am is True
        
        Args:
            labels (numpy.ndarray): Original labels
            
        Returns:
            numpy.ndarray: Modified labels with grouped AM signals
        """
        if not self.combine_am:
            return labels
            
        # Create a copy of labels to modify
        new_labels = labels.copy()
        
        # Define AM-related signal types
        am_types = ['AM-DSB', 'AM-SSB', 'ASK']
        
        # Identify AM-related indices
        for i, label in enumerate(new_labels):
            for am_type in am_types:
                if am_type in label:
                    new_labels[i] = 'AM_combined'
                    break
        
        return new_labels
    
    def _process_signal(self, signal):
        """
        Process a single signal with normalization, spectrogram, wavelet, and frequency features.
        
        Args:
            signal (numpy.ndarray): Raw signal data
        
        Returns:
            numpy.ndarray: Feature vector (spectrogram + wavelet + frequency)
        """
        # Ensure signal is a numpy array
        if np.isscalar(signal):
            signal = np.array([signal])
        
        # Normalize signal
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        
        # Spectrogram
        f, t, Sxx = scipy.signal.spectrogram(signal.squeeze(), nperseg=64, noverlap=32)
        Sxx = np.abs(Sxx)
        Sxx = (Sxx - np.mean(Sxx)) / (np.std(Sxx) + 1e-8)
        Sxx = Sxx[..., np.newaxis]  # (freq, time, 1)
        
        # Wavelet features (Daubechies 4, level 3)
        coeffs = pywt.wavedec(signal.squeeze(), 'db4', level=3)
        wavelet_feats = np.concatenate([c.flatten() for c in coeffs])
        
        # Frequency: FFT and statistics
        fft_vals = np.abs(np.fft.fft(signal.squeeze()))
        fft_vals = fft_vals[:len(fft_vals) // 2]  # Only positive frequencies
        fft_stats = np.array([
            np.mean(fft_vals),
            np.std(fft_vals),
            np.max(fft_vals),
            np.median(fft_vals),
            np.sum(fft_vals > 0.5 * np.max(fft_vals)),  # Number of high peaks
            np.sum(fft_vals),
        ])
        
        # Concatenate everything into a single feature vector
        # Flatten spectrogram
        spec_flat = Sxx.flatten()
        features = np.concatenate([spec_flat, wavelet_feats, fft_stats])
        
        return features

    def _load_from_hdf5(self, file_path):
        """
        Load data from an HDF5 file
        
        Args:
            file_path (str): Path to HDF5 file
            
        Returns:
            tuple: (X, y) - Features and labels
        """
        self.logger.info(f"Loading data from {file_path}")
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Get all keys
                all_keys = list(f.keys())
                self.logger.info(f"Total keys in the HDF5 file: {len(all_keys)}")
                
                # Group keys by signal type
                class_map = {}
                for k in all_keys:
                    # Extract signal type from key
                    if isinstance(k, tuple) or (isinstance(k, str) and ('(' in k)):
                        # Convert string tuple to actual tuple if necessary
                        if isinstance(k, str):
                            import ast
                            try:
                                k_tuple = ast.literal_eval(k)
                            except:
                                self.logger.warning(f"Could not parse key: {k}")
                                continue
                        else:
                            k_tuple = k
                        signal_type = f"{k_tuple[0]}_{k_tuple[1]}"
                    else:
                        signal_type = k
                        
                    if signal_type not in class_map:
                        class_map[signal_type] = []
                        
                    class_map[signal_type].append(k)
                
                # Select samples per class
                X_selected = []
                y_selected = []
                
                for signal_type, key_list in class_map.items():
                    # Select up to samples_per_class samples per class
                    if len(key_list) > 0:
                        # Use deterministic random selection
                        np.random.seed(self.random_state)
                        selected_keys = np.random.choice(
                            key_list, 
                            size=min(len(key_list), self.samples_per_class), 
                            replace=False
                        )
                        
                        for k in selected_keys:
                            signal = f[k][()]
                            processed_signal = self._process_signal(signal)
                            X_selected.append(processed_signal)
                            y_selected.append(signal_type)
                
                X = np.array(X_selected)
                y = np.array(y_selected)
                
                # Group AM signals if requested
                y = self._group_modulations(y)
                
                # Log class distribution
                unique, counts = np.unique(y, return_counts=True)
                self.logger.info(f"Class distribution: {dict(zip(unique, counts))}")
                
                return X, y
                
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def load_data(self):
        """
        Load and preprocess data from HDF5 files
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Load training data
        X, y = self._load_from_hdf5(self.train_dataset_path)
        
        # Save class names
        self.class_names = np.unique(y)
        
        # First split: 70% train, 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=0.3,  # 30% for temp
            random_state=self.random_state,
            stratify=y if self.stratified else None
        )
        
        # Second split: 50-50 of the 30% to get 15-15 for val and test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,  # 50% of the 30% = 15% of total
            random_state=self.random_state,
            stratify=y_temp if self.stratified else None
        )
        
        # Store data in class attributes
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        # Log the shapes
        self.logger.info("Splitting data into train, validation and test sets")
        self.logger.info(f"X_train shape: {X_train.shape}")
        self.logger.info(f"X_val shape: {X_val.shape}")
        self.logger.info(f"X_test shape: {X_test.shape}")
        self.logger.info(f"Number of classes: {len(self.class_names)}")
        self.logger.info(f"Class names: {self.class_names}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_class_names(self):
        """
        Get list of class names
        
        Returns:
            list: Class names
        """
        return self.class_names
    
    def load_data_by_snr(self, snr_level, max_samples_per_class=100):
        """
        Load data for a specific SNR level
        
        Args:
            snr_level (int): SNR level in dB
            max_samples_per_class (int): Maximum samples per class to load
        
        Returns:
            tuple: (X, y) - Features and labels for the specified SNR level
        """
        self.logger.info(f"Loading data for SNR level: {snr_level} dB")
        
        try:
            with h5py.File(self.train_dataset_path, 'r') as f:
                # Get all keys
                all_keys = list(f.keys())
                
                # Filter keys by SNR level
                filtered_keys = []
                for key in all_keys:
                    key_snr = self._extract_snr_from_key(key)
                    if key_snr == snr_level:
                        filtered_keys.append(key)
                
                self.logger.info(f"Found {len(filtered_keys)} samples for SNR {snr_level} dB")
                
                if len(filtered_keys) == 0:
                    self.logger.warning(f"No samples found for SNR level {snr_level} dB")
                    return np.array([]), np.array([])
                
                # Group keys by signal type
                class_map = {}
                for k in filtered_keys:
                    # Extract signal type from key
                    if isinstance(k, tuple) or (isinstance(k, str) and ('(' in k)):
                        # Convert string tuple to actual tuple if necessary
                        if isinstance(k, str):
                            try:
                                k_tuple = ast.literal_eval(k)
                            except:
                                self.logger.warning(f"Could not parse key: {k}")
                                continue
                        else:
                            k_tuple = k
                        signal_type = f"{k_tuple[0]}_{k_tuple[1]}"
                    else:
                        signal_type = k
                        
                    if signal_type not in class_map:
                        class_map[signal_type] = []
                        
                    class_map[signal_type].append(k)
                
                # Select samples per class
                X_selected = []
                y_selected = []
                
                for signal_type, key_list in class_map.items():
                    # Select up to max_samples_per_class samples per class
                    if len(key_list) > 0:
                        # Use deterministic random selection
                        np.random.seed(self.random_state)
                        selected_keys = np.random.choice(
                            key_list, 
                            size=min(len(key_list), max_samples_per_class), 
                            replace=False
                        )
                        
                        for k in selected_keys:
                            signal = f[k][()]
                            processed_signal = self._process_signal(signal)
                            X_selected.append(processed_signal)
                            y_selected.append(signal_type)
                
                X = np.array(X_selected)
                y = np.array(y_selected)
                
                # Group AM signals if requested
                y = self._group_modulations(y)
                
                # Log class distribution
                unique, counts = np.unique(y, return_counts=True)
                self.logger.info(f"Class distribution for SNR {snr_level} dB: {dict(zip(unique, counts))}")
                
                return X, y
                
        except Exception as e:
            self.logger.error(f"Error loading data for SNR {snr_level} dB: {str(e)}")
            raise