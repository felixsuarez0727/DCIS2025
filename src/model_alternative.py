import numpy as np
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class SimpleRadarSignalClassifier:
    def __init__(self, input_shape, num_classes, n_estimators=100, max_depth=10, 
                 min_samples_split=10, min_samples_leaf=5, class_weight='balanced'):
        """
        Simple Radar Signal Classifier using RandomForest (alternative to CNN)
        
        Args:
            input_shape (tuple): Shape of input data (not used directly)
            num_classes (int): Number of signal classes
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of the trees
            min_samples_split (int): Minimum samples required to split a node
            min_samples_leaf (int): Minimum samples required at a leaf node
            class_weight (str or dict): Class weights for imbalanced data
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.logger = logging.getLogger('src.model_alternative')
        
        # Initialize RandomForest model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,        
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features='sqrt',
            bootstrap=True,
            n_jobs=-1,
            random_state=42,
            class_weight=class_weight
        )
        
        # Initialize scaler
        self.scaler = StandardScaler()
        self.history = None
    
    def train(self, X_train, y_train, X_val, y_val, epochs=None, batch_size=None, 
              use_class_weights=None, use_early_stopping=None, verbose=1):
        """
        Train the model
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            X_val (numpy.ndarray): Validation features
            y_val (numpy.ndarray): Validation labels
            epochs (int): Not used
            batch_size (int): Not used
            use_class_weights (bool): Not used (class weights set in initialization)
            use_early_stopping (bool): Not used
            verbose (int): Verbosity mode
        
        Returns:
            dict: Training history-like object
        """
        # Reshape data for sklearn
        X_train_reshaped = self._reshape_for_sklearn(X_train)
        X_val_reshaped = self._reshape_for_sklearn(X_val)
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        
        if verbose:
            self.logger.info(f"Training RandomForest model on {X_train_scaled.shape[0]} samples...")
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        
        # Calculate validation metrics
        X_val_scaled = self.scaler.transform(X_val_reshaped)
        val_pred = self.model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_pred)
        
        if verbose:
            self.logger.info(f"Training accuracy: {train_acc:.4f}")
            self.logger.info(f"Validation accuracy: {val_acc:.4f}")
            self.logger.info("\nClassification Report:")
            report = classification_report(y_val, val_pred)
            self.logger.info(f"\n{report}")
            
            cm = confusion_matrix(y_val, val_pred)
            self.logger.info("\nConfusion Matrix:")
            self.logger.info(f"\n{cm}")
        
        # Create history-like object for API compatibility with TensorFlow model
        self.history = {
            'accuracy': [train_acc],
            'val_accuracy': [val_acc],
            'loss': [0],  # Placeholder
            'val_loss': [0]  # Placeholder
        }
        
        return self.history
    
    def _reshape_for_sklearn(self, X):
        """
        Reshape input data to 2D for sklearn models
        
        Args:
            X (numpy.ndarray): Input data (samples, height, width, channels)
        
        Returns:
            numpy.ndarray: Reshaped data (samples, height*width*channels)
        """
        sample_count = X.shape[0]
        flattened_shape = np.prod(X.shape[1:])
        return X.reshape(sample_count, flattened_shape)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
        
        Returns:
            tuple: (0 as placeholder for loss, test accuracy)
        """
        self.logger.info("Evaluating model on test data")
        X_test_reshaped = self._reshape_for_sklearn(X_test)
        X_test_scaled = self.scaler.transform(X_test_reshaped)
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Print detailed classification report
        self.logger.info("\nTest Classification Report:")
        report = classification_report(y_test, y_pred)
        self.logger.info(f"\n{report}")
        
        # Print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.logger.info("\nConfusion Matrix:")
        self.logger.info(f"\n{cm}")
        
        # Return tuple with 0 as placeholder for loss to match keras interface
        return 0, accuracy
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (numpy.ndarray): Input features
        
        Returns:
            numpy.ndarray: Predicted class probabilities
        """
        X_reshaped = self._reshape_for_sklearn(X)
        X_scaled = self.scaler.transform(X_reshaped)
        
        # Get probabilities
        proba = self.model.predict_proba(X_scaled)
        return proba
    
    def save_model(self, filepath):
        """
        Save model to disk
        
        Args:
            filepath (str): Path to save the model
        """
        joblib.dump((self.model, self.scaler), filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load model from disk
        
        Args:
            filepath (str): Path to load the model from
        """
        self.model, self.scaler = joblib.load(filepath)
        self.logger.info(f"Model loaded from {filepath}")