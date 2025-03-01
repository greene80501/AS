import tensorflow as tf
import numpy as np
import os
import time
from typing import List, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ASLRecognitionModel:
    def __init__(self, input_shape: int = 63*24, num_hidden_layers: int = 3):
        self.input_shape = input_shape
        self.num_classes = 26  # A-Z
        self.num_hidden_layers = num_hidden_layers
        self.model = None
        
        # Use multiple CPU cores efficiently
        self.num_cores = multiprocessing.cpu_count()
        tf.config.threading.set_intra_op_parallelism_threads(self.num_cores)
        tf.config.threading.set_inter_op_parallelism_threads(self.num_cores)
    
    def build_model(self) -> None:
        """Builds a more efficient neural network for ASL recognition."""
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=(self.input_shape,)),
            
            # Use smaller, more efficient layers
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            # Output layer
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logging.info("Model built successfully")

    def _process_file(self, file_data: Tuple[str, str]) -> Tuple[List[float], int]:
        """Process a single data file in parallel."""
        letter_file, data_dir = file_data
        try:
            with open(os.path.join(data_dir, letter_file), 'r') as f:
                content = f.readlines()
                results = []
                labels = []
                
                for line in content:
                    if line.strip() and not line.startswith('x'):
                        dats = line.strip().split(',')[:-1]
                        
                        if len(dats) == self.input_shape:
                            results.append([float(dat) for dat in dats if dat])
                            labels.append(ord(letter_file[0].upper()) - 65)
                
                return results, labels
        except Exception as e:
            logging.error(f"Error processing {letter_file}: {str(e)}")
            return [], []

    def load_data(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Loads and preprocesses the ASL data using parallel processing."""
        start_time = time.time()
        logging.info("Loading letter data...")
        
        # Prepare file list for parallel processing
        file_list = [(f, data_dir) for f in os.listdir(data_dir)]
        
        # Process files in parallel
        x = []
        y = []
        with ThreadPoolExecutor(max_workers=self.num_cores) as executor:
            results = executor.map(self._process_file, file_list)
            
            for result_x, result_y in results:
                x.extend(result_x)
                y.extend(result_y)
        
        x_array = np.array(x, dtype=np.float32)
        y_array = np.array(y)
        
        # Normalize input data
        x_array = (x_array - np.mean(x_array, axis=0)) / (np.std(x_array, axis=0) + 1e-7)
        
        logging.info(f"Data loaded in {time.time() - start_time:.2f} seconds")
        logging.info(f"Data shapes: X: {x_array.shape}, Y: {y_array.shape}")
        
        return x_array, y_array

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 64) -> None:
        """Trains the model with optimized parameters."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Convert labels to one-hot encoding
        y_encoded = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
        
        # Split the data manually into training and validation sets
        indices = np.random.permutation(len(x))
        split_idx = int(len(x) * 0.8)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        x_train, x_val = x[train_indices], x[val_indices]
        y_train, y_val = y_encoded[train_indices], y_encoded[val_indices]
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.0001
            )
        ]
        
        start_time = time.time()
        try:
            history = self.model.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
                shuffle=True
            )
            
            train_acc = history.history['accuracy'][-1]
            val_acc = history.history['val_accuracy'][-1]
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            
            logging.info(f"Training completed in {time.time() - start_time:.2f} seconds")
            logging.info(f"Final training accuracy: {train_acc:.4f}")
            logging.info(f"Final validation accuracy: {val_acc:.4f}")
            logging.info(f"Final training loss: {train_loss:.4f}")
            logging.info(f"Final validation loss: {val_loss:.4f}")
            
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise

    def predict(self, input_data: np.ndarray) -> List[str]:
        """Predicts ASL letters from input data."""
        if self.model is None:
            raise ValueError("No model loaded. Either train or load a model first.")
        
        try:
            predictions = self.model.predict(
                input_data,
                batch_size=64
            )
            return [chr(np.argmax(pred) + 65) for pred in predictions]
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise

    def save_model(self, path: str) -> None:
        """Saves the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        try:
            self.model.save(path)
            logging.info(f"Model saved successfully to {path}")
        except Exception as e:
            logging.error(f"Failed to save model: {str(e)}")
            raise

def main():
    model = ASLRecognitionModel()
    model.build_model()
    x, y = model.load_data("multiframe_csv_data")
    model.train(x, y)
    model.save_model("asl_model_Test.keras")

if __name__ == "__main__":
    main()