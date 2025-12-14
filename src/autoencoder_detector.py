import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib


class AnomalyDetector:
    """
    Autoencoder-based anomaly detection for EnerMod.
    Detects unusual energy consumption patterns in real-time.
    Optimized for edge deployment on Raspberry Pi.
    """

    def __init__(self, encoding_dim=8):
        self.encoding_dim = encoding_dim
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.threshold = None
        self.feature_columns = None

    def prepare_features(self, df, target_column='Global_active_power'):
        """
        Extract and prepare features for anomaly detection.
        Uses rolling statistics to capture recent behavior.
        """
        df = df.copy()

        # Basic features
        df['hour'] = df.index.hour
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

        # Rolling statistics (captures recent patterns)
        for window in [3, 6, 24]:  # 3h, 6h, 24h windows
            df[f'rolling_mean_{window}h'] = df[target_column].rolling(
                window=window, min_periods=1).mean()
            df[f'rolling_std_{window}h'] = df[target_column].rolling(
                window=window, min_periods=1).std().fillna(0)

        # Deviation from daily average
        df['daily_avg'] = df[target_column].rolling(
            window=24, min_periods=1).mean()
        df['deviation_from_avg'] = df[target_column] - df['daily_avg']

        # Rate of change
        df['rate_of_change'] = df[target_column].diff().fillna(0)

        # Feature columns
        feature_cols = [
            target_column,
            'hour',
            'is_weekend',
            'rolling_mean_3h',
            'rolling_std_3h',
            'rolling_mean_6h',
            'rolling_std_6h',
            'rolling_mean_24h',
            'rolling_std_24h',
            'deviation_from_avg',
            'rate_of_change'
        ]

        # Drop any remaining NaN
        df = df.dropna()

        self.feature_columns = feature_cols
        return df[feature_cols]

    def build_autoencoder(self, input_dim):
        """
        Build a lightweight autoencoder architecture.
        Designed to be efficient for edge deployment.
        """
        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(32, activation='relu')(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(16, activation='relu')(encoded)
        encoded = Dense(self.encoding_dim, activation='relu')(encoded)

        # Decoder
        decoded = Dense(16, activation='relu')(encoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(32, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)

        # Autoencoder model
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])

        return autoencoder

    def train(self, df, target_column='Global_active_power',
              epochs=50, batch_size=64, validation_split=0.2,
              contamination=0.02):
        """
        Train the autoencoder on normal energy consumption patterns.

        Parameters:
        - df: DataFrame with normal operating data
        - contamination: Expected proportion of anomalies (for threshold)
        """
        # Prepare features
        X = self.prepare_features(df, target_column)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train = X_scaled[:split_idx]
        X_val = X_scaled[split_idx:]

        # Build model
        self.model = self.build_autoencoder(X_train.shape[1])

        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10,
                                   restore_best_weights=True)

        # Train
        history = self.model.fit(
            X_train, X_train,  # Autoencoder reconstructs input
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )

        # Calculate reconstruction errors on training data
        X_train_pred = self.model.predict(X_train, verbose=0)
        train_mse = np.mean(np.power(X_train - X_train_pred, 2), axis=1)

        # Set threshold based on contamination rate
        self.threshold = np.percentile(train_mse, 100 * (1 - contamination))

        print(f"\nAnomaly detection threshold set to: {self.threshold:.6f}")

        return history

    def detect_anomaly(self, data_point):
        """
        Detect if a single data point is anomalous.

        Parameters:
        - data_point: DataFrame row or dict with required features

        Returns:
        - is_anomaly: Boolean
        - reconstruction_error: Float
        - confidence: Float (0-1, higher = more anomalous)
        """
        # Prepare features
        if isinstance(data_point, dict):
            data_point = pd.DataFrame([data_point])

        X = data_point[self.feature_columns].values
        X_scaled = self.scaler.transform(X)

        # Reconstruct
        X_pred = self.model.predict(X_scaled, verbose=0)

        # Calculate reconstruction error
        reconstruction_error = np.mean(np.power(X_scaled - X_pred, 2))

        # Determine if anomaly
        is_anomaly = reconstruction_error > self.threshold

        # Calculate confidence (normalized by threshold)
        confidence = min(reconstruction_error / self.threshold, 2.0) - 1.0
        confidence = max(0, confidence)  # Clip to [0, 1+]

        return {
            'is_anomaly': bool(is_anomaly),
            'reconstruction_error': float(reconstruction_error),
            'confidence': float(confidence),
            'threshold': float(self.threshold)
        }

    def detect_batch(self, df):
        """
        Detect anomalies in a batch of data.
        Returns array of anomaly flags and reconstruction errors.
        """
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)

        X_pred = self.model.predict(X_scaled, verbose=0)
        reconstruction_errors = np.mean(np.power(X_scaled - X_pred, 2), axis=1)

        anomalies = reconstruction_errors > self.threshold

        return anomalies, reconstruction_errors

    def convert_to_tflite(self, tflite_path='anomaly_detector.tflite'):
        """
        Convert the model to TensorFlow Lite for Raspberry Pi deployment.
        This creates a highly optimized model for edge computing.
        """
        import tensorflow as tf

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Optimize for edge deployment
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

        # Save the model
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        print(f"TFLite model saved to {tflite_path}")
        print(f"Model size: {len(tflite_model) / 1024:.2f} KB")

        return tflite_path

    def save_model(self, model_path='autoencoder.h5',
                   scaler_path='anomaly_scaler.pkl'):
        """Save model and preprocessing objects."""
        self.model.save(model_path)
        joblib.dump({
            'scaler': self.scaler,
            'threshold': self.threshold,
            'feature_columns': self.feature_columns,
            'encoding_dim': self.encoding_dim
        }, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler and config saved to {scaler_path}")

    def load_model(self, model_path='autoencoder.h5',
                   scaler_path='anomaly_scaler.pkl'):
        """Load trained model and preprocessing objects."""
        from tensorflow.keras.models import load_model
        self.model = load_model(model_path)

        config = joblib.load(scaler_path)
        self.scaler = config['scaler']
        self.threshold = config['threshold']
        self.feature_columns = config['feature_columns']
        self.encoding_dim = config['encoding_dim']
        print("Model and configuration loaded successfully")

    def visualize_reconstruction(self, df, n_samples=100):
        """
        Visualize reconstruction quality and detected anomalies.
        """
        anomalies, errors = self.detect_batch(df.head(n_samples))

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Plot reconstruction errors
        axes[0].plot(errors, label='Reconstruction Error', color='blue')
        axes[0].axhline(y=self.threshold, color='r', linestyle='--',
                        label='Threshold')
        axes[0].scatter(np.where(anomalies)[0], errors[anomalies],
                        color='red', s=50, label='Anomalies', zorder=5)
        axes[0].set_xlabel('Sample')
        axes[0].set_ylabel('Reconstruction Error')
        axes[0].set_title('Anomaly Detection Results')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot actual consumption with anomalies highlighted
        consumption = df['Global_active_power'].head(n_samples)
        axes[1].plot(consumption.values, label='Energy Consumption',
                     color='green')
        axes[1].scatter(np.where(anomalies)[0],
                        consumption.values[anomalies],
                        color='red', s=50, label='Detected Anomalies',
                        zorder=5)
        axes[1].set_xlabel('Sample')
        axes[1].set_ylabel('Power (kW)')
        axes[1].set_title('Energy Consumption with Anomalies')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('anomaly_detection_results.png', dpi=150)
        print("Visualization saved to anomaly_detection_results.png")
        return fig


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = AnomalyDetector(encoding_dim=8)

    # Train on normal data
    df = pd.read_csv('household_power_consumption.txt', sep=';',
                    parse_dates={'datetime': ['Date', 'Time']},
                    na_values=['?'])
    df = df.set_index('datetime')
    df = df.resample('H').mean()

    # Train the autoencoder
    history = detector.train(df, target_column='Global_active_power',
                             epochs=50, batch_size=64)

    # Detect anomaly in real-time
    result = detector.detect_anomaly(current_reading)
    if result['is_anomaly']:
        print(f"ANOMALY DETECTED! Confidence: {result['confidence']:.2%}")

    # Convert to TFLite for Raspberry Pi
    detector.convert_to_tflite('anomaly_detector.tflite')

    # Save models
    detector.save_model()

    print("Autoencoder Anomaly Detector ready!")
    print("This model is optimized for edge deployment on Raspberry Pi.")