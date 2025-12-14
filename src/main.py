"""
Complete training pipeline for EnerMod AI models.
This script handles data loading, preprocessing, model training, and evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# Import your EnerMod models
from lstm_forecaster import EnergyForecaster
from autoencoder_detector import AnomalyDetector


class EnerModPipeline:
    """Complete pipeline for training both AI models."""

    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        self.df_train = None
        self.df_test = None
        self.forecaster = None
        self.detector = None

    def load_and_clean_data(self, data_path=None):
        """
        Load the UCI household power consumption dataset.
        """
        if data_path:
            self.data_path = data_path

        print(" Loading data from UCI dataset...")

        # Load the dataset
        # Format: Date;Time;Global_active_power;Global_reactive_power;Voltage;...
        self.df = pd.read_csv(
            self.data_path,
            sep=';',
            parse_dates=[['Date', 'Time']],
            na_values=['?'],
            low_memory=False
        )

        # Rename the combined datetime column
        self.df.rename(columns={'Date_Time': 'datetime'}, inplace=True)

        # Set datetime as index
        self.df = self.df.set_index('datetime')

        print(f" Loaded {len(self.df)} records")
        print(f"Date range: {self.df.index.min()} to {self.df.index.max()}")

        # Handle missing values
        print("\n Cleaning data...")
        missing_before = self.df.isnull().sum().sum()

        # Forward fill small gaps (< 5 minutes)
        self.df = self.df.ffill(limit=5)

        # Interpolate remaining gaps
        self.df = self.df.interpolate(method='linear', limit_direction='both')

        # Drop any remaining NaN rows
        self.df = self.df.dropna()

        missing_after = self.df.isnull().sum().sum()
        print(f" Handled {missing_before - missing_after} missing values")

        return self.df

    def add_weather_data(self, weather_df=None):
        """
        Add weather data to the dataset.
        If weather_df is provided, merge it. Otherwise, generate synthetic weather.
        """
        if weather_df is not None:
            print(" Merging weather data...")
            self.df = self.df.join(weather_df, how='left')
        else:
            print(" Generating synthetic weather features...")
            # Generate realistic synthetic weather patterns

            # Temperature: seasonal variation
            day_of_year = self.df.index.dayofyear
            self.df['temperature'] = 15 + 10 * np.sin(
                2 * np.pi * (day_of_year - 80) / 365
            ) + np.random.normal(0, 2, len(self.df))

            # Humidity: inversely related to temperature with noise
            self.df['humidity'] = 70 - 0.5 * (
                    self.df['temperature'] - 15
            ) + np.random.normal(0, 10, len(self.df))
            self.df['humidity'] = self.df['humidity'].clip(30, 100)

            # Wind speed: random with seasonal component
            self.df['wind_speed'] = 10 + 5 * np.sin(
                2 * np.pi * day_of_year / 365
            ) + np.random.exponential(3, len(self.df))

            # Precipitation: sparse events
            self.df['precipitation'] = np.random.choice(
                [0, 0, 0, 0, 0, 0.1, 0.5, 1.0, 2.0],
                size=len(self.df),
                p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.005, 0.005]
            )

            # UV index: depends on hour and season
            hour = self.df.index.hour
            self.df['uv_index'] = np.maximum(
                0,
                5 + 3 * np.sin(2 * np.pi * day_of_year / 365) *
                np.sin(np.pi * hour / 24)
            )

        print(" Weather features added")
        return self.df

    def resample_to_hourly(self):
        """
        Resample minute-level data to hourly averages.
        This reduces data volume and smooths out noise.
        """
        print(" Resampling to hourly data...")

        original_size = len(self.df)
        self.df = self.df.resample('H').mean()

        print(f" Resampled from {original_size} to {len(self.df)} records")
        return self.df

    def train_test_split(self, test_size=0.2):
        """
        Split data into training and testing sets.
        Uses temporal split (not random) to preserve time series nature.
        """
        split_idx = int(len(self.df) * (1 - test_size))

        self.df_train = self.df.iloc[:split_idx]
        self.df_test = self.df.iloc[split_idx:]

        print(f"\n Data split:")
        print(f"  Training: {len(self.df_train)} samples "
              f"({self.df_train.index.min()} to {self.df_train.index.max()})")
        print(f"  Testing: {len(self.df_test)} samples "
              f"({self.df_test.index.min()} to {self.df_test.index.max()})")

        return self.df_train, self.df_test

    def train_forecasting_model(self, epochs=50, batch_size=32):
        """
        Train the LSTM forecasting model.
        """
        print("\n" + "=" * 60)
        print(" TRAINING LSTM FORECASTING MODEL")
        print("=" * 60 + "\n")

        # Initialize forecaster
        from lstm_forecaster import EnergyForecaster
        self.forecaster = EnergyForecaster(lookback_hours=168)

        # Train
        history = self.forecaster.train(
            self.df_train,
            target_column='Global_active_power',
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )

        # Plot training history
        self._plot_training_history(history, 'LSTM Forecaster')

        # Save model
        self.forecaster.save_model(
            model_path='models/lstm_forecaster.h5',
            scaler_path='models/lstm_scalers.pkl'
        )

        print("\n LSTM model training complete!")
        return self.forecaster

    def train_anomaly_detector(self, epochs=50, batch_size=64):
        """
        Train the Autoencoder anomaly detection model.
        """
        print("\n" + "=" * 60)
        print(" TRAINING AUTOENCODER ANOMALY DETECTOR")
        print("=" * 60 + "\n")

        # Initialize detector
        from autoencoder_detector import AnomalyDetector
        self.detector = AnomalyDetector(encoding_dim=8)

        # Train on normal data (training set)
        history = self.detector.train(
            self.df_train,
            target_column='Global_active_power',
            epochs=epochs,
            batch_size=batch_size,
            contamination=0.02
        )

        # Plot training history
        self._plot_training_history(history, 'Autoencoder Detector')

        # Visualize results
        self.detector.visualize_reconstruction(
            self.df_test,
            n_samples=min(500, len(self.df_test))
        )

        # Save model
        self.detector.save_model(
            model_path='models/autoencoder.h5',
            scaler_path='models/anomaly_scaler.pkl'
        )

        # Convert to TFLite for Raspberry Pi
        self.detector.convert_to_tflite('models/anomaly_detector.tflite')

        print("\n Anomaly detector training complete!")
        return self.detector

    def evaluate_forecaster(self, n_forecasts=10):
        """
        Evaluate the forecasting model on test data.
        """
        print("\n" + "=" * 60)
        print(" EVALUATING LSTM FORECASTER")
        print("=" * 60 + "\n")

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        all_predictions = []
        all_actuals = []

        # Make multiple forecasts on test data
        step_size = len(self.df_test) // n_forecasts

        for i in range(n_forecasts):
            start_idx = i * step_size
            if start_idx + 168 + 24 > len(self.df_test):
                break

            # Get recent data for forecast
            recent_data = self.df_test.iloc[start_idx:start_idx + 168]

            # Get actual next 24 hours
            actual_24h = self.df_test.iloc[
                start_idx + 168:start_idx + 168 + 24
            ]['Global_active_power'].values

            # Predict
            forecast_24h = self.forecaster.predict_24h(recent_data)

            all_predictions.extend(forecast_24h[:len(actual_24h)])
            all_actuals.extend(actual_24h)

        # Calculate metrics
        mae = mean_absolute_error(all_actuals, all_predictions)
        rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
        r2 = r2_score(all_actuals, all_predictions)
        mape = np.mean(np.abs((np.array(all_actuals) - np.array(all_predictions)) /
                              np.array(all_actuals))) * 100

        print(f" Forecasting Metrics:")
        print(f"  MAE:  {mae:.4f} kW")
        print(f"  RMSE: {rmse:.4f} kW")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")

        # Visualize predictions
        plt.figure(figsize=(14, 6))
        sample_size = min(200, len(all_actuals))
        plt.plot(all_actuals[:sample_size], label='Actual',
                 linewidth=2, alpha=0.7)
        plt.plot(all_predictions[:sample_size], label='Predicted',
                 linewidth=2, alpha=0.7)
        plt.xlabel('Hour')
        plt.ylabel('Power Consumption (kW)')
        plt.title('LSTM Forecasting Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/forecast_evaluation.png', dpi=150)
        print("\n Forecast visualization saved to results/forecast_evaluation.png")

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }

    def evaluate_detector(self):
        """
        Evaluate the anomaly detector on test data.
        """
        print("\n" + "=" * 60)
        print(" EVALUATING ANOMALY DETECTOR")
        print("=" * 60 + "\n")

        # Detect anomalies in test set
        anomalies, errors = self.detector.detect_batch(self.df_test)

        anomaly_rate = np.mean(anomalies) * 100

        print(f" Detection Results:")
        print(f"  Total samples: {len(self.df_test)}")
        print(f"  Anomalies detected: {np.sum(anomalies)}")
        print(f"  Anomaly rate: {anomaly_rate:.2f}%")
        print(f"  Detection threshold: {self.detector.threshold:.6f}")
        print(f"  Mean reconstruction error: {np.mean(errors):.6f}")
        print(f"  Max reconstruction error: {np.max(errors):.6f}")

        return {
            'anomaly_count': int(np.sum(anomalies)),
            'anomaly_rate': anomaly_rate,
            'threshold': self.detector.threshold
        }

    def _plot_training_history(self, history, model_name):
        """Plot training and validation loss."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        ax.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{model_name} - Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"results/{model_name.lower().replace(' ', '_')}_history.png"
        plt.savefig(filename, dpi=150)
        print(f" Training history saved to {filename}")

    def generate_deployment_package(self):
        """
        Generate all files needed for Raspberry Pi deployment.
        """
        print("\n" + "=" * 60)
        print(" GENERATING DEPLOYMENT PACKAGE")
        print("=" * 60 + "\n")

        import os
        import json

        # Create directories
        os.makedirs('deployment', exist_ok=True)
        os.makedirs('deployment/models', exist_ok=True)

        # Copy models
        import shutil

        files_to_copy = [
            ('models/lstm_forecaster.h5', 'deployment/models/'),
            ('models/lstm_scalers.pkl', 'deployment/models/'),
            ('models/autoencoder.h5', 'deployment/models/'),
            ('models/anomaly_scaler.pkl', 'deployment/models/'),
            ('models/anomaly_detector.tflite', 'deployment/models/')
        ]

        for src, dst in files_to_copy:
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f" Copied {src}")

        # Generate config file
        config = {
            'peak_hours': [17, 18, 19, 20, 21],
            'off_peak_hours': [0, 1, 2, 3, 4, 5, 22, 23],
            'consumption_limit_kw': 5.0,
            'peak_buffer_percent': 0.9,
            'anomaly_action': 'alert',
            'enable_load_shifting': True,
            'enable_peak_shaving': True,
            'forecast_update_hour': 3,
            'measurement_interval_seconds': 60
        }

        with open('deployment/config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print(" Generated config.json")

        # Generate README
        readme = """
# EnerMod Deployment Package

## Contents
- `models/`: Trained AI models
  - `lstm_forecaster.h5`: LSTM forecasting model
  - `autoencoder.h5`: Anomaly detection model
  - `anomaly_detector.tflite`: TFLite optimized for Raspberry Pi
  - Scaler files for preprocessing

- `config.json`: System configuration

## Installation on Raspberry Pi

1. Install dependencies:
```bash
pip install tensorflow-lite numpy pandas
```

2. Copy all files to Raspberry Pi

3. Update config.json with your preferences

4. Run the moderator system

## Configuration
Edit `config.json` to customize:
- Peak/off-peak hours
- Consumption limits
- Anomaly handling behavior
- Load shifting preferences

## Support
Refer to the main project documentation for detailed setup instructions.
"""

        with open('deployment/README.md', 'w') as f:
            f.write(readme)
        print(" Generated README.md")

        print("\n Deployment package ready in 'deployment/' directory!")
        print(" Transfer this directory to your Raspberry Pi")


def main():
    """
    Main training pipeline execution.
    """
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║              EnerMod AI Training Pipeline                 ║
    ║        Intelligent Energy Management System               ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # Initialize pipeline
    pipeline = EnerModPipeline()

    # Step 1: Load and clean data
    data_path = '../data/household_power_consumption.txt'  # Update with your path
    pipeline.load_and_clean_data(data_path)

    # Step 2: Add weather features
    pipeline.add_weather_data()

    # Step 3: Resample to hourly
    pipeline.resample_to_hourly()

    # Step 4: Train-test split
    pipeline.train_test_split(test_size=0.2)

    # Step 5: Train forecasting model
    pipeline.train_forecasting_model(epochs=50, batch_size=32)

    # Step 6: Train anomaly detector
    pipeline.train_anomaly_detector(epochs=50, batch_size=64)

    # Step 7: Evaluate models
    forecast_metrics = pipeline.evaluate_forecaster(n_forecasts=10)
    detector_metrics = pipeline.evaluate_detector()

    # Step 8: Generate deployment package
    pipeline.generate_deployment_package()

    print("\n" + "=" * 60)
    print(" TRAINING PIPELINE COMPLETE!")
    print("=" * 60)
    print("\n Final Results Summary:")
    print(f"  Forecast MAPE: {forecast_metrics['mape']:.2f}%")
    print(f"  Anomaly Detection Rate: {detector_metrics['anomaly_rate']:.2f}%")
    print("\n Your EnerMod system is ready for deployment!")


if __name__ == "__main__":
    # Create necessary directories
    import os
    import torch
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    torch.device('cuda:0')
    print ("training on : " , torch.cuda.get_device_name(0))
    # Run pipeline
    main()