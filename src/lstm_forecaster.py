import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime
import joblib


class EnergyForecaster:
    """
    LSTM-based energy consumption forecasting model for EnerMod.
    Predicts 24-hour energy demand based on historical patterns and weather data.
    """

    def __init__(self, lookback_hours=168):  # 1 week of history
        self.lookback_hours = lookback_hours
        self.model = None
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None

    def prepare_data(self, df, target_column='Global_active_power'):
        """
        Prepare and engineer features from the raw dataset.

        Parameters:
        - df: DataFrame with datetime index
        - target_column: Column to predict
        """
        df = df.copy()

        # Create time-based features
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear

        # Cyclic encoding for hour and day of week
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # If weather data is available, include it
        weather_features = ['temperature', 'humidity', 'wind_speed',
                            'precipitation', 'uv_index']
        available_weather = [col for col in weather_features if col in df.columns]

        # Define feature columns (exclude target)
        feature_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                        'is_weekend', 'month'] + available_weather

        # Add lagged features (previous hour values)
        for lag in [1, 2, 3, 24, 168]:  # 1h, 2h, 3h, 1day, 1week
            if len(df) > lag:
                df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
                feature_cols.append(f'{target_column}_lag_{lag}')

        # Drop rows with NaN (from lagging)
        df = df.dropna()

        self.feature_columns = feature_cols
        return df

    def create_sequences(self, X, y, lookback):
        """
        Create sequences for LSTM training.

        Parameters:
        - X: Feature array
        - y: Target array
        - lookback: Number of timesteps to look back
        """
        X_seq, y_seq = [], []

        for i in range(lookback, len(X)):
            X_seq.append(X[i - lookback:i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    def build_model(self, input_shape):
        """
        Build the LSTM neural network architecture.
        """
        model = Sequential([
            LSTM(128, activation='tanh', return_sequences=True,
                 input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, activation='tanh', return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='tanh'),
            Dropout(0.2),
            Dense(24, activation='relu'),  # 24 hours ahead
            Dense(24)  # Output layer for 24-hour forecast
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train(self, df, target_column='Global_active_power',
              epochs=100, batch_size=32, validation_split=0.2):
        """
        Train the LSTM model on prepared data.
        """
        # Prepare features
        df_prepared = self.prepare_data(df, target_column)

        # Extract features and target
        X = df_prepared[self.feature_columns].values
        y = df_prepared[target_column].values.reshape(-1, 1)

        # Scale features and target
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled,
                                             self.lookback_hours)

        # For 24-hour forecast, we need to reshape y
        # We'll predict the next 24 hours, so we need to create that target
        y_forecast = []
        for i in range(len(y_seq) - 24):
            y_forecast.append(y_scaled[i:i + 24].flatten())

        y_forecast = np.array(y_forecast)
        X_seq = X_seq[:len(y_forecast)]

        # Split into train and validation
        split_idx = int(len(X_seq) * (1 - validation_split))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_forecast[:split_idx], y_forecast[split_idx:]

        # Build model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))

        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=15,
                                   restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=5, min_lr=0.00001)

        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        return history

    def predict_24h(self, recent_data):
        """
        Predict the next 24 hours of energy consumption.

        Parameters:
        - recent_data: DataFrame with at least lookback_hours of recent data
        """
        # Prepare data
        df_prepared = self.prepare_data(recent_data)

        # Take the last lookback_hours
        X = df_prepared[self.feature_columns].tail(self.lookback_hours).values
        X_scaled = self.scaler_X.transform(X)
        X_seq = X_scaled.reshape(1, self.lookback_hours, -1)

        # Predict
        y_pred_scaled = self.model.predict(X_seq, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

        return y_pred.flatten()

    def save_model(self, model_path='lstm_forecaster.h5',
                   scaler_path='scalers.pkl'):
        """Save model and scalers for deployment."""
        self.model.save(model_path)
        joblib.dump({
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_columns': self.feature_columns,
            'lookback_hours': self.lookback_hours
        }, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scalers saved to {scaler_path}")

    def load_model(self, model_path='lstm_forecaster.h5',
                   scaler_path='scalers.pkl'):
        """Load trained model and scalers."""
        from tensorflow.keras.models import load_model
        self.model = load_model(model_path)

        scalers = joblib.load(scaler_path)
        self.scaler_X = scalers['scaler_X']
        self.scaler_y = scalers['scaler_y']
        self.feature_columns = scalers['feature_columns']
        self.lookback_hours = scalers['lookback_hours']
        print("Model and scalers loaded successfully")


# Example usage
if __name__ == "__main__":
    #Example: Load and prepare your data
    df = pd.read_csv('data/household_power_consumption.txt', sep=';',
                     parse_dates={'datetime': ['Date', 'Time']},
                     na_values=['?'])
    df = df.set_index('datetime')
    df = df.resample('H').mean()  # Resample to hourly

    # Initialize forecaster
    forecaster = EnergyForecaster(lookback_hours=168)

    # Train the model
    history = forecaster.train(df, target_column='Global_active_power',
                               epochs=50, batch_size=32)

    # Make predictions
    forecast_24h = forecaster.predict_24h(df.tail(200))
    print(f"24-hour forecast: {forecast_24h}")

    # Save for deployment
    forecaster.save_model()

    print("LSTM Energy Forecaster ready for training!")
    print("Load your data and call forecaster.train(df) to begin.")