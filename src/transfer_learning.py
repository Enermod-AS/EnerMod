"""
Transfer Learning Approach for EnerMod
Adapt pre-trained models to new households with minimal data
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import joblib


class TransferLearningEnerMod:
    """
    Adapt pre-trained EnerMod models to new households.
    Requires only 1-2 weeks of data from the new household.
    """

    def __init__(self, base_lstm_path, base_autoencoder_path):
        """
        Load pre-trained base models.

        Parameters:
        - base_lstm_path: Path to pre-trained LSTM model
        - base_autoencoder_path: Path to pre-trained Autoencoder
        """
        print(" Loading base models...")
        self.lstm_model = load_model(base_lstm_path)
        self.autoencoder_model = load_model(base_autoencoder_path)

        # Freeze early layers (general patterns)
        # Retrain later layers (household-specific patterns)
        self._freeze_base_layers()

        print(" Base models loaded and configured for transfer learning")

    def _freeze_base_layers(self):
        """Freeze early layers that capture general patterns."""
        # For LSTM: Freeze first 2 LSTM layers
        for i, layer in enumerate(self.lstm_model.layers):
            if i < 4:  # First 2 LSTM layers + dropouts
                layer.trainable = False
            else:
                layer.trainable = True

        # For Autoencoder: Freeze encoder
        for i, layer in enumerate(self.autoencoder_model.layers):
            if i < len(self.autoencoder_model.layers) // 2:
                layer.trainable = False
            else:
                layer.trainable = True

        # Recompile models
        self.lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.autoencoder_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        print(" Frozen base layers (general patterns)")
        print(" Unfrozen top layers (household-specific patterns)")

    def fine_tune_for_household(self, household_data, household_id,
                                epochs=20, batch_size=32):
        """
        Fine-tune models for a specific household.

        Parameters:
        - household_data: DataFrame with 1-2 weeks of household consumption data
        - household_id: Unique identifier for this household
        - epochs: Number of training epochs (fewer than initial training)

        Minimum data requirement: 168 hours (1 week)
        Recommended: 336 hours (2 weeks)
        """
        print(f"\n Fine-tuning models for Household: {household_id}")

        if len(household_data) < 168:
            raise ValueError(f"Insufficient data! Need at least 168 hours, got {len(household_data)}")

        print(f" Using {len(household_data)} hours of data")

        # Prepare data (same feature engineering as base model)
        X_train, y_train = self._prepare_household_data(household_data)

        # Fine-tune LSTM
        print("\n Fine-tuning LSTM forecaster...")
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

        history_lstm = self.lstm_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )

        # Fine-tune Autoencoder
        print("\n Fine-tuning Anomaly Detector...")
        X_anomaly = self._prepare_anomaly_features(household_data)

        history_autoencoder = self.autoencoder_model.fit(
            X_anomaly, X_anomaly,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )

        print(f"\n Fine-tuning complete for Household {household_id}!")

        return history_lstm, history_autoencoder

    def _prepare_household_data(self, df):
        """Prepare household data for LSTM (simplified version)."""
        # This would use the same feature engineering as the base model
        # For brevity, showing the structure

        # Add time features
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        # ... (add all other features)

        # Create sequences
        lookback = 168
        forecast_horizon = 24
        # ... (sequence creation logic)

        return X_train, y_train

    def _prepare_anomaly_features(self, df):
        """Prepare features for anomaly detection."""
        # Add rolling statistics
        df['rolling_mean_24h'] = df['Global_active_power'].rolling(24).mean()
        # ... (add all anomaly features)

        return X_anomaly

    def save_household_model(self, household_id, save_dir='household_models/'):
        """Save fine-tuned model for specific household."""
        import os
        os.makedirs(save_dir, exist_ok=True)

        lstm_path = f"{save_dir}/lstm_{household_id}.h5"
        autoencoder_path = f"{save_dir}/autoencoder_{household_id}.h5"

        self.lstm_model.save(lstm_path)
        self.autoencoder_model.save(autoencoder_path)

        print(f" Saved household-specific models to {save_dir}")
        print(f"   - {lstm_path}")
        print(f"   - {autoencoder_path}")


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     Transfer Learning for Multiple Households             ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # Step 1: Load base models (trained on UCI dataset)
    transfer_model = TransferLearningEnerMod(
        base_lstm_path='lstm_forecaster.h5',
        base_autoencoder_path='autoencoder.h5'
    )

    # Step 2: Collect 1-2 weeks of data from new household
    # household_data = collect_household_data()  # Your data collection

    # Step 3: Fine-tune for this specific household
    # history = transfer_model.fine_tune_for_household(
    #     household_data=household_data,
    #     household_id='house_001',
    #     epochs=20
    # )

    # Step 4: Save household-specific model
    # transfer_model.save_household_model('house_001')

    # Step 5: Deploy to that household's Raspberry Pi

    print("\n Transfer Learning Workflow:")
    print("  1. Train base model on UCI dataset (general patterns) ")
    print("  2. Deploy to new household with base model")
    print("  3. Collect 1-2 weeks of household data")
    print("  4. Fine-tune model (20 epochs, ~5 minutes)")
    print("  5. Deploy household-specific model")
    print("  6. Repeat for each new household")

    print("\n Benefits:")
    print("   Only needs 1-2 weeks of data per household")
    print("   Much faster training (20 epochs vs 50)")
    print("   Leverages general patterns from UCI dataset")
    print("   Adapts to specific household characteristics")

    print("\n️ Important:")
    print("  - Collect data during 'normal' household operation")
    print("  - Avoid training during holidays or unusual periods")
    print("  - Retrain every 3-6 months to adapt to seasonal changes")