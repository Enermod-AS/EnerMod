"""
EnerMod Raspberry Pi Deployment Script
Real-time energy monitoring and intelligent control system
"""

import time
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enermod.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('EnerMod-Pi')


class CurrentSensor:
    """
    Interface for current clamp sensor connected to Raspberry Pi.
    Reads real-time power consumption.
    """

    def __init__(self, gpio_pin=None, voltage=230, calibration_factor=1.0):
        """
        Parameters:
        - gpio_pin: GPIO pin number for analog reading
        - voltage: Mains voltage (230V for EU, 120V for US)
        - calibration_factor: Sensor calibration multiplier
        """
        self.gpio_pin = gpio_pin
        self.voltage = voltage
        self.calibration_factor = calibration_factor
        self.history = deque(maxlen=168)  # Keep 1 week of hourly data

        logger.info(f"Initialized current sensor on GPIO pin {gpio_pin}")

    def read_current(self):
        """
        Read current from sensor (in Amperes).
        In production, this reads from ADC connected to current clamp.
        """
        try:
            # PRODUCTION CODE (with ADS1115 ADC):
            # import Adafruit_ADS1x15
            # adc = Adafruit_ADS1x15.ADS1115()
            # raw_value = adc.read_adc(0, gain=1)
            # voltage_reading = raw_value * (4.096 / 32767.0)
            # current = voltage_reading * self.calibration_factor
            # return current

            # SIMULATION (for testing without hardware):
            # Generate realistic consumption pattern
            hour = datetime.now().hour
            base_load = 0.5  # Base load in kW

            # Time-of-day pattern
            if 6 <= hour <= 9:  # Morning peak
                load_factor = 1.5
            elif 17 <= hour <= 22:  # Evening peak
                load_factor = 2.0
            elif 0 <= hour <= 6:  # Night low
                load_factor = 0.3
            else:
                load_factor = 1.0

            # Add some randomness
            power_kw = base_load * load_factor + np.random.normal(0, 0.1)
            current = power_kw * 1000 / self.voltage

            return max(0, current)

        except Exception as e:
            logger.error(f"Error reading current sensor: {e}")
            return None

    def read_power(self):
        """Calculate power consumption in kW."""
        current = self.read_current()
        if current is None:
            return None

        power_kw = (current * self.voltage) / 1000
        return power_kw

    def get_reading(self):
        """
        Get complete sensor reading with metadata.
        Returns dict suitable for AI models.
        """
        power = self.read_power()
        if power is None:
            return None

        reading = {
            'datetime': datetime.now(),
            'Global_active_power': power,
            'hour': datetime.now().hour,
            'is_weekend': datetime.now().weekday() >= 5
        }

        # Add rolling statistics if we have history
        if len(self.history) > 0:
            recent = [r['Global_active_power'] for r in self.history]

            reading['rolling_mean_3h'] = np.mean(recent[-3:]) if len(recent) >= 3 else power
            reading['rolling_std_3h'] = np.std(recent[-3:]) if len(recent) >= 3 else 0
            reading['rolling_mean_6h'] = np.mean(recent[-6:]) if len(recent) >= 6 else power
            reading['rolling_std_6h'] = np.std(recent[-6:]) if len(recent) >= 6 else 0
            reading['rolling_mean_24h'] = np.mean(recent[-24:]) if len(recent) >= 24 else power
            reading['rolling_std_24h'] = np.std(recent[-24:]) if len(recent) >= 24 else 0
            reading['daily_avg'] = np.mean(recent[-24:]) if len(recent) >= 24 else power
            reading['deviation_from_avg'] = power - reading['daily_avg']

            if len(recent) > 1:
                reading['rate_of_change'] = power - recent[-1]
            else:
                reading['rate_of_change'] = 0
        else:
            # Initialize with defaults
            reading['rolling_mean_3h'] = power
            reading['rolling_std_3h'] = 0
            reading['rolling_mean_6h'] = power
            reading['rolling_std_6h'] = 0
            reading['rolling_mean_24h'] = power
            reading['rolling_std_24h'] = 0
            reading['daily_avg'] = power
            reading['deviation_from_avg'] = 0
            reading['rate_of_change'] = 0

        # Store in history
        self.history.append(reading.copy())

        return reading


class SmartPlugController:
    """
    Interface for controlling smart plugs via GPIO or WiFi API.
    """

    def __init__(self):
        self.plugs = {}
        logger.info("Initialized smart plug controller")

    def register_plug(self, name, gpio_pin=None, api_endpoint=None):
        """Register a smart plug for control."""
        self.plugs[name] = {
            'gpio_pin': gpio_pin,
            'api_endpoint': api_endpoint,
            'state': False
        }
        logger.info(f"Registered smart plug: {name}")

    def turn_on(self, name):
        """Turn on a smart plug."""
        if name not in self.plugs:
            logger.error(f"Unknown plug: {name}")
            return False

        try:
            plug = self.plugs[name]

            if plug['gpio_pin']:
                # GPIO control (relay)
                # import RPi.GPIO as GPIO
                # GPIO.setmode(GPIO.BCM)
                # GPIO.setup(plug['gpio_pin'], GPIO.OUT)
                # GPIO.output(plug['gpio_pin'], GPIO.HIGH)
                pass

            elif plug['api_endpoint']:
                # WiFi smart plug API
                # import requests
                # requests.post(f"{plug['api_endpoint']}/on")
                pass

            plug['state'] = True
            logger.info(f" Turned ON: {name}")
            return True

        except Exception as e:
            logger.error(f"Error turning on {name}: {e}")
            return False

    def turn_off(self, name):
        """Turn off a smart plug."""
        if name not in self.plugs:
            logger.error(f"Unknown plug: {name}")
            return False

        try:
            plug = self.plugs[name]

            if plug['gpio_pin']:
                # GPIO control
                # import RPi.GPIO as GPIO
                # GPIO.output(plug['gpio_pin'], GPIO.LOW)
                pass

            elif plug['api_endpoint']:
                # WiFi smart plug API
                # import requests
                # requests.post(f"{plug['api_endpoint']}/off")
                pass

            plug['state'] = False
            logger.info(f" Turned OFF: {name}")
            return True

        except Exception as e:
            logger.error(f"Error turning off {name}: {e}")
            return False

    def get_state(self, name):
        """Get current state of a plug."""
        if name in self.plugs:
            return self.plugs[name]['state']
        return None


class EnerModRaspberryPi:
    """
    Main application class for Raspberry Pi deployment.
    Integrates sensor reading, AI inference, and control decisions.
    """

    def __init__(self, config_path='config.json'):
        logger.info("Initializing EnerMod on Raspberry Pi...")

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Initialize components
        self.sensor = CurrentSensor(
            gpio_pin=self.config.get('sensor_gpio_pin'),
            voltage=self.config.get('voltage', 230)
        )

        self.plug_controller = SmartPlugController()

        # Load AI models
        self.load_models()

        # Import and initialize moderator
        from moderator import EnerModModerator, Appliance
        self.moderator = EnerModModerator(
            self.forecaster,
            self.detector,
            config_path
        )

        # Register appliances from config
        self.register_appliances()

        # State variables
        self.last_forecast_update = None
        self.running = False

        logger.info(" EnerMod initialized successfully!")

    def load_models(self):
        """Load trained AI models."""
        try:
            logger.info("Loading AI models...")

            # Load LSTM forecaster
            from lstm_forecaster import EnergyForecaster
            self.forecaster = EnergyForecaster()
            self.forecaster.load_model(
                model_path='models/lstm_forecaster.h5',
                scaler_path='models/lstm_scalers.pkl'
            )

            # Load Autoencoder detector (use TFLite for efficiency)
            try:
                import tensorflow as tf
                self.tflite_interpreter = tf.lite.Interpreter(
                    model_path='models/anomaly_detector.tflite'
                )
                self.tflite_interpreter.allocate_tensors()
                logger.info(" Using TFLite optimized anomaly detector")
            except:
                # Fallback to regular model
                from autoencoder_detector import AnomalyDetector
                self.detector = AnomalyDetector()
                self.detector.load_model(
                    model_path='models/autoencoder.h5',
                    scaler_path='models/anomaly_scaler.pkl'
                )
                logger.info(" Using standard anomaly detector")

            logger.info(" All models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def register_appliances(self):
        """Register controllable appliances from configuration."""
        from moderator import Appliance

        appliances_config = self.config.get('appliances', [])

        for app_config in appliances_config:
            appliance = Appliance(
                name=app_config['name'],
                priority=app_config['priority'],
                power_rating=app_config['power_rating'],
                is_deferrable=app_config.get('is_deferrable', True),
                min_runtime=app_config.get('min_runtime', 0)
            )

            self.moderator.register_appliance(appliance)

            # Register with plug controller
            self.plug_controller.register_plug(
                name=app_config['name'],
                gpio_pin=app_config.get('gpio_pin'),
                api_endpoint=app_config.get('api_endpoint')
            )

    def update_forecast(self):
        """Update 24-hour forecast (run daily)."""
        try:
            logger.info("Updating 24-hour forecast...")

            # Get recent history
            history_df = pd.DataFrame(list(self.sensor.history))

            if len(history_df) < 168:
                logger.warning("Not enough data for forecast (need 168 hours)")
                return

            history_df = history_df.set_index('datetime')

            # Generate forecast
            forecast = self.moderator.update_forecast(history_df)

            if forecast is not None:
                logger.info(f" Forecast updated. Peak: {np.max(forecast):.2f} kW")
                self.last_forecast_update = datetime.now()

        except Exception as e:
            logger.error(f"Error updating forecast: {e}")

    def control_loop(self):
        """
        Main control loop - runs continuously.
        """
        logger.info("Starting control loop...")
        self.running = True

        measurement_interval = self.config.get('measurement_interval_seconds', 60)
        forecast_update_hour = self.config.get('forecast_update_hour', 3)

        while self.running:
            try:
                # Get current reading
                reading = self.sensor.get_reading()

                if reading is None:
                    logger.warning("Failed to get sensor reading")
                    time.sleep(measurement_interval)
                    continue

                logger.info(f" Current consumption: "
                            f"{reading['Global_active_power']:.2f} kW")

                # Check if forecast needs updating (once per day)
                current_hour = datetime.now().hour
                if (self.last_forecast_update is None or
                        datetime.now() - self.last_forecast_update > timedelta(hours=23)):
                    if current_hour == forecast_update_hour:
                        self.update_forecast()

                # Make intelligent decisions
                decisions = self.moderator.moderate(reading)

                # Execute actions
                self.execute_decisions(decisions)

                # Log status
                if decisions['actions']:
                    logger.info(f" Executed {len(decisions['actions'])} actions")

                # Sleep until next measurement
                time.sleep(measurement_interval)

            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                self.running = False
                break

            except Exception as e:
                logger.error(f"Error in control loop: {e}")
                time.sleep(measurement_interval)

    def execute_decisions(self, decisions):
        """Execute the moderator's decisions via smart plug controller."""
        for action in decisions['actions']:
            action_type = action['type']

            if action_type in ['pause', 'shutdown']:
                appliance_name = action['appliance']
                self.plug_controller.turn_off(appliance_name)

            elif action_type == 'turn_on':
                appliance_name = action['appliance']
                self.plug_controller.turn_on(appliance_name)

            elif action_type == 'alert':
                logger.warning(f"🚨 ALERT: {action['message']}")
                # Could send notification via email/SMS here
                # self.send_notification(action['message'])

    def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down EnerMod...")
        self.running = False

        # Turn off all controllable appliances safely
        for plug_name in self.plug_controller.plugs.keys():
            self.plug_controller.turn_off(plug_name)

        logger.info(" Shutdown complete")


def main():
    """
    Main entry point for Raspberry Pi deployment.
    """
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║                     EnerMod System                        ║
    ║            AI-Powered Energy Moderator                    ║
    ║                                                           ║
    ║                  Running on Raspberry Pi                  ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    try:
        # Initialize system
        enermod = EnerModRaspberryPi(config_path='config.json')

        # Start control loop
        enermod.control_loop()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")

    except Exception as e:
        logger.error(f"Fatal error: {e}")

    finally:
        if 'enermod' in locals():
            enermod.shutdown()


if __name__ == "__main__":
    main()