import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('EnerMod')


class Appliance:
    """Represents a controllable appliance in the system."""

    def __init__(self, name, priority, power_rating,
                 is_deferrable=True, min_runtime=0):
        """
        Parameters:
        - name: Appliance identifier
        - priority: 1-10, where 10 is highest priority (critical)
        - power_rating: Power consumption in watts
        - is_deferrable: Can be shifted to off-peak hours
        - min_runtime: Minimum continuous runtime in hours (for appliances
                       like washing machines)
        """
        self.name = name
        self.priority = priority
        self.power_rating = power_rating
        self.is_deferrable = is_deferrable
        self.min_runtime = min_runtime
        self.state = False  # On/Off
        self.scheduled_start = None


class EnerModModerator:
    """
    The intelligent decision engine for EnerMod.
    Uses forecasts and anomaly detection to actively manage energy consumption.
    """

    def __init__(self, forecaster, anomaly_detector, config_path='config.json'):
        """
        Parameters:
        - forecaster: Trained LSTM forecasting model
        - anomaly_detector: Trained autoencoder anomaly detector
        - config_path: Path to user configuration file
        """
        self.forecaster = forecaster
        self.anomaly_detector = anomaly_detector
        self.appliances = {}
        self.config = self.load_config(config_path)
        self.current_forecast = None
        self.action_log = []

    def load_config(self, config_path):
        """Load user configuration including peak hours and limits."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            # Default configuration
            config = {
                'peak_hours': [17, 18, 19, 20, 21],  # 5 PM - 9 PM
                'off_peak_hours': [0, 1, 2, 3, 4, 5, 22, 23],
                'consumption_limit_kw': 5.0,  # Peak shaving threshold
                'peak_buffer_percent': 0.9,  # Trigger at 90% of limit
                'anomaly_action': 'alert',  # 'alert' or 'shutdown'
                'enable_load_shifting': True,
                'enable_peak_shaving': True
            }
            logger.warning(f"Config file not found. Using defaults.")

        return config

    def register_appliance(self, appliance):
        """Register a new controllable appliance."""
        self.appliances[appliance.name] = appliance
        logger.info(f"Registered appliance: {appliance.name} "
                    f"(Priority: {appliance.priority})")

    def update_forecast(self, recent_data):
        """
        Get fresh 24-hour forecast from the LSTM model.
        Should be called once per day or when patterns change significantly.
        """
        try:
            forecast = self.forecaster.predict_24h(recent_data)
            self.current_forecast = forecast
            logger.info(f"Forecast updated. Peak predicted at "
                        f"{np.max(forecast):.2f} kW")
            return forecast
        except Exception as e:
            logger.error(f"Forecast update failed: {e}")
            return None

    def is_peak_period(self, hour=None):
        """Check if current time is during peak hours."""
        if hour is None:
            hour = datetime.now().hour
        return hour in self.config['peak_hours']

    def is_off_peak(self, hour=None):
        """Check if current time is during off-peak hours."""
        if hour is None:
            hour = datetime.now().hour
        return hour in self.config['off_peak_hours']

    def check_anomaly(self, current_reading):
        """
        Check for anomalous consumption pattern.

        Returns:
        - anomaly_result: Dict with detection results
        """
        result = self.anomaly_detector.detect_anomaly(current_reading)

        if result['is_anomaly']:
            logger.warning(f" ANOMALY DETECTED! "
                           f"Error: {result['reconstruction_error']:.4f}, "
                           f"Confidence: {result['confidence']:.2%}")
            self.action_log.append({
                'timestamp': datetime.now(),
                'event': 'anomaly_detected',
                'details': result
            })

        return result

    def get_forecasted_consumption(self, hours_ahead=0):
        """Get forecasted consumption for specified hours ahead."""
        if self.current_forecast is None:
            return None

        if hours_ahead >= len(self.current_forecast):
            return None

        return self.current_forecast[hours_ahead]

    def calculate_total_load(self):
        """Calculate current total load from active appliances."""
        return sum(app.power_rating for app in self.appliances.values()
                   if app.state) / 1000.0  # Convert to kW

    def peak_shaving_decision(self, current_consumption):
        """
        Decide if peak shaving action is needed.

        Returns:
        - action_needed: Boolean
        - appliances_to_pause: List of appliance names
        """
        limit = self.config['consumption_limit_kw']
        threshold = limit * self.config['peak_buffer_percent']

        if current_consumption < threshold:
            return False, []

        # Calculate how much we need to reduce
        reduction_needed = current_consumption - threshold

        # Sort appliances by priority (lowest first) and deferrable status
        sortable = [(name, app) for name, app in self.appliances.items()
                    if app.state and app.is_deferrable]
        sortable.sort(key=lambda x: x[1].priority)

        to_pause = []
        reduction_achieved = 0.0

        for name, app in sortable:
            to_pause.append(name)
            reduction_achieved += app.power_rating / 1000.0

            if reduction_achieved >= reduction_needed:
                break

        if to_pause:
            logger.info(f" Peak shaving activated. Pausing: {to_pause}")
            self.action_log.append({
                'timestamp': datetime.now(),
                'event': 'peak_shaving',
                'appliances_paused': to_pause,
                'consumption': current_consumption,
                'reduction': reduction_achieved
            })

        return len(to_pause) > 0, to_pause

    def load_shifting_decision(self):
        """
        Decide which deferrable appliances should be shifted to off-peak.

        Returns:
        - appliances_to_shift: Dict with appliance names and suggested times
        """
        if not self.config['enable_load_shifting']:
            return {}

        current_hour = datetime.now().hour

        # Only shift if we're in or approaching peak period
        if not self.is_peak_period(current_hour):
            return {}

        # Find next off-peak period
        next_off_peak = None
        for hour_offset in range(1, 24):
            future_hour = (current_hour + hour_offset) % 24
            if self.is_off_peak(future_hour):
                next_off_peak = future_hour
                break

        if next_off_peak is None:
            return {}

        # Identify deferrable appliances that are on or scheduled
        to_shift = {}
        for name, app in self.appliances.items():
            if app.is_deferrable and app.state:
                # Don't shift high-priority or if minimum runtime not met
                if app.priority < 7:
                    to_shift[name] = {
                        'current_time': current_hour,
                        'suggested_time': next_off_peak,
                        'reason': 'peak_avoidance'
                    }
                    logger.info(f" Scheduling {name} for off-peak hours "
                                f"({next_off_peak}:00)")

        if to_shift:
            self.action_log.append({
                'timestamp': datetime.now(),
                'event': 'load_shifting',
                'shifts': to_shift
            })

        return to_shift

    def handle_anomaly(self, anomaly_result):
        """
        Take action based on anomaly detection.

        Returns:
        - actions: List of actions taken
        """
        actions = []

        if not anomaly_result['is_anomaly']:
            return actions

        action_type = self.config.get('anomaly_action', 'alert')

        if action_type == 'alert':
            actions.append({
                'type': 'alert',
                'message': f"Unusual consumption detected! "
                           f"Confidence: {anomaly_result['confidence']:.2%}",
                'severity': 'warning'
            })

        elif action_type == 'shutdown' and anomaly_result['confidence'] > 0.5:
            # Shutdown non-critical appliances for safety
            to_shutdown = [name for name, app in self.appliances.items()
                           if app.state and app.priority < 8]

            for name in to_shutdown:
                actions.append({
                    'type': 'shutdown',
                    'appliance': name,
                    'reason': 'safety_anomaly'
                })
                logger.warning(f" Emergency shutdown: {name}")

        return actions

    def moderate(self, current_reading):
        """
        Main decision loop - called every measurement cycle.

        Parameters:
        - current_reading: Dict or DataFrame row with current sensor data

        Returns:
        - decisions: Dict with all decisions and actions
        """
        decisions = {
            'timestamp': datetime.now(),
            'current_consumption': current_reading.get('Global_active_power', 0),
            'actions': []
        }

        # 1. Check for anomalies
        anomaly_result = self.check_anomaly(current_reading)
        if anomaly_result['is_anomaly']:
            anomaly_actions = self.handle_anomaly(anomaly_result)
            decisions['actions'].extend(anomaly_actions)
            decisions['anomaly_detected'] = True

        # 2. Check if peak shaving is needed
        if self.config['enable_peak_shaving']:
            current_total = self.calculate_total_load()
            needs_shaving, to_pause = self.peak_shaving_decision(current_total)

            if needs_shaving:
                for appliance_name in to_pause:
                    decisions['actions'].append({
                        'type': 'pause',
                        'appliance': appliance_name,
                        'reason': 'peak_shaving',
                        'duration': 'until_load_reduces'
                    })

        # 3. Check for load shifting opportunities
        if self.config['enable_load_shifting']:
            shifts = self.load_shifting_decision()

            for appliance_name, shift_info in shifts.items():
                decisions['actions'].append({
                    'type': 'schedule',
                    'appliance': appliance_name,
                    'scheduled_time': shift_info['suggested_time'],
                    'reason': shift_info['reason']
                })

        # 4. Compare with forecast
        if self.current_forecast is not None:
            forecasted = self.get_forecasted_consumption(0)
            actual = decisions['current_consumption']

            if forecasted and abs(actual - forecasted) > forecasted * 0.3:
                logger.info(f" Significant deviation from forecast. "
                            f"Expected: {forecasted:.2f} kW, "
                            f"Actual: {actual:.2f} kW")
                decisions['forecast_deviation'] = {
                    'expected': forecasted,
                    'actual': actual,
                    'deviation_percent': ((actual - forecasted) / forecasted) * 100
                }

        # Log decisions
        if decisions['actions']:
            logger.info(f" Moderator decisions: {len(decisions['actions'])} "
                        f"actions taken")

        return decisions

    def execute_actions(self, decisions):
        """
        Execute the decisions by controlling smart plugs.
        This is where you'd integrate with GPIO or smart plug APIs.

        For now, it simulates the actions.
        """
        for action in decisions['actions']:
            action_type = action['type']

            if action_type in ['pause', 'shutdown']:
                appliance_name = action['appliance']
                if appliance_name in self.appliances:
                    self.appliances[appliance_name].state = False
                    logger.info(f"⏸ Turned OFF: {appliance_name}")
                    # In real implementation:
                    # smart_plug.turn_off(appliance_name)

            elif action_type == 'schedule':
                appliance_name = action['appliance']
                scheduled_time = action['scheduled_time']
                if appliance_name in self.appliances:
                    self.appliances[appliance_name].scheduled_start = scheduled_time
                    logger.info(f" Scheduled {appliance_name} for "
                                f"{scheduled_time}:00")

            elif action_type == 'alert':
                logger.warning(f" ALERT: {action['message']}")
                # In real implementation:
                # send_notification(action['message'])

    def get_status_report(self):
        """Generate a status report for the dashboard."""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_appliances': len(self.appliances),
            'active_appliances': sum(1 for app in self.appliances.values()
                                     if app.state),
            'current_load_kw': self.calculate_total_load(),
            'forecast_available': self.current_forecast is not None,
            'recent_actions': self.action_log[-10:],  # Last 10 actions
            'configuration': self.config
        }


# Example usage and integration
if __name__ == "__main__":
    # This demonstrates how all components work together

    print("=== EnerMod Moderator System ===\n")

    # Initialize (assume models are already trained and loaded)
    # forecaster = EnergyForecaster()
    # forecaster.load_model()
    # detector = AnomalyDetector()
    # detector.load_model()

    # moderator = EnerModModerator(forecaster, detector)

    # Register appliances
    # moderator.register_appliance(
    #     Appliance('washing_machine', priority=4, power_rating=2000,
    #              is_deferrable=True, min_runtime=1)
    # )
    # moderator.register_appliance(
    #     Appliance('ev_charger', priority=3, power_rating=7000,
    #              is_deferrable=True, min_runtime=2)
    # )
    # moderator.register_appliance(
    #     Appliance('refrigerator', priority=10, power_rating=150,
    #              is_deferrable=False)
    # )
    # moderator.register_appliance(
    #     Appliance('hvac', priority=7, power_rating=3000,
    #              is_deferrable=False)
    # )

    # Main control loop (runs continuously on Raspberry Pi)
    # while True:
    #     # Get current sensor reading
    #     current_reading = get_sensor_data()  # Your sensor reading function
    #
    #     # Make decisions
    #     decisions = moderator.moderate(current_reading)
    #
    #     # Execute actions
    #     moderator.execute_actions(decisions)
    #
    #     # Update dashboard
    #     status = moderator.get_status_report()
    #     update_dashboard(status)
    #
    #     # Sleep until next reading (e.g., every minute)
    #     time.sleep(60)

    print("Moderator system ready for deployment!")
    print("Register your appliances and start the control loop.")