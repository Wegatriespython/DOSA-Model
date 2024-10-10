import numpy as np

def adaptive_expectations(historical_data, previous_forecasts, time_horizon):
    new_forecasts = {}

    for key in historical_data:
        history = np.array(historical_data[key])
        prev_forecast = np.array(previous_forecasts.get(key, []))

        if len(history) == 0:
            new_forecasts[key] = np.zeros(time_horizon)
            continue

        if len(prev_forecast) == 0:
            # If there's no previous forecast, use the last historical value
            aligned_forecast = np.full_like(history, history[-1])
        else:
            # Align forecast with history
            aligned_forecast = prev_forecast[-min(len(history), len(prev_forecast)):]
            
        aligned_history = history[-len(aligned_forecast):]

        # Ensure aligned arrays have the same length
        min_length = min(len(aligned_history), len(aligned_forecast))
        aligned_history = aligned_history[-min_length:]
        aligned_forecast = aligned_forecast[-min_length:]

        forecast_error = aligned_history - aligned_forecast
        mse = np.mean(forecast_error**2)

        # Adaptive learning rate based on mean squared error
        alpha = 1 / (1 + np.exp(-mse))  # Sigmoid function to keep alpha between 0 and 1

        # Generate new forecast
        forecast = np.zeros(len(history) + time_horizon)
        forecast[:len(history)] = history  # Historical data remains unchanged

        for i in range(len(history), len(forecast)):
            if i - len(history) < len(prev_forecast):
                # Adapt the forecast based on previous error
                forecast[i] = prev_forecast[i - len(history)] + alpha * (history[-1] - prev_forecast[i - len(history)])
            else:
                # For periods beyond previous forecast, use the last adapted value
                forecast[i] = forecast[i-1]

        new_forecasts[key] = forecast

    return new_forecasts
