import numpy as np

def adaptive_expectations(historical_data, previous_forecasts, time_horizon):
    new_forecasts = {}

    for key in historical_data:
        history = np.array(historical_data[key])
        prev_forecast = np.array(previous_forecasts.get(key, []))

        if len(history) == 0:
            new_forecasts[key] = np.zeros(len(history) + time_horizon)
            continue

        # Align previous forecast with historical data
        aligned_forecast = prev_forecast[-len(history):]

        # Calculate forecast error for the overlapping period
        forecast_error = history[-len(aligned_forecast):] - aligned_forecast[-len(history):]
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
