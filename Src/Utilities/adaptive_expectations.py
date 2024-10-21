import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim



def autoregressive(historical_data, previous_forecasts, time_horizon= 10):
    new_forecasts = {}
    
    for key in historical_data:
        history = np.array(historical_data[key])
        
        if len(history) < 2:
            new_forecasts[key] = np.full(time_horizon, history[-1])
        else:
            # Use the last 10 periods (or all if less than 10) for the AR model
            recent_history = history[-min(time_horizon, len(history)):]
            
            # Fit a simple AR(1) model
            diff = np.diff(recent_history)
            ar_coef = np.mean(diff)
            
            # Generate forecast
            forecast = np.zeros(time_horizon)
            forecast[0] = history[-1] + ar_coef  # First forecast
            
            for i in range(1, time_horizon):
                forecast[i] = forecast[i-1] + ar_coef
            
            new_forecasts[key] = forecast

    return new_forecasts

def adaptive_expectations(historical_data, previous_forecasts, time_horizon):
   

    #transformer_forecasts = transformer_expectations(historical_data, time_horizon)
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
        
        # Calculate weighted MSE with more emphasis on recent errors
        weights = np.exp(np.arange(len(forecast_error)) / 5) / np.exp(np.arange(len(forecast_error)) / 5).sum()
        weighted_mse = np.average(forecast_error**2, weights=weights)

        # Adaptive learning rate based on weighted MSE
        alpha = 1 / (1 + 0.5 * np.exp(-weighted_mse))  # Adjusted sigmoid for more responsiveness

        # Increase alpha for the most recent 5 periods
        recent_alpha = min(1.0, alpha * 2)  # Double alpha for recent periods, but cap at 1.0
        alpha_values = np.full(len(forecast_error), alpha)
        alpha_values[-min(5, len(alpha_values)):] = recent_alpha

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

        new_forecasts[key] = forecast[-time_horizon:]  # Return only the forecasted values
    #print(f"adaptive: {new_forecasts}, transformer: {transformer_forecasts}")
    return new_forecasts

def transformer_expectations(historical_data, time_horizon, model_dir='models'):
    new_forecasts = {}

    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)

    for key in historical_data:
        history = np.array(historical_data[key])

        if len(history) < 2:
            new_forecasts[key] = np.zeros(time_horizon)
            continue

        # Prepare input and target for the Transformer
        input_data = torch.FloatTensor(history[:-1]).view(-1, 1, 1)
        target_data = torch.FloatTensor(history[1:]).view(-1, 1, 1)

        # Initialize the model
        model = TransformerTimeSeriesPredictor()

        model_path = os.path.join(model_dir, f"{key}_transformer.pth")

        if os.path.exists(model_path):
            # Load existing model weights
            model.load_state_dict(torch.load(model_path))
        else:
            # Train the model for the first time
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            epochs = 100
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                output = model(input_data)
                loss = criterion(output, target_data)
                loss.backward()
                optimizer.step()

            # Save the trained model
            torch.save(model.state_dict(), model_path)

        # Optionally, perform quick fine-tuning with the latest data
        # Uncomment the following lines to enable fine-tuning
        """
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        epochs = 5  # Few epochs for quick adaptation
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(input_data)
            loss = criterion(output, target_data)
            loss.backward()
            optimizer.step()
        # Save the updated model
        torch.save(model.state_dict(), model_path)
        """

        # Generate predictions
        model.eval()
        with torch.no_grad():
            forecasts = []
            src = input_data.clone()
            for _ in range(time_horizon):
                output = model(src)
                next_pred = output[-1].unsqueeze(0)
                forecasts.append(next_pred.item())
                src = torch.cat([src, next_pred], dim=0)

        new_forecasts[key] = forecasts

    return new_forecasts
