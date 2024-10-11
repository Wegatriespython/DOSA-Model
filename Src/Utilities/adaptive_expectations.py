import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create constant 'pe' matrix with values dependent on
        # pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(1)  # Shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to input tensor
        x = x + self.pe[:x.size(0), :]
        return x
    
class TransformerTimeSeriesPredictor(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, output_size=1, dropout=0.1):
        super(TransformerTimeSeriesPredictor, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_size)

    def forward(self, src):
        # src shape: [seq_len, batch_size, input_size]
        src = self.input_projection(src) * np.sqrt(src.size(-1))
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output


def adaptive_expectations(historical_data, previous_forecasts, time_horizon):

    transformer_forecasts = transformer_expectations(historical_data, time_horizon)
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
    print(f"adaptive: {new_forecasts}, transformer: {transformer_forecasts}")
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