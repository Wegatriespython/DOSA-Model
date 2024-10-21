import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
import copy
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Transformer model
class MarketTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048):
        super(MarketTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Custom Dataset
class MarketDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+1:idx+self.seq_length+1]
        return torch.FloatTensor(x), torch.FloatTensor(y)

# Load and preprocess data
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    processed_data = []
    for step in data:
        step_data = []
        for market, values in step['markets'].items():
            # Convert string values to float where possible, handle numeric values
            step_data.extend([float(v) if isinstance(v, str) and v.replace('.', '').isdigit() 
                              else float(v) if isinstance(v, (int, float)) 
                              else 0 for v in values.values()])
        processed_data.append(step_data)
    
    return np.array(processed_data, dtype=np.float32)

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Add this function for early stopping
def early_stopping(val_losses, patience=10, min_delta=0.001):
    if len(val_losses) < patience + 1:
        return False
    
    # We are looking for a minimum, so we should multiply by -1
    recent_best = max(-loss for loss in val_losses[-patience-1:-1])
    return (-val_losses[-1] - min_delta) < recent_best

# Evaluate model
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            total_loss += loss.item()
            
            # Reshape output and batch_y to 2D
            all_predictions.append(output.reshape(-1, output.shape[-1]).cpu().numpy())
            all_targets.append(batch_y.reshape(-1, batch_y.shape[-1]).cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    return avg_loss, all_predictions, all_targets

# Normalize data
def normalize_data(data, method='standard'):
    # Reshape data to 2D array if it's not already
    original_shape = data.shape
    data_2d = data.reshape(-1, data.shape[-1])
    
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Unknown normalization method")
    
    # Fit and transform the data
    normalized_data = scaler.fit_transform(data_2d)
    
    # Reshape back to original shape
    normalized_data = normalized_data.reshape(original_shape)
    
    return normalized_data, scaler

# Denormalize predictions
def denormalize_predictions(normalized_predictions, scaler):
    # Reshape predictions if necessary
    original_shape = normalized_predictions.shape
    predictions_2d = normalized_predictions.reshape(-1, normalized_predictions.shape[-1])
    
    # Denormalize
    denormalized_predictions = scaler.inverse_transform(predictions_2d)
    
    # Reshape back to original shape
    denormalized_predictions = denormalized_predictions.reshape(original_shape)
    
    return denormalized_predictions

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    data = load_data('market_data_20241013_230001.json')
    
    # Normalize the data
    normalized_data, scaler = normalize_data(data, method='standard')
    
    # Print statistics of normalized data
    print(f"Normalized data shape: {normalized_data.shape}")
    print(f"Normalized data mean: {np.mean(normalized_data)}")
    print(f"Normalized data std: {np.std(normalized_data)}")
    print(f"Normalized data min: {np.min(normalized_data)}")
    print(f"Normalized data max: {np.max(normalized_data)}")
    
    # Print data statistics
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Min value: {np.min(data)}")
    print(f"Max value: {np.max(data)}")
    print(f"Mean value: {np.mean(data)}")
    print(f"Number of NaNs: {np.isnan(data).sum()}")
    print(f"Number of Infs: {np.isinf(data).sum()}")

    # Define parameters
    input_dim = output_dim = data.shape[1]
    seq_length = 5
    batch_size = 8
    num_epochs = 1000
    patience = 80
    k_folds = 5

    # Create dataset
    dataset = MarketDataset(normalized_data, seq_length)

    # Prepare for k-fold cross-validation
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Lists to store results
    fold_results = []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"FOLD {fold}")
        print("--------------------------------")
        
        # Split data into train and validation sets
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
        
        # Initialize model
        model = MarketTransformer(input_dim, output_dim).to(device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Training loop
        best_val_loss = float('inf')
        val_losses = []
        best_model = None
        
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, criterion, optimizer, device)
            
            # Validation step
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    output = model(batch_x)
                    val_loss += criterion(output, batch_y).item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
            
            if early_stopping(val_losses, patience):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save the best model for this fold
        torch.save(best_model.state_dict(), f'market_transformer_fold_{fold}.pth')
        
        # Evaluate on the validation set
        val_loss, predictions, targets = evaluate_model(best_model, val_loader, criterion, device)
        
        # Ensure predictions and targets are 2D
        predictions = predictions.reshape(-1, predictions.shape[-1])
        targets = targets.reshape(-1, targets.shape[-1])
        
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        fold_results.append({
            'fold': fold,
            'val_loss': val_loss,
            'mse': mse,
            'mae': mae,
            'r2': r2
        })
        
        print(f"Fold {fold} Validation Results:")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R2 Score: {r2:.4f}")
        print("--------------------------------")

    # Calculate and print average metrics across all folds
    avg_val_loss = np.mean([r['val_loss'] for r in fold_results])
    avg_mse = np.mean([r['mse'] for r in fold_results])
    avg_mae = np.mean([r['mae'] for r in fold_results])
    avg_r2 = np.mean([r['r2'] for r in fold_results])

    print("\nAverage Validation Results Across All Folds:")
    print(f"  Val Loss: {avg_val_loss:.4f}")
    print(f"  MSE: {avg_mse:.4f}")
    print(f"  MAE: {avg_mae:.4f}")
    print(f"  R2 Score: {avg_r2:.4f}")


    print("\nValidation results saved to 'validation_results.json'")
