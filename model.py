"""
PyTorch LSTM Model for Electricity Load Forecasting
Implements both LSTM and Transformer architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting"""
    
    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int,
        prediction_horizon: int
    ):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
    
    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[
            idx + self.sequence_length:
            idx + self.sequence_length + self.prediction_horizon
        ]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class LSTMForecaster(nn.Module):
    """LSTM-based electricity load forecaster"""
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 24,
        dropout: float = 0.2
    ):
        super(LSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(self.relu(self.fc1(last_hidden)))
        out = self.fc2(out)
        
        return out


class TransformerForecaster(nn.Module):
    """Transformer-based electricity load forecaster"""
    
    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 512,
        output_size: int = 24,
        dropout: float = 0.2
    ):
        super(TransformerForecaster, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = self.input_projection(x)
        
        # Transformer encoding
        transformer_out = self.transformer_encoder(x)
        
        # Global average pooling
        pooled = transformer_out.mean(dim=1)
        
        # Fully connected layers
        out = self.dropout(self.relu(self.fc1(pooled)))
        out = self.fc2(out)
        
        return out


class ElectricityLoadForecaster:
    """Wrapper class for training and inference"""
    
    def __init__(
        self,
        model_type: str = "lstm",
        sequence_length: int = 168,
        prediction_horizon: int = 24,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        device: Optional[str] = None
    ):
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        if model_type == "lstm":
            self.model = LSTMForecaster(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=prediction_horizon,
                dropout=dropout
            ).to(self.device)
        elif model_type == "transformer":
            self.model = TransformerForecaster(
                input_size=1,
                d_model=hidden_size,
                nhead=8,
                num_encoder_layers=num_layers,
                output_size=prediction_horizon,
                dropout=dropout
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.scaler = StandardScaler()
        
        logger.info(f"Initialized {model_type.upper()} model on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        train_split: float = 0.7,
        val_split: float = 0.15
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare train/val/test dataloaders"""
        
        # Extract values and normalize
        values = df['value'].values.reshape(-1, 1)
        values_scaled = self.scaler.fit_transform(values)
        
        # Split data
        n = len(values_scaled)
        train_size = int(n * train_split)
        val_size = int(n * val_split)
        
        train_data = values_scaled[:train_size]
        val_data = values_scaled[train_size:train_size + val_size]
        test_data = values_scaled[train_size + val_size:]
        
        # Create datasets
        train_dataset = TimeSeriesDataset(
            train_data, self.sequence_length, self.prediction_horizon
        )
        val_dataset = TimeSeriesDataset(
            val_data, self.sequence_length, self.prediction_horizon
        )
        test_dataset = TimeSeriesDataset(
            test_data, self.sequence_length, self.prediction_horizon
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y.squeeze(-1))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y.squeeze(-1))
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.squeeze(-1).cpu().numpy())
        
        # Calculate metrics
        mse = total_loss / len(data_loader)
        rmse = np.sqrt(mse)
        
        # Calculate MAPE
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        mape = np.mean(np.abs((all_targets - all_predictions) / (all_targets + 1e-8))) * 100
        
        return rmse, mape
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10
    ):
        """Full training loop with early stopping"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_rmse, val_mape = self.evaluate(val_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val RMSE: {val_rmse:.4f}, "
                f"Val MAPE: {val_mape:.2f}%"
            )
            
            # Early stopping
            if val_rmse < best_val_loss:
                best_val_loss = val_rmse
                patience_counter = 0
                # Save best model
                self.save_model("best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.load_model("best_model.pt")
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            if len(x_tensor.shape) == 2:
                x_tensor = x_tensor.unsqueeze(0)
            
            predictions = self.model(x_tensor)
            predictions = predictions.cpu().numpy()
        
        # Inverse transform
        predictions_original = self.scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).flatten()
        
        return predictions_original
    
    def save_model(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler,
            'config': {
                'model_type': self.model_type,
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon
            }
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler = checkpoint['scaler']
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test model
    forecaster = ElectricityLoadForecaster(
        model_type="lstm",
        sequence_length=168,
        prediction_horizon=24
    )
    
    print(f"Model: {forecaster.model}")
