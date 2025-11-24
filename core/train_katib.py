"""
Training Script for Katib HPO
This script is called by Katib with different hyperparameter combinations
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from minio import Minio
from io import BytesIO
import mlflow
import mlflow.pytorch
import os
import sys


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, pred_horizon):
        self.data = data
        self.seq_length = seq_length
        self.pred_horizon = pred_horizon
        
    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_horizon + 1
        
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length:idx + self.seq_length + self.pred_horizon]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--sequence-length', type=int, default=168)
    parser.add_argument('--prediction-horizon', type=int, default=24)
    args = parser.parse_args()
    
    print(f"Training with hyperparameters:")
    print(f"  hidden_size: {args.hidden_size}")
    print(f"  num_layers: {args.num_layers}")
    print(f"  dropout: {args.dropout}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  batch_size: {args.batch_size}")
    
    # Get environment variables
    minio_endpoint = os.getenv('MINIO_ENDPOINT')
    minio_access_key = os.getenv('MINIO_ACCESS_KEY')
    minio_secret_key = os.getenv('MINIO_SECRET_KEY')
    bucket_name = os.getenv('BUCKET_NAME')
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    
    # Setup MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("electricity-katib-hpo")
    
    # Load data from MinIO
    client = Minio(
        minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False
    )
    
    # Get latest validated data
    objects = client.list_objects(bucket_name, prefix='processed/validated_data_')
    latest_object = sorted(objects, key=lambda x: x.last_modified, reverse=True)[0]
    
    response = client.get_object(bucket_name, latest_object.object_name)
    df = pd.read_csv(BytesIO(response.read()))
    df['period'] = pd.to_datetime(df['period'])
    
    # Select region with most data
    region_counts = df['respondent'].value_counts()
    selected_region = region_counts.index[0]
    df = df[df['respondent'] == selected_region].copy()
    df = df.sort_values('period').reset_index(drop=True)
    
    print(f"Training on region: {selected_region} with {len(df)} records")
    
    # Prepare data
    values = df['value'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)
    
    # Split data
    train_size = int(0.7 * len(scaled_values))
    val_size = int(0.15 * len(scaled_values))
    
    train_data = scaled_values[:train_size]
    val_data = scaled_values[train_size:train_size + val_size]
    test_data = scaled_values[train_size + val_size:]
    
    train_dataset = TimeSeriesDataset(train_data, args.sequence_length, args.prediction_horizon)
    val_dataset = TimeSeriesDataset(val_data, args.sequence_length, args.prediction_horizon)
    test_dataset = TimeSeriesDataset(test_data, args.sequence_length, args.prediction_horizon)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMForecaster(
        input_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=args.prediction_horizon,
        dropout=args.dropout
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop with MLflow
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("hidden_size", args.hidden_size)
        mlflow.log_param("num_layers", args.num_layers)
        mlflow.log_param("dropout", args.dropout)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("batch_size", args.batch_size)
        
        best_val_loss = float('inf')
        
        for epoch in range(args.epochs):
            # Train
            model.train()
            train_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                output = model(x_batch)
                loss = criterion(output, y_batch.squeeze())
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    output = model(x_batch)
                    loss = criterion(output, y_batch.squeeze())
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            # Log to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Test evaluation
        model.eval()
        test_predictions = []
        test_actuals = []
        
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                output = model(x_batch)
                test_predictions.extend(output.cpu().numpy())
                test_actuals.extend(y_batch.numpy())
        
        test_predictions = np.array(test_predictions).reshape(-1, 1)
        test_actuals = np.array(test_actuals).reshape(-1, 1)
        
        # Inverse transform
        test_predictions_scaled = scaler.inverse_transform(test_predictions)
        test_actuals_scaled = scaler.inverse_transform(test_actuals)
        
        # Calculate metrics
        test_rmse = np.sqrt(mean_squared_error(test_actuals_scaled, test_predictions_scaled))
        test_mape = mean_absolute_percentage_error(test_actuals_scaled, test_predictions_scaled) * 100
        
        # Log final metrics
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mape", test_mape)
        mlflow.log_metric("best_val_loss", best_val_loss)
        
        # Print metrics for Katib to capture
        print(f"test_rmse={test_rmse:.6f}")
        print(f"test_mape={test_mape:.6f}")
        print(f"val_loss={best_val_loss:.6f}")
        
        # Save model to MLflow
        mlflow.pytorch.log_model(model, "model")
        
        print("Training completed successfully!")


if __name__ == '__main__':
    # Install required packages first
    import subprocess
    packages = [
        'pandas', 'numpy', 'torch', 'scikit-learn',
        'mlflow', 'minio', 'requests'
    ]
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
    
    main()
