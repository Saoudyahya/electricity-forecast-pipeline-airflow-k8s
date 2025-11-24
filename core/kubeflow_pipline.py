"""
Kubeflow Pipeline for Electricity Load Forecasting with MinIO Storage
All data and models are stored in MinIO
"""

from kfp import dsl, compiler
from kfp.dsl import component, OutputPath, InputPath
from typing import NamedTuple

# Base image with all dependencies
BASE_IMAGE = "python:3.10-slim"


@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "pandas==2.0.3",
        "numpy==1.24.3",
        "requests==2.31.0",
        "minio==7.2.0",
        "python-dateutil==2.8.2"
    ]
)
def extract_data_to_minio(
    eia_api_key: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str,
    days: int,
    output_path: OutputPath(str)
) -> NamedTuple('Outputs', [('object_name', str), ('num_records', int)]):
    """Extract electricity data from EIA API and save to MinIO"""
    import pandas as pd
    import requests
    from datetime import datetime, timedelta
    from minio import Minio
    from io import BytesIO
    from collections import namedtuple

    print(f"Extracting data for last {days} days...")

    # EIA API configuration
    base_url = "https://api.eia.gov/v2"
    endpoint = "/electricity/rto/region-data/data/"

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    params = {
        'api_key': eia_api_key,
        'frequency': 'hourly',
        'data[]': 'value',
        'start': start_date.strftime('%Y-%m-%dT%H'),
        'end': end_date.strftime('%Y-%m-%dT%H'),
        'sort[0][column]': 'period',
        'sort[0][direction]': 'asc',
        'offset': 0,
        'length': 5000
    }

    all_data = []
    max_retries = 10

    for retry in range(max_retries):
        try:
            response = requests.get(f"{base_url}{endpoint}", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'response' in data and 'data' in data['response']:
                records = data['response']['data']
                if not records:
                    break
                all_data.extend(records)

                if len(records) < params['length']:
                    break

                params['offset'] += params['length']
            else:
                break
        except Exception as e:
            print(f"API call retry {retry + 1}/{max_retries}: {e}")
            if retry == max_retries - 1:
                raise
            continue

    if not all_data:
        raise ValueError("No data retrieved from EIA API")

    df = pd.DataFrame(all_data)
    print(f"Extracted {len(df)} records")

    # Initialize MinIO client
    client = Minio(
        minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False
    )

    # Ensure bucket exists
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        print(f"Created bucket: {bucket_name}")

    # Save to MinIO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    object_name = f"raw/electricity_data_{timestamp}.csv"

    csv_bytes = df.to_csv(index=False).encode('utf-8')
    csv_buffer = BytesIO(csv_bytes)

    client.put_object(
        bucket_name,
        object_name,
        csv_buffer,
        length=len(csv_bytes),
        content_type='text/csv'
    )

    print(f"Data saved to MinIO: {bucket_name}/{object_name}")

    # Write output path for next component
    with open(output_path, 'w') as f:
        f.write(object_name)

    outputs = namedtuple('Outputs', ['object_name', 'num_records'])
    return outputs(object_name, len(df))


@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "pandas==2.0.3",
        "numpy==1.24.3",
        "pandera==0.17.2",
        "minio==7.2.0"
    ]
)
def validate_data_from_minio(
    input_object_name: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str,
    output_path: OutputPath(str)
) -> NamedTuple('Outputs', [('validated_object_name', str), ('is_valid', bool), ('num_records', int)]):
    """Validate data from MinIO using Pandera"""
    import pandas as pd
    import pandera as pa
    from pandera import Column, Check
    from minio import Minio
    from io import BytesIO
    from datetime import datetime
    from collections import namedtuple

    print(f"Validating data from MinIO: {input_object_name}")

    # Initialize MinIO client
    client = Minio(
        minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False
    )

    # Download data from MinIO
    response = client.get_object(bucket_name, input_object_name)
    df = pd.read_csv(BytesIO(response.read()))
    df['period'] = pd.to_datetime(df['period'])

    print(f"Loaded {len(df)} records from MinIO")

    # Define validation schema
    schema = pa.DataFrameSchema({
        'period': Column(pa.DateTime, nullable=False),
        'respondent': Column(pa.String, nullable=False),
        'type': Column(pa.String, nullable=False),
        'value': Column(
            pa.Float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(1000000)
            ],
            nullable=False
        )
    })

    # Validate
    try:
        validated_df = schema.validate(df)
        is_valid = True
        print("✓ Validation passed!")
    except pa.errors.SchemaError as e:
        print(f"✗ Validation failed: {e}")
        is_valid = False
        validated_df = df

    # Save validated data back to MinIO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    validated_object_name = f"processed/validated_data_{timestamp}.csv"

    csv_bytes = validated_df.to_csv(index=False).encode('utf-8')
    csv_buffer = BytesIO(csv_bytes)

    client.put_object(
        bucket_name,
        validated_object_name,
        csv_buffer,
        length=len(csv_bytes),
        content_type='text/csv'
    )

    print(f"Validated data saved to MinIO: {bucket_name}/{validated_object_name}")

    # Write output path
    with open(output_path, 'w') as f:
        f.write(validated_object_name)

    if not is_valid:
        raise ValueError("Data validation failed - check schema constraints")

    outputs = namedtuple('Outputs', ['validated_object_name', 'is_valid', 'num_records'])
    return outputs(validated_object_name, is_valid, len(validated_df))


@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "pandas==2.0.3",
        "numpy==1.24.3",
        "torch==2.1.0",
        "scikit-learn==1.3.2",
        "mlflow==2.9.2",
        "minio==7.2.0",
        "boto3==1.34.0"
    ]
)
def train_lstm_model(
    input_object_name: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str,
    mlflow_tracking_uri: str,
    experiment_name: str,
    # Hyperparameters
    model_type: str,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    sequence_length: int,
    prediction_horizon: int,
    # Output
    model_output_path: OutputPath(str)
) -> NamedTuple('Outputs', [('model_uri', str), ('test_rmse', float), ('test_mape', float), ('model_version', str)]):
    """Train LSTM/Transformer model with MLflow tracking and MinIO storage"""
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
    from datetime import datetime
    import os
    from collections import namedtuple

    print(f"Training {model_type.upper()} model...")
    print(f"Loading data from MinIO: {input_object_name}")

    # Configure MLflow to use MinIO for artifacts
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"http://{minio_endpoint}"
    os.environ['AWS_ACCESS_KEY_ID'] = minio_access_key
    os.environ['AWS_SECRET_ACCESS_KEY'] = minio_secret_key
    os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'

    # Set MLflow tracking
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Initialize MinIO client
    client = Minio(
        minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False
    )

    # Download data from MinIO
    response = client.get_object(bucket_name, input_object_name)
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

    # Create time series dataset
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

    # Split data
    train_size = int(0.7 * len(scaled_values))
    val_size = int(0.15 * len(scaled_values))

    train_data = scaled_values[:train_size]
    val_data = scaled_values[train_size:train_size + val_size]
    test_data = scaled_values[train_size + val_size:]

    train_dataset = TimeSeriesDataset(train_data, sequence_length, prediction_horizon)
    val_dataset = TimeSeriesDataset(val_data, sequence_length, prediction_horizon)
    test_dataset = TimeSeriesDataset(test_data, sequence_length, prediction_horizon)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Define model architecture
    if model_type.lower() == 'lstm':
        class LSTMForecaster(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
                super().__init__()
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
                return self.fc(last_hidden)

        model = LSTMForecaster(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=prediction_horizon,
            dropout=dropout
        )
    else:  # Transformer
        class TransformerForecaster(nn.Module):
            def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout):
                super().__init__()
                self.input_proj = nn.Linear(input_size, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.fc = nn.Linear(d_model, output_size)

            def forward(self, x):
                x = self.input_proj(x)
                transformer_out = self.transformer(x)
                last_hidden = transformer_out[:, -1, :]
                return self.fc(last_hidden)

        model = TransformerForecaster(
            input_size=1,
            d_model=hidden_size,
            nhead=4,
            num_layers=num_layers,
            output_size=prediction_horizon,
            dropout=dropout
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Start MLflow run
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("sequence_length", sequence_length)
        mlflow.log_param("prediction_horizon", prediction_horizon)
        mlflow.log_param("region", selected_region)

        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
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

            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), '/tmp/best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        model.load_state_dict(torch.load('/tmp/best_model.pt'))

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

        print(f"✓ Training complete!")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Test MAPE: {test_mape:.2f}%")

        # Save model with scaler to MLflow (will store in MinIO)
        mlflow.pytorch.log_model(model, "model")

        # Also save scaler
        import pickle
        with open('/tmp/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact('/tmp/scaler.pkl')

        # Register model
        model_uri = f"runs:/{run.info.run_id}/model"
        try:
            model_version = mlflow.register_model(model_uri, "electricity-load-forecaster")
            version_number = model_version.version
            print(f"✓ Model registered: electricity-load-forecaster v{version_number}")
        except Exception as e:
            print(f"Model registration warning: {e}")
            version_number = "unknown"

        # Save model info for next component
        with open(model_output_path, 'w') as f:
            f.write(model_uri)

        outputs = namedtuple('Outputs', ['model_uri', 'test_rmse', 'test_mape', 'model_version'])
        return outputs(model_uri, float(test_rmse), float(test_mape), str(version_number))


@dsl.pipeline(
    name='Electricity Load Forecasting Pipeline with MinIO',
    description='End-to-end pipeline with MinIO storage for all artifacts'
)
def electricity_forecasting_pipeline(
    eia_api_key: str,
    minio_endpoint: str = "minio.minio.svc.cluster.local:9000",
    minio_access_key: str = "minioadmin",
    minio_secret_key: str = "minioadmin",
    bucket_name: str = "electricity-data",
    mlflow_tracking_uri: str = "http://mlflow.mlflow.svc.cluster.local:5000",
    experiment_name: str = "electricity-load-forecasting",
    days: int = 90,
    # Model configuration
    model_type: str = "lstm",  # "lstm" or "transformer"
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 50,
    sequence_length: int = 168,
    prediction_horizon: int = 24
):
    """
    Kubeflow Pipeline for Electricity Load Forecasting
    All data stored in MinIO, models registered in MLflow
    """

    # Task 1: Extract data and save to MinIO
    extract_task = extract_data_to_minio(
        eia_api_key=eia_api_key,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name,
        days=days
    )

    # Task 2: Validate data from MinIO
    validate_task = validate_data_from_minio(
        input_object_name=extract_task.outputs['object_name'],
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name
    )

    # Task 3: Train model with data from MinIO
    train_task = train_lstm_model(
        input_object_name=validate_task.outputs['validated_object_name'],
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        model_type=model_type,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon
    )


if __name__ == '__main__':
    # Compile pipeline
    compiler.Compiler().compile(
        pipeline_func=electricity_forecasting_pipeline,
        package_path='electricity_forecasting_pipeline.yaml'
    )
    print("=" * 60)
    print("✓ Pipeline compiled successfully!")
    print("=" * 60)
    print("Output file: electricity_forecasting_pipeline.yaml")
    print("\nPipeline features:")
    print("  • Data extraction from EIA API → MinIO")
    print("  • Data validation with Pandera")
    print("  • Model training (LSTM/Transformer)")
    print("  • MLflow tracking with MinIO artifact storage")
    print("  • Model registration in MLflow")
    print("=" * 60)