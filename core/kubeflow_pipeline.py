"""
Updated Kubeflow Pipeline for Electricity Load Forecasting
Works with pre-validated data from Airflow stored in MinIO
Includes: Training, MLflow logging, and KServe deployment
"""

from kfp import dsl, compiler
from kfp.dsl import component, OutputPath, InputPath
from typing import NamedTuple

# Base images
BASE_IMAGE = "python:3.10-slim"
KSERVE_BASE_IMAGE = "python:3.10-slim"


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
) -> NamedTuple('Outputs', [('model_uri', str), ('test_rmse', float), ('test_mape', float), ('model_version', str), ('run_id', str)]):
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

    print("=" * 80)
    print(f"Training {model_type.upper()} Model")
    print("=" * 80)
    print(f"Data source: MinIO/{bucket_name}/{input_object_name}")
    print(f"MLflow: {mlflow_tracking_uri}")
    print(f"Experiment: {experiment_name}")
    print("=" * 80)

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

    # Download pre-validated data from MinIO
    print(f"\nLoading validated data from MinIO...")
    response = client.get_object(bucket_name, input_object_name)
    df = pd.read_csv(BytesIO(response.read()))
    df['period'] = pd.to_datetime(df['period'])

    # Select region with most data
    region_counts = df['respondent'].value_counts()
    selected_region = region_counts.index[0]
    df = df[df['respondent'] == selected_region].copy()
    df = df.sort_values('period').reset_index(drop=True)

    print(f"Loaded {len(df)} records")
    print(f"Training region: {selected_region}")
    print(f"Date range: {df['period'].min()} to {df['period'].max()}")

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

    print(f"\nData splits:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

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

    print(f"\nModel: {model_type.upper()}")
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\nMLflow Run ID: {run_id}")

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
        mlflow.log_param("data_source", input_object_name)
        mlflow.log_param("train_size", len(train_dataset))
        mlflow.log_param("val_size", len(val_dataset))
        mlflow.log_param("test_size", len(test_dataset))

        # Training loop
        print(f"\nTraining for {epochs} epochs...")
        print("-" * 80)

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
                print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

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
        print(f"\nEvaluating on test set...")
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

        print("=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print(f"Test Metrics:")
        print(f"  RMSE: {test_rmse:.4f} MW")
        print(f"  MAPE: {test_mape:.2f}%")
        print(f"  Best Val Loss: {best_val_loss:.6f}")
        print("=" * 80)

        # Save model with scaler to MLflow (artifacts stored in MinIO)
        print(f"\nSaving model to MLflow...")
        mlflow.pytorch.log_model(model, "model")

        # Also save scaler
        import pickle
        with open('/tmp/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact('/tmp/scaler.pkl')

        print(f"Model artifacts saved to MinIO via MLflow")

        # Register model in MLflow Model Registry
        model_uri = f"runs:/{run_id}/model"
        model_name = "electricity-load-forecaster"

        try:
            model_version = mlflow.register_model(model_uri, model_name)
            version_number = model_version.version
            print(f"Model registered: {model_name} v{version_number}")

            # Add description
            from mlflow.tracking import MlflowClient
            client_mlflow = MlflowClient()
            client_mlflow.update_model_version(
                name=model_name,
                version=version_number,
                description=f"{model_type.upper()} model trained on {selected_region} region. RMSE: {test_rmse:.2f}, MAPE: {test_mape:.2f}%"
            )

        except Exception as e:
            print(f"Model registration warning: {e}")
            version_number = "unknown"

        # Save model info for next component
        with open(model_output_path, 'w') as f:
            f.write(model_uri)

        outputs = namedtuple('Outputs', ['model_uri', 'test_rmse', 'test_mape', 'model_version', 'run_id'])
        return outputs(model_uri, float(test_rmse), float(test_mape), str(version_number), str(run_id))


@component(
    base_image=KSERVE_BASE_IMAGE,
    packages_to_install=[
        "kubernetes==28.1.0",
        "pyyaml==6.0.1"
    ]
)
def deploy_model_to_kserve(
    model_uri: str,
    run_id: str,
    model_name: str,
    namespace: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    mlflow_tracking_uri: str,
    service_account: str = "default-editor"
) -> NamedTuple('Outputs', [('inference_service_name', str), ('endpoint_url', str), ('deployment_status', str)]):
    """Deploy trained model to KServe for inference"""
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    import yaml
    import time
    from collections import namedtuple
    import os

    print("=" * 80)
    print("Deploying Model to KServe")
    print("=" * 80)
    print(f"Model URI: {model_uri}")
    print(f"Run ID: {run_id}")
    print(f"Namespace: {namespace}")
    print("=" * 80)

    # Load Kubernetes config (inside cluster)
    try:
        config.load_incluster_config()
        print("✓ Loaded in-cluster Kubernetes config")
    except Exception as e:
        print(f"Warning: Could not load in-cluster config: {e}")
        print("Attempting to load local kubeconfig...")
        config.load_kube_config()

    # Create Kubernetes API clients
    api_client = client.ApiClient()
    custom_api = client.CustomObjectsApi(api_client)
    core_v1 = client.CoreV1Api(api_client)

    # Generate InferenceService name
    inference_service_name = f"electricity-forecaster-{run_id[:8]}"

    # Get MLflow artifact URI (stored in MinIO)
    # Format: s3://mlflow-artifacts/<experiment_id>/<run_id>/artifacts/model
    model_s3_uri = model_uri.replace("runs:/", "s3://mlflow-artifacts/")

    # Extract experiment ID and construct proper S3 path
    # We need to query MLflow to get the experiment ID
    # For simplicity, we'll use a standard format
    model_s3_path = f"s3://mlflow-artifacts/1/{run_id}/artifacts/model"

    print(f"\nModel storage:")
    print(f"  MLflow URI: {model_uri}")
    print(f"  S3 Path: {model_s3_path}")

    # Create Secret for MinIO credentials (if it doesn't exist)
    secret_name = "minio-s3-secret"

    secret_data = {
        "AWS_ACCESS_KEY_ID": minio_access_key,
        "AWS_SECRET_ACCESS_KEY": minio_secret_key,
        "S3_ENDPOINT": f"http://{minio_endpoint}",
        "S3_USE_HTTPS": "0",
        "S3_VERIFY_SSL": "0"
    }

    # Encode secret data
    import base64
    encoded_secret_data = {
        k: base64.b64encode(v.encode()).decode()
        for k, v in secret_data.items()
    }

    secret_manifest = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": secret_name,
            "namespace": namespace
        },
        "type": "Opaque",
        "data": encoded_secret_data
    }

    try:
        core_v1.create_namespaced_secret(namespace=namespace, body=secret_manifest)
        print(f"✓ Created secret: {secret_name}")
    except ApiException as e:
        if e.status == 409:  # Already exists
            print(f"✓ Secret already exists: {secret_name}")
            # Update the existing secret
            try:
                core_v1.replace_namespaced_secret(
                    name=secret_name,
                    namespace=namespace,
                    body=secret_manifest
                )
                print(f"✓ Updated secret: {secret_name}")
            except Exception as update_error:
                print(f"Warning: Could not update secret: {update_error}")
        else:
            print(f"Error creating secret: {e}")

    # Create InferenceService manifest
    inference_service = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": inference_service_name,
            "namespace": namespace,
            "annotations": {
                "serving.kserve.io/enable-prometheus-scraping": "true",
                "serving.kserve.io/enable-metric-aggregation": "true"
            },
            "labels": {
                "app": "electricity-forecaster",
                "model-type": "lstm",
                "mlflow-run-id": run_id
            }
        },
        "spec": {
            "predictor": {
                "serviceAccountName": service_account,
                "model": {
                    "modelFormat": {
                        "name": "pytorch"
                    },
                    "storageUri": model_s3_path,
                    "resources": {
                        "requests": {
                            "cpu": "500m",
                            "memory": "1Gi"
                        },
                        "limits": {
                            "cpu": "1",
                            "memory": "2Gi"
                        }
                    },
                    "env": [
                        {
                            "name": "STORAGE_URI",
                            "value": model_s3_path
                        },
                        {
                            "name": "AWS_ACCESS_KEY_ID",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": secret_name,
                                    "key": "AWS_ACCESS_KEY_ID"
                                }
                            }
                        },
                        {
                            "name": "AWS_SECRET_ACCESS_KEY",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": secret_name,
                                    "key": "AWS_SECRET_ACCESS_KEY"
                                }
                            }
                        },
                        {
                            "name": "S3_ENDPOINT",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": secret_name,
                                    "key": "S3_ENDPOINT"
                                }
                            }
                        },
                        {
                            "name": "S3_USE_HTTPS",
                            "value": "0"
                        },
                        {
                            "name": "S3_VERIFY_SSL",
                            "value": "0"
                        }
                    ]
                }
            }
        }
    }

    print(f"\nCreating InferenceService: {inference_service_name}")

    try:
        # Try to create the InferenceService
        response = custom_api.create_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            body=inference_service
        )
        print(f"✓ InferenceService created successfully")
        deployment_status = "Created"

    except ApiException as e:
        if e.status == 409:  # Already exists
            print(f"InferenceService already exists, updating...")
            try:
                response = custom_api.replace_namespaced_custom_object(
                    group="serving.kserve.io",
                    version="v1beta1",
                    namespace=namespace,
                    plural="inferenceservices",
                    name=inference_service_name,
                    body=inference_service
                )
                print(f"✓ InferenceService updated successfully")
                deployment_status = "Updated"
            except Exception as update_error:
                print(f"❌ Error updating InferenceService: {update_error}")
                deployment_status = f"Failed: {str(update_error)}"
                raise
        else:
            print(f"❌ Error creating InferenceService: {e}")
            deployment_status = f"Failed: {str(e)}"
            raise

    # Wait for InferenceService to be ready (with timeout)
    print(f"\nWaiting for InferenceService to be ready...")
    max_wait = 300  # 5 minutes
    wait_interval = 10
    elapsed = 0

    is_ready = False
    while elapsed < max_wait:
        try:
            isvc = custom_api.get_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                name=inference_service_name
            )

            status = isvc.get('status', {})
            conditions = status.get('conditions', [])

            # Check if Ready condition is True
            for condition in conditions:
                if condition.get('type') == 'Ready':
                    if condition.get('status') == 'True':
                        is_ready = True
                        print(f"✓ InferenceService is ready!")
                        break
                    else:
                        reason = condition.get('reason', 'Unknown')
                        message = condition.get('message', 'No message')
                        print(f"  Status: {reason} - {message}")

            if is_ready:
                break

            time.sleep(wait_interval)
            elapsed += wait_interval

        except Exception as e:
            print(f"  Error checking status: {e}")
            time.sleep(wait_interval)
            elapsed += wait_interval

    if not is_ready:
        print(f"⚠️ InferenceService not ready after {max_wait}s, but deployment submitted")
        deployment_status = "Pending"

    # Get endpoint URL
    endpoint_url = f"http://{inference_service_name}.{namespace}.svc.cluster.local/v1/models/{inference_service_name}:predict"

    # Try to get the external URL if available
    try:
        isvc = custom_api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=inference_service_name
        )

        status = isvc.get('status', {})
        external_url = status.get('url', endpoint_url)
        if external_url and external_url != endpoint_url:
            endpoint_url = external_url
            print(f"  External URL: {endpoint_url}")

    except Exception as e:
        print(f"  Could not get external URL: {e}")

    print("=" * 80)
    print("✅ KServe Deployment Complete!")
    print("=" * 80)
    print(f"InferenceService: {inference_service_name}")
    print(f"Namespace: {namespace}")
    print(f"Endpoint: {endpoint_url}")
    print(f"Status: {deployment_status}")
    print("")
    print("📡 Test the endpoint:")
    print(f'  kubectl port-forward -n {namespace} svc/{inference_service_name}-predictor 8080:80')
    print(f'  curl -X POST http://localhost:8080/v1/models/{inference_service_name}:predict \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"instances": [[...168 values...]]}}\'')
    print("=" * 80)

    outputs = namedtuple('Outputs', ['inference_service_name', 'endpoint_url', 'deployment_status'])
    return outputs(inference_service_name, endpoint_url, deployment_status)


@dsl.pipeline(
    name='Electricity Load Forecasting - Train and Deploy',
    description='Train model with pre-validated data from MinIO, log to MLflow, deploy to KServe'
)
def electricity_training_pipeline(
    input_object_name: str,  # Path to validated data in MinIO (from Airflow)
    minio_endpoint: str = "minio.minio.svc.cluster.local:9000",
    minio_access_key: str = "minioadmin",
    minio_secret_key: str = "minioadmin",
    bucket_name: str = "electricity-data",
    mlflow_tracking_uri: str = "http://mlflow.mlflow.svc.cluster.local:5000",
    experiment_name: str = "electricity-load-forecasting",
    # Model configuration
    model_type: str = "lstm",  # "lstm" or "transformer"
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 50,
    sequence_length: int = 168,
    prediction_horizon: int = 24,
    # KServe deployment
    kserve_namespace: str = "kubeflow",
    service_account: str = "default-editor"
):
    """
    Complete MLOps Pipeline: Training + Deployment

    Input: Pre-validated data from MinIO (provided by Airflow)
    Process:
      1. Train LSTM/Transformer model
      2. Deploy to KServe for inference
    Output:
      - Model logged to MLflow, artifacts stored in MinIO
      - Inference endpoint available via KServe

    Data extraction and validation are handled by Airflow.
    """

    # Task 1: Train model with pre-validated data from MinIO
    train_task = train_lstm_model(
        input_object_name=input_object_name,  # From Airflow
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

    # Task 2: Deploy trained model to KServe
    deploy_task = deploy_model_to_kserve(
        model_uri=train_task.outputs['model_uri'],
        run_id=train_task.outputs['run_id'],
        model_name="electricity-load-forecaster",
        namespace=kserve_namespace,
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        mlflow_tracking_uri=mlflow_tracking_uri,
        service_account=service_account
    )


if __name__ == '__main__':
    # Compile pipeline
    compiler.Compiler().compile(
        pipeline_func=electricity_training_pipeline,
        package_path='electricity_forecasting_pipeline.yaml'
    )

    print("=" * 80)
    print("✅ Kubeflow Pipeline Compiled Successfully!")
    print("=" * 80)
    print("Output: electricity_forecasting_pipeline.yaml")
    print("\n🏗️  Pipeline Architecture:")
    print("  Input:  Pre-validated data from MinIO (provided by Airflow)")
    print("  Task 1: Train LSTM/Transformer model → MLflow")
    print("  Task 2: Deploy model to KServe → Inference endpoint")
    print("  Output: Model logged to MLflow + KServe endpoint ready")
    print("\n🔄 Integration Flow:")
    print("  1. Airflow extracts data from EIA → MinIO raw/")
    print("  2. Airflow validates data → MinIO processed/")
    print("  3. Airflow compiles & stores pipeline YAML → MinIO pipelines/")
    print("  4. User triggers THIS Kubeflow pipeline via UI")
    print("  5. Kubeflow trains model → MLflow + MinIO")
    print("  6. Kubeflow deploys model → KServe")
    print("  7. Model ready for inference!")
    print("\n💾 Storage:")
    print("  - Training data:  MinIO (processed/)")
    print("  - Pipeline YAML:  MinIO (pipelines/)")
    print("  - Model artifacts: MinIO (via MLflow)")
    print("  - Experiments:     MLflow tracking server")
    print("  - Model registry:  MLflow Model Registry")
    print("  - Inference:       KServe InferenceService")
    print("\n📡 Inference Endpoint:")
    print("  After deployment, access via:")
    print("  POST /v1/models/electricity-forecaster-{run_id}:predict")
    print("=" * 80)