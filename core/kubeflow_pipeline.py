"""
Kubeflow Pipeline - FINAL FIX
Short InferenceService name to avoid DNS 63-character limit
"""

from kfp import dsl, compiler
from kfp.dsl import component, OutputPath
from typing import NamedTuple

BASE_IMAGE = "python:3.10-slim"


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
    model_type: str,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    sequence_length: int,
    prediction_horizon: int,
    model_output_path: OutputPath(str)
) -> NamedTuple('Outputs', [('model_uri', str), ('test_rmse', float), ('test_mape', float), ('model_version', str), ('run_id', str)]):
    """Train LSTM model"""
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
    from collections import namedtuple

    print("Training LSTM Model")

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"http://{minio_endpoint}"
    os.environ['AWS_ACCESS_KEY_ID'] = minio_access_key
    os.environ['AWS_SECRET_ACCESS_KEY'] = minio_secret_key
    os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    client = Minio(minio_endpoint, access_key=minio_access_key, secret_key=minio_secret_key, secure=False)
    response = client.get_object(bucket_name, input_object_name)
    df = pd.read_csv(BytesIO(response.read()))
    df['period'] = pd.to_datetime(df['period'])

    selected_region = df['respondent'].value_counts().index[0]
    df = df[df['respondent'] == selected_region].copy().sort_values('period').reset_index(drop=True)

    print(f"Training on {selected_region}: {len(df)} records")

    values = df['value'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)

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

    class LSTMForecaster(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                              dropout=dropout if num_layers > 1 else 0, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMForecaster(1, hidden_size, num_layers, prediction_horizon, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run: {run_id}")

        mlflow.log_params({
            "model_type": model_type, "hidden_size": hidden_size, "num_layers": num_layers,
            "dropout": dropout, "learning_rate": learning_rate, "batch_size": batch_size,
            "epochs": epochs, "sequence_length": sequence_length, "prediction_horizon": prediction_horizon,
            "region": selected_region
        })

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x_batch), y_batch.squeeze())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    val_loss += criterion(model(x_batch), y_batch.squeeze()).item()
            val_loss /= len(val_loader)

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), '/tmp/best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        model.load_state_dict(torch.load('/tmp/best_model.pt'))

        model.eval()
        test_predictions, test_actuals = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                test_predictions.extend(model(x_batch.to(device)).cpu().numpy())
                test_actuals.extend(y_batch.numpy())

        test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
        test_actuals = scaler.inverse_transform(np.array(test_actuals).reshape(-1, 1))

        test_rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
        test_mape = mean_absolute_percentage_error(test_actuals, test_predictions) * 100

        mlflow.log_metrics({"test_rmse": test_rmse, "test_mape": test_mape, "best_val_loss": best_val_loss})

        print(f"Test RMSE: {test_rmse:.4f} MW, MAPE: {test_mape:.2f}%")

        mlflow.pytorch.log_model(model, "model")
        import pickle
        with open('/tmp/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact('/tmp/scaler.pkl')

        model_uri = f"runs:/{run_id}/model"
        try:
            model_version = mlflow.register_model(model_uri, "electricity-load-forecaster")
            version_number = str(model_version.version)
        except:
            version_number = "unknown"

        with open(model_output_path, 'w') as f:
            f.write(model_uri)

        outputs = namedtuple('Outputs', ['model_uri', 'test_rmse', 'test_mape', 'model_version', 'run_id'])
        return outputs(model_uri, float(test_rmse), float(test_mape), version_number, run_id)


@component(
    base_image=BASE_IMAGE,
    packages_to_install=["kubernetes==28.1.0"]
)
def deploy_to_kserve(
    model_uri: str,
    run_id: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str
) -> NamedTuple('Outputs', [('service_name', str), ('endpoint', str), ('status', str)]):
    """Deploy with SHORT name to avoid DNS limit"""
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    import time
    import base64
    from collections import namedtuple

    print("Deploying to KServe")

    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()

    with open('/var/run/secrets/kubernetes.io/serviceaccount/namespace', 'r') as f:
        namespace = f.read().strip()

    print(f"Namespace: {namespace}")

    custom_api = client.CustomObjectsApi()
    core_v1 = client.CoreV1Api()

    # SHORT NAME - only 8 chars + run_id prefix = max 16 chars total
    # This avoids the 63-char DNS limit
    isvc_name = f"elec-{run_id[:8]}"
    model_s3_path = f"s3://mlflow-artifacts/1/{run_id}/artifacts/model"

    print(f"InferenceService name: {isvc_name} (short to avoid DNS limit)")

    secret_name = "minio-s3-secret"
    secret_data = {
        "AWS_ACCESS_KEY_ID": base64.b64encode(minio_access_key.encode()).decode(),
        "AWS_SECRET_ACCESS_KEY": base64.b64encode(minio_secret_key.encode()).decode(),
        "AWS_ENDPOINT_URL": base64.b64encode(f"http://{minio_endpoint}".encode()).decode(),
        "S3_ENDPOINT": base64.b64encode(f"http://{minio_endpoint}".encode()).decode(),
        "S3_USE_HTTPS": base64.b64encode(b"0").decode(),
        "S3_VERIFY_SSL": base64.b64encode(b"0").decode()
    }

    secret = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": secret_name, "namespace": namespace},
        "type": "Opaque",
        "data": secret_data
    }

    try:
        core_v1.create_namespaced_secret(namespace, secret)
    except ApiException as e:
        if e.status == 409:
            core_v1.replace_namespaced_secret(secret_name, namespace, secret)

    # InferenceService WITHOUT RawDeployment mode (uses Knative)
    isvc = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": isvc_name,
            "namespace": namespace
        },
        "spec": {
            "predictor": {
                "model": {
                    "modelFormat": {"name": "mlflow"},
                    "runtime": "kserve-mlserver",
                    "storageUri": model_s3_path,
                    "resources": {
                        "requests": {"cpu": "100m", "memory": "512Mi"},
                        "limits": {"cpu": "1", "memory": "2Gi"}
                    },
                    "env": [
                        {"name": "MLSERVER_MODEL_URI", "value": model_s3_path},
                        {"name": "AWS_ACCESS_KEY_ID", "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "AWS_ACCESS_KEY_ID"}}},
                        {"name": "AWS_SECRET_ACCESS_KEY", "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "AWS_SECRET_ACCESS_KEY"}}},
                        {"name": "AWS_ENDPOINT_URL", "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "AWS_ENDPOINT_URL"}}},
                        {"name": "S3_USE_HTTPS", "value": "0"},
                        {"name": "S3_VERIFY_SSL", "value": "0"}
                    ]
                }
            }
        }
    }

    try:
        custom_api.create_namespaced_custom_object(
            "serving.kserve.io", "v1beta1", namespace, "inferenceservices", isvc
        )
        print(f"Created: {isvc_name}")
        status = "Created"
    except ApiException as e:
        if e.status == 409:
            custom_api.replace_namespaced_custom_object(
                "serving.kserve.io", "v1beta1", namespace, "inferenceservices", isvc_name, isvc
            )
            print("Updated")
            status = "Updated"
        else:
            print(f"Error: {e}")
            status = f"Failed: {e.reason}"
            raise

    print("Waiting for ready...")
    for i in range(30):
        try:
            obj = custom_api.get_namespaced_custom_object(
                "serving.kserve.io", "v1beta1", namespace, "inferenceservices", isvc_name
            )
            conditions = obj.get('status', {}).get('conditions', [])
            for cond in conditions:
                if cond.get('type') == 'Ready' and cond.get('status') == 'True':
                    print("Ready!")
                    status = "Ready"
                    break
        except:
            pass
        time.sleep(10)

    endpoint = f"http://{isvc_name}-predictor.{namespace}.svc.cluster.local/v2/models/{isvc_name}/infer"

    outputs = namedtuple('Outputs', ['service_name', 'endpoint', 'status'])
    return outputs(isvc_name, endpoint, status)


@dsl.pipeline(
    name='Electricity Forecasting Final',
    description='LSTM training and KServe deployment with short names'
)
def electricity_pipeline(
    input_object_name: str,
    minio_endpoint: str = "minio.minio.svc.cluster.local:9000",
    minio_access_key: str = "minioadmin",
    minio_secret_key: str = "minioadmin",
    bucket_name: str = "electricity-data",
    mlflow_tracking_uri: str = "http://mlflow.mlflow.svc.cluster.local:5000",
    experiment_name: str = "electricity-load-forecasting",
    model_type: str = "lstm",
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 50,
    sequence_length: int = 168,
    prediction_horizon: int = 24
):
    train_task = train_lstm_model(
        input_object_name=input_object_name,
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

    deploy_task = deploy_to_kserve(
        model_uri=train_task.outputs['model_uri'],
        run_id=train_task.outputs['run_id'],
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key
    )


if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=electricity_pipeline,
        package_path='electricity_forecasting_pipeline.yaml'
    )
    print("Pipeline compiled with SHORT InferenceService names!")
    print("Name format: elec-XXXXXXXX (avoids 63-char DNS limit)")