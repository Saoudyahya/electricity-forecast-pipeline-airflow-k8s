"""
Updated Airflow DAG for MLOps Pipeline
Integrates:
- Kubeflow Pipelines for training
- Katib for HPO
- MLflow for model registry
- EvidentlyAI for drift detection
- Daily batch inference
"""

from airflow import DAG
try:
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python import PythonOperator

from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import yaml
import logging
import pandas as pd
from io import BytesIO
import sys
import os
import json
import numpy as np

# Fix imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

# Load config
config_path = None
possible_paths = [
    '/opt/airflow/config/config.yaml',
    'config.yaml',
    '../config.yaml',
    os.path.join(parent_dir, 'config.yaml'),
    os.path.join(current_dir, '..', 'config.yaml')
]

for path in possible_paths:
    if os.path.exists(path):
        config_path = path
        break

if config_path is None:
    raise FileNotFoundError("config.yaml not found")

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Default args
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}


def trigger_kubeflow_pipeline(**context):
    """Trigger Kubeflow Pipeline for model training"""
    import kfp
    from datetime import datetime

    logger.info("Triggering Kubeflow Pipeline for model training")

    # Connect to Kubeflow Pipelines
    client = kfp.Client(
        host=config['kubeflow']['pipeline_host']
    )

    # Pipeline parameters
    pipeline_params = {
        'eia_api_key': os.getenv('EIA_API_KEY'),
        'minio_endpoint': config['storage']['minio_endpoint'],
        'minio_access_key': config['storage']['minio_access_key'],
        'minio_secret_key': config['storage']['minio_secret_key'],
        'bucket_name': config['storage']['bucket_name'],
        'mlflow_tracking_uri': config['mlflow']['tracking_uri'],
        'experiment_name': config['mlflow']['experiment_name'],
        'days': 90,
        'hidden_size': config['model']['hidden_size'],
        'num_layers': config['model']['num_layers'],
        'dropout': config['model']['dropout'],
        'learning_rate': config['model']['learning_rate'],
        'batch_size': config['model']['batch_size'],
        'epochs': config['model']['epochs'],
        'sequence_length': config['model']['sequence_length'],
        'prediction_horizon': config['model']['prediction_horizon']
    }

    # Create run
    run_name = f"training-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Submit pipeline
    run = client.create_run_from_pipeline_package(
        pipeline_file='electricity_forecasting_pipeline.yaml',
        arguments=pipeline_params,
        run_name=run_name,
        experiment_name=config['mlflow']['experiment_name']
    )

    logger.info(f"Pipeline run created: {run.run_id}")

    # Wait for completion (optional - can be async)
    # run_detail = client.wait_for_run_completion(run.run_id, timeout=3600)

    # Push run ID to XCom
    context['task_instance'].xcom_push(key='pipeline_run_id', value=run.run_id)

    return run.run_id


def trigger_katib_hpo(**context):
    """Trigger Katib HPO experiment"""
    import subprocess
    import time

    logger.info("Triggering Katib HPO experiment")

    # Apply Katib experiment
    result = subprocess.run(
        ['kubectl', 'apply', '-f', '/opt/airflow/dags/katib-experiment.yaml'],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        logger.error(f"Failed to create Katib experiment: {result.stderr}")
        raise Exception(f"Katib experiment creation failed: {result.stderr}")

    logger.info("Katib experiment created successfully")

    # Optional: Wait for completion
    # This is a simplified version - in production, use proper Katib API
    time.sleep(60)  # Wait a bit for experiment to start

    # Get experiment status
    status_result = subprocess.run(
        ['kubectl', 'get', 'experiment', 'electricity-forecast-hpo', '-n', 'kubeflow', '-o', 'json'],
        capture_output=True,
        text=True
    )

    if status_result.returncode == 0:
        experiment_status = json.loads(status_result.stdout)
        logger.info(f"Experiment status: {experiment_status.get('status', {})}")

    return "Katib experiment triggered"


def get_latest_model_from_mlflow(**context):
    """Get the latest production model from MLflow"""
    import mlflow
    from mlflow.tracking import MlflowClient

    logger.info("Fetching latest model from MLflow")

    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    client = MlflowClient()

    model_name = "electricity-load-forecaster"

    # Get latest production model
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            logger.warning("No model versions found in MLflow")
            return None

        # Get production or latest version
        production_versions = [v for v in versions if v.current_stage == 'Production']

        if production_versions:
            latest_version = max(production_versions, key=lambda x: int(x.version))
        else:
            latest_version = max(versions, key=lambda x: int(x.version))

        logger.info(f"Using model version: {latest_version.version}, Stage: {latest_version.current_stage}")

        # Download model
        model_uri = f"models:/{model_name}/{latest_version.version}"
        model_path = f"/tmp/model_{latest_version.version}"

        mlflow.pytorch.load_model(model_uri)

        # Push to XCom
        context['task_instance'].xcom_push(key='model_version', value=latest_version.version)
        context['task_instance'].xcom_push(key='model_uri', value=model_uri)

        return model_uri

    except Exception as e:
        logger.error(f"Error fetching model from MLflow: {e}")
        raise


def batch_inference(**context):
    """Run batch inference using latest model from MLflow"""
    import mlflow
    import mlflow.pytorch
    import torch
    from minio import Minio

    logger.info("Starting batch inference")

    # Get model from previous task
    model_uri = context['task_instance'].xcom_pull(
        task_ids='get_latest_model',
        key='model_uri'
    )

    if model_uri is None:
        logger.warning("No model available, skipping inference")
        return "No model available"

    # Load model from MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    model = mlflow.pytorch.load_model(model_uri)

    # Get latest data from MinIO
    client = Minio(
        config['storage']['minio_endpoint'],
        access_key=config['storage']['minio_access_key'],
        secret_key=config['storage']['minio_secret_key'],
        secure=False
    )

    # Get latest processed data
    objects = client.list_objects(
        config['storage']['bucket_name'],
        prefix=config['storage']['processed_data_prefix']
    )
    objects_list = list(objects)

    if not objects_list:
        logger.warning("No processed data found")
        return "No data available"

    latest_object = max(objects_list, key=lambda x: x.last_modified)

    response = client.get_object(
        config['storage']['bucket_name'],
        latest_object.object_name
    )
    df = pd.read_csv(BytesIO(response.read()))
    df['period'] = pd.to_datetime(df['period'])

    # Select region
    if 'respondent' in df.columns:
        region_counts = df['respondent'].value_counts()
        selected_region = region_counts.index[0]
        df = df[df['respondent'] == selected_region].copy()

    # Prepare input (last sequence_length hours)
    sequence_length = config['model']['sequence_length']
    input_data = df['value'].values[-sequence_length:].reshape(-1, 1)

    # Scale data (load scaler from model artifacts)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(df['value'].values.reshape(-1, 1))
    input_scaled = scaler.transform(input_data)

    # Make predictions
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0)
        predictions = model(input_tensor).numpy().flatten()

    # Inverse transform
    predictions_scaled = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    # Create predictions dataframe
    last_timestamp = df['period'].iloc[-1]
    future_timestamps = pd.date_range(
        start=last_timestamp + timedelta(hours=1),
        periods=len(predictions_scaled),
        freq='H'
    )

    predictions_df = pd.DataFrame({
        'timestamp': future_timestamps,
        'predicted_load': predictions_scaled
    })

    # Save predictions to MinIO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_path = f"{config['storage']['predictions_prefix']}predictions_{timestamp}.csv"

    csv_bytes = predictions_df.to_csv(index=False).encode('utf-8')
    csv_buffer = BytesIO(csv_bytes)

    client.put_object(
        config['storage']['bucket_name'],
        predictions_path,
        csv_buffer,
        length=len(csv_bytes),
        content_type='text/csv'
    )

    logger.info(f"Predictions saved to MinIO: {predictions_path}")

    # Push to XCom
    context['task_instance'].xcom_push(key='predictions_path', value=predictions_path)
    context['task_instance'].xcom_push(key='num_predictions', value=len(predictions_df))

    return predictions_path


def detect_drift(**context):
    """Detect data and model drift using EvidentlyAI"""
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, RegressionPreset
    from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesMetric
    from minio import Minio
    import json

    logger.info("Starting drift detection")

    client = Minio(
        config['storage']['minio_endpoint'],
        access_key=config['storage']['minio_access_key'],
        secret_key=config['storage']['minio_secret_key'],
        secure=False
    )

    # Get reference data (older data)
    reference_days = config['drift_detection']['reference_window_days']
    current_days = config['drift_detection']['current_window_days']

    # Get processed data
    objects = list(client.list_objects(
        config['storage']['bucket_name'],
        prefix=config['storage']['processed_data_prefix']
    ))

    if len(objects) < 2:
        logger.warning("Not enough data for drift detection")
        return "Insufficient data"

    # Sort by date
    objects_sorted = sorted(objects, key=lambda x: x.last_modified, reverse=True)

    # Get current data
    current_object = objects_sorted[0]
    response = client.get_object(config['storage']['bucket_name'], current_object.object_name)
    current_df = pd.read_csv(BytesIO(response.read()))
    current_df['period'] = pd.to_datetime(current_df['period'])

    # Get reference data
    reference_object = objects_sorted[-1]  # Oldest data
    response = client.get_object(config['storage']['bucket_name'], reference_object.object_name)
    reference_df = pd.read_csv(BytesIO(response.read()))
    reference_df['period'] = pd.to_datetime(reference_df['period'])

    # Filter to same region
    if 'respondent' in current_df.columns:
        region = current_df['respondent'].value_counts().index[0]
        current_df = current_df[current_df['respondent'] == region]
        reference_df = reference_df[reference_df['respondent'] == region]

    # Select features for drift detection
    feature_columns = ['value']

    # Create column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = 'value'

    # Create drift report
    drift_report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
    ])

    drift_report.run(
        reference_data=reference_df[feature_columns],
        current_data=current_df[feature_columns],
        column_mapping=column_mapping
    )

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"/tmp/drift_report_{timestamp}.html"
    drift_report.save_html(report_path)

    # Get drift metrics as dict
    report_dict = drift_report.as_dict()

    # Check for drift
    drift_detected = False
    drift_share = 0.0

    try:
        dataset_drift = report_dict['metrics'][1]['result']
        drift_detected = dataset_drift.get('dataset_drift', False)
        drift_share = dataset_drift.get('drift_share', 0.0)
    except (KeyError, IndexError):
        logger.warning("Could not parse drift metrics")

    # Save report to MinIO
    with open(report_path, 'rb') as f:
        client.put_object(
            config['storage']['bucket_name'],
            f"drift_reports/drift_report_{timestamp}.html",
            f,
            length=os.path.getsize(report_path),
            content_type='text/html'
        )

    # Save drift metrics
    drift_metrics = {
        'timestamp': timestamp,
        'drift_detected': drift_detected,
        'drift_share': float(drift_share),
        'threshold': config['drift_detection']['drift_threshold'],
        'reference_size': len(reference_df),
        'current_size': len(current_df)
    }

    metrics_bytes = json.dumps(drift_metrics, indent=2).encode('utf-8')
    metrics_buffer = BytesIO(metrics_bytes)

    client.put_object(
        config['storage']['bucket_name'],
        f"drift_reports/drift_metrics_{timestamp}.json",
        metrics_buffer,
        length=len(metrics_bytes),
        content_type='application/json'
    )

    logger.info(f"Drift detection complete. Drift detected: {drift_detected}, Drift share: {drift_share:.2%}")

    # Push to XCom
    context['task_instance'].xcom_push(key='drift_detected', value=drift_detected)
    context['task_instance'].xcom_push(key='drift_share', value=drift_share)

    # Alert if drift detected
    if drift_detected or drift_share > config['drift_detection']['drift_threshold']:
        logger.warning(f"⚠️ DRIFT ALERT: Data drift detected! Drift share: {drift_share:.2%}")
        # In production, send alert via email/Slack/etc.

    return drift_metrics


# Define DAG for weekly training
with DAG(
    'electricity_training_pipeline',
    default_args=default_args,
    description='Weekly model training with Kubeflow and Katib',
    schedule='0 0 * * 0',  # Every Sunday at midnight
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['mlops', 'training', 'kubeflow', 'katib'],
) as training_dag:

    # Trigger Kubeflow Pipeline
    trigger_kfp = PythonOperator(
        task_id='trigger_kubeflow_pipeline',
        python_callable=trigger_kubeflow_pipeline,
    )

    # Optionally trigger Katib HPO (monthly)
    trigger_hpo = PythonOperator(
        task_id='trigger_katib_hpo',
        python_callable=trigger_katib_hpo,
    )

    trigger_kfp >> trigger_hpo


# Define DAG for daily inference and drift detection
with DAG(
    'electricity_daily_inference',
    default_args=default_args,
    description='Daily batch inference and drift detection',
    schedule='0 2 * * *',  # Every day at 2 AM
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['mlops', 'inference', 'drift-detection'],
) as inference_dag:

    # Get latest model from MLflow
    get_model = PythonOperator(
        task_id='get_latest_model',
        python_callable=get_latest_model_from_mlflow,
    )

    # Run batch inference
    inference = PythonOperator(
        task_id='batch_inference',
        python_callable=batch_inference,
    )

    # Detect drift
    drift_detection = PythonOperator(
        task_id='detect_drift',
        python_callable=detect_drift,
    )

    get_model >> inference >> drift_detection


if __name__ == "__main__":
    print("=" * 60)
    print("✓ Updated DAGs loaded successfully!")
    print("=" * 60)
    print("\nDAG 1: electricity_training_pipeline")
    print("  Schedule: Weekly (Sundays)")
    print("  Tasks:")
    print("    1. trigger_kubeflow_pipeline")
    print("    2. trigger_katib_hpo")
    print("\nDAG 2: electricity_daily_inference")
    print("  Schedule: Daily (2 AM)")
    print("  Tasks:")
    print("    1. get_latest_model")
    print("    2. batch_inference")
    print("    3. detect_drift")
    print("=" * 60)