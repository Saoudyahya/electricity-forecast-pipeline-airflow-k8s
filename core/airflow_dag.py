"""
Airflow DAG for Electricity Load Forecasting Pipeline

DAG Structure:
1. Extract data from EIA API
2. Validate data with Pandera
3. Train model
4. Batch inference
"""

from airflow import DAG
try:
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python import PythonOperator

from datetime import datetime, timedelta
import yaml
import logging
import pandas as pd
from io import BytesIO
import sys
import os
import pandera
# Fix imports - when running from core/ directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add both current and parent directory to path
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# Import with fallback
try:
    from data_extraction import EIADataExtractor
    from data_validation import ElectricityDataValidator
    from model import ElectricityLoadForecaster
except ImportError:
    try:
        from core.data_extraction import EIADataExtractor
        from core.data_validation import ElectricityDataValidator
        from core.model import ElectricityLoadForecaster
    except ImportError:
        sys.path.insert(0, '/opt/airflow/dags/core')
        from data_extraction import EIADataExtractor
        from data_validation import ElectricityDataValidator
        from model import ElectricityLoadForecaster

logger = logging.getLogger(__name__)

# Load config - try multiple locations
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
    raise FileNotFoundError("config.yaml not found in any expected location")

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


def extract_data(**context):
    """Extract data from EIA API"""
    logger.info("Starting data extraction from EIA API")

    extractor = EIADataExtractor(config_path=config_path)

    # Fetch last 90 days of data for all regions
    df = extractor.fetch_recent_data(
        days=90,
        regions=None  # Fetch all regions
    )

    # Save locally first
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_path = f"/tmp/electricity_data_{timestamp}.csv"
    extractor.save_to_csv(df, local_path)

    # Try to save to MinIO if available
    try:
        object_name = f"{config['storage']['raw_data_prefix']}electricity_data_{timestamp}.csv"
        extractor.save_to_minio(df, object_name)
        logger.info(f"Data saved to MinIO: {object_name}")
        data_path = object_name
    except Exception as e:
        logger.warning(f"MinIO not available, using local storage: {e}")
        data_path = local_path

    # Push to XCom for next task
    context['task_instance'].xcom_push(key='data_path', value=data_path)
    context['task_instance'].xcom_push(key='local_path', value=local_path)
    context['task_instance'].xcom_push(key='num_records', value=len(df))

    logger.info(f"Extracted {len(df)} records")

    return data_path


def validate_data(**context):
    """Validate data using Pandera"""
    logger.info("Starting data validation")

    # Get data path from previous task
    data_path = context['task_instance'].xcom_pull(
        task_ids='extract_data',
        key='data_path'
    )
    local_path = context['task_instance'].xcom_pull(
        task_ids='extract_data',
        key='local_path'
    )

    # Try to load from MinIO, fallback to local
    try:
        from minio import Minio

        client = Minio(
            config['storage']['minio_endpoint'],
            access_key=config['storage']['minio_access_key'],
            secret_key=config['storage']['minio_secret_key'],
            secure=False
        )

        response = client.get_object(config['storage']['bucket_name'], data_path)
        df = pd.read_csv(BytesIO(response.read()))
        logger.info("Loaded data from MinIO")
    except Exception as e:
        logger.warning(f"Could not load from MinIO, using local file: {e}")
        df = pd.read_csv(local_path)

    df['period'] = pd.to_datetime(df['period'])

    # Validate
    validator = ElectricityDataValidator(config_path=config_path)
    validated_df, report = validator.validate(df)

    if not report['is_valid']:
        raise ValueError(f"Data validation failed: {report['errors']}")

    # Save validated data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    validated_local_path = f"/tmp/validated_data_{timestamp}.csv"
    validated_df.to_csv(validated_local_path, index=False)

    # Save validation report
    report_local_path = f"/tmp/validation_report_{timestamp}.json"
    import json
    with open(report_local_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Try to upload to MinIO if available
    try:
        from minio import Minio

        client = Minio(
            config['storage']['minio_endpoint'],
            access_key=config['storage']['minio_access_key'],
            secret_key=config['storage']['minio_secret_key'],
            secure=False
        )

        # Upload validated data
        validated_path = f"{config['storage']['processed_data_prefix']}validated_data_{timestamp}.csv"
        csv_bytes = validated_df.to_csv(index=False).encode('utf-8')
        csv_buffer = BytesIO(csv_bytes)

        client.put_object(
            config['storage']['bucket_name'],
            validated_path,
            csv_buffer,
            length=len(csv_bytes),
            content_type='text/csv'
        )

        # Upload report
        report_path = f"{config['storage']['processed_data_prefix']}validation_report_{timestamp}.json"
        report_bytes = json.dumps(report, indent=2).encode('utf-8')
        report_buffer = BytesIO(report_bytes)

        client.put_object(
            config['storage']['bucket_name'],
            report_path,
            report_buffer,
            length=len(report_bytes),
            content_type='application/json'
        )

        logger.info("Uploaded validated data to MinIO")
    except Exception as e:
        logger.warning(f"MinIO upload failed, data saved locally: {e}")
        validated_path = validated_local_path

    # Push to XCom
    context['task_instance'].xcom_push(key='validated_data_path', value=validated_path)
    context['task_instance'].xcom_push(key='validated_local_path', value=validated_local_path)
    context['task_instance'].xcom_push(key='validation_report', value=report)

    logger.info("Data validation passed")

    return validated_path


def train_model(**context):
    """Train LSTM model"""
    logger.info("Starting model training")

    # Get validated data path
    validated_local_path = context['task_instance'].xcom_pull(
        task_ids='validate_data',
        key='validated_local_path'
    )

    # Load data
    df = pd.read_csv(validated_local_path)
    df['period'] = pd.to_datetime(df['period'])

    # Select one region for training
    if 'respondent' in df.columns:
        region_counts = df['respondent'].value_counts()
        selected_region = region_counts.index[0]
        df = df[df['respondent'] == selected_region].copy()
        logger.info(f"Training on region: {selected_region}")

    # Initialize forecaster
    forecaster = ElectricityLoadForecaster(
        model_type=config['model'].get('model_type', 'lstm'),
        sequence_length=config['model']['sequence_length'],
        prediction_horizon=config['model']['prediction_horizon'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model'].get('dropout', 0.2),
        learning_rate=config['model'].get('learning_rate', 0.001)
    )

    # Prepare data
    train_loader, val_loader, test_loader = forecaster.prepare_data(
        df,
        train_split=config['validation']['train_split'],
        val_split=config['validation']['val_split']
    )

    # Train
    forecaster.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['model']['epochs'],
        early_stopping_patience=10
    )

    # Evaluate
    test_rmse, test_mape = forecaster.evaluate(test_loader)
    logger.info(f"Test RMSE: {test_rmse:.4f}, Test MAPE: {test_mape:.2f}%")

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"/tmp/model_{timestamp}.pt"
    forecaster.save_model(model_path)

    # Push to XCom
    context['task_instance'].xcom_push(key='model_path', value=model_path)
    context['task_instance'].xcom_push(key='test_rmse', value=test_rmse)
    context['task_instance'].xcom_push(key='test_mape', value=test_mape)

    logger.info(f"Model training complete. Saved to {model_path}")

    return model_path


def batch_inference(**context):
    """Run batch inference on latest data"""
    logger.info("Starting batch inference")

    # Get paths
    validated_local_path = context['task_instance'].xcom_pull(
        task_ids='validate_data',
        key='validated_local_path'
    )
    model_path = context['task_instance'].xcom_pull(
        task_ids='train_model',
        key='model_path'
    )

    # Load data
    df = pd.read_csv(validated_local_path)
    df['period'] = pd.to_datetime(df['period'])

    # Select same region as training
    if 'respondent' in df.columns:
        region_counts = df['respondent'].value_counts()
        selected_region = region_counts.index[0]
        df = df[df['respondent'] == selected_region].copy()

    # Load model
    forecaster = ElectricityLoadForecaster(
        model_type='lstm',
        sequence_length=config['model']['sequence_length'],
        prediction_horizon=config['model']['prediction_horizon']
    )
    forecaster.load_model(model_path)

    # Prepare input sequence
    sequence_length = config['model']['sequence_length']
    input_sequence = df['value'].values[-sequence_length:].reshape(-1, 1)
    input_scaled = forecaster.scaler.transform(input_sequence)

    # Make predictions
    predictions = forecaster.predict(input_scaled)

    # Create predictions dataframe
    last_timestamp = df['period'].iloc[-1]
    future_timestamps = pd.date_range(
        start=last_timestamp + timedelta(hours=1),
        periods=len(predictions),
        freq='H'
    )

    predictions_df = pd.DataFrame({
        'timestamp': future_timestamps,
        'predicted_load': predictions
    })

    # Save predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_path = f"/tmp/predictions_{timestamp}.csv"
    predictions_df.to_csv(predictions_path, index=False)

    # Try to upload to MinIO if available
    try:
        from minio import Minio

        client = Minio(
            config['storage']['minio_endpoint'],
            access_key=config['storage']['minio_access_key'],
            secret_key=config['storage']['minio_secret_key'],
            secure=False
        )

        minio_path = f"{config['storage']['predictions_prefix']}predictions_{timestamp}.csv"
        csv_bytes = predictions_df.to_csv(index=False).encode('utf-8')
        csv_buffer = BytesIO(csv_bytes)

        client.put_object(
            config['storage']['bucket_name'],
            minio_path,
            csv_buffer,
            length=len(csv_bytes),
            content_type='text/csv'
        )

        logger.info(f"Predictions uploaded to MinIO: {minio_path}")
    except Exception as e:
        logger.warning(f"MinIO upload failed: {e}")

    logger.info(f"Predictions saved to {predictions_path}")
    context['task_instance'].xcom_push(key='predictions_path', value=predictions_path)

    return predictions_path


# Define the DAG
with DAG(
    'electricity_load_forecasting',
    default_args=default_args,
    description='End-to-end electricity load forecasting pipeline',
    schedule='0 */6 * * *',  # Updated from schedule_interval to schedule (Airflow 2.8+)
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['mlops', 'forecasting', 'electricity'],
) as dag:

    # Task 1: Extract data from EIA API
    extract_task = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data,
    )

    # Task 2: Validate data with Pandera
    validate_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
    )

    # Task 3: Train model
    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    # Task 4: Batch inference
    inference_task = PythonOperator(
        task_id='batch_inference',
        python_callable=batch_inference,
    )

    # Define task dependencies
    extract_task >> validate_task >> train_task >> inference_task


if __name__ == "__main__":
    print("=" * 60)
    print("✓ DAG file loaded successfully!")
    print("=" * 60)
    print(f"Config path: {config_path}")
    print(f"DAG ID: electricity_load_forecasting")
    print(f"Schedule: Every 6 hours")
    print(f"\nTasks:")
    print("  1. extract_data")
    print("  2. validate_data")
    print("  3. train_model")
    print("  4. batch_inference")
    print("=" * 60)
    print("\n⚠️  Note: This DAG is designed to run in Airflow.")
    print("To use it, copy to your Airflow dags/ folder.")
    print("=" * 60)