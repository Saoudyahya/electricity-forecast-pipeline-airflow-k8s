"""
Airflow DAG for Electricity Load Forecasting Pipeline

DAG Structure:
1. Extract data from EIA API
2. Validate data with Pandera
3. Trigger Kubeflow training pipeline
4. Daily batch inference
5. Drift detection with EvidentlyAI
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from datetime import datetime, timedelta
import yaml
import logging

# Import custom modules
import sys
sys.path.append('/opt/airflow/dags')

from data_extraction import EIADataExtractor
from data_validation import ElectricityDataValidator

logger = logging.getLogger(__name__)

# Load config
with open('/opt/airflow/config/config.yaml', 'r') as f:
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
    
    extractor = EIADataExtractor(config_path='/opt/airflow/config/config.yaml')
    
    # Fetch last 90 days of data
    df = extractor.fetch_recent_data(
        days=90,
        regions=['CAL', 'MIDA', 'TEX']  # California, Mid-Atlantic, Texas
    )
    
    # Save to MinIO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    object_name = f"{config['storage']['raw_data_prefix']}electricity_data_{timestamp}.csv"
    
    extractor.save_to_minio(df, object_name)
    
    # Push to XCom for next task
    context['task_instance'].xcom_push(key='data_path', value=object_name)
    context['task_instance'].xcom_push(key='num_records', value=len(df))
    
    logger.info(f"Extracted {len(df)} records and saved to {object_name}")
    
    return object_name


def validate_data(**context):
    """Validate data using Pandera"""
    logger.info("Starting data validation")
    
    from minio import Minio
    import pandas as pd
    from io import BytesIO
    
    # Get data path from previous task
    data_path = context['task_instance'].xcom_pull(
        task_ids='extract_data',
        key='data_path'
    )
    
    # Load data from MinIO
    client = Minio(
        config['storage']['minio_endpoint'],
        access_key=config['storage']['minio_access_key'],
        secret_key=config['storage']['minio_secret_key'],
        secure=False
    )
    
    response = client.get_object(config['storage']['bucket_name'], data_path)
    df = pd.read_csv(BytesIO(response.read()))
    df['period'] = pd.to_datetime(df['period'])
    
    # Validate
    validator = ElectricityDataValidator(config_path='/opt/airflow/config/config.yaml')
    validated_df, report = validator.validate(df)
    
    if not report['is_valid']:
        raise ValueError(f"Data validation failed: {report['errors']}")
    
    # Save validated data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    
    # Save validation report
    import json
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
    
    # Push to XCom
    context['task_instance'].xcom_push(key='validated_data_path', value=validated_path)
    context['task_instance'].xcom_push(key='validation_report', value=report)
    
    logger.info(f"Data validation passed. Saved to {validated_path}")
    
    return validated_path


def trigger_kubeflow_training(**context):
    """Trigger Kubeflow training pipeline"""
    logger.info("Triggering Kubeflow training pipeline")
    
    import kfp
    from kfp import dsl
    
    # Get validated data path
    data_path = context['task_instance'].xcom_pull(
        task_ids='validate_data',
        key='validated_data_path'
    )
    
    # Create Kubeflow client
    client = kfp.Client(host=config['kubeflow']['pipeline_host'])
    
    # Submit pipeline run
    run_name = f"electricity-forecast-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    run = client.create_run_from_pipeline_func(
        pipeline_func=training_pipeline,
        arguments={
            'data_path': data_path,
            'model_type': 'lstm',
            'sequence_length': config['model']['sequence_length'],
            'prediction_horizon': config['model']['prediction_horizon'],
            'hidden_size': config['model']['hidden_size'],
            'num_layers': config['model']['num_layers'],
            'epochs': config['model']['epochs']
        },
        run_name=run_name,
        namespace=config['kubeflow']['namespace']
    )
    
    logger.info(f"Started Kubeflow run: {run_name}")
    context['task_instance'].xcom_push(key='kfp_run_id', value=run.run_id)
    
    return run.run_id


def batch_inference(**context):
    """Run batch inference on latest data"""
    logger.info("Starting batch inference")
    
    from minio import Minio
    import pandas as pd
    from io import BytesIO
    import mlflow
    from model import ElectricityLoadForecaster
    
    # Setup MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    # Load latest model from MLflow
    model_name = "electricity-load-forecaster"
    model_version = "latest"
    
    try:
        model_uri = f"models:/{model_name}/{model_version}"
        # In production, load model from MLflow
        # For now, we'll use local model
        forecaster = ElectricityLoadForecaster()
        forecaster.load_model("best_model.pt")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Load recent data
    client = Minio(
        config['storage']['minio_endpoint'],
        access_key=config['storage']['minio_access_key'],
        secret_key=config['storage']['minio_secret_key'],
        secure=False
    )
    
    # Get latest validated data
    data_path = context['task_instance'].xcom_pull(
        task_ids='validate_data',
        key='validated_data_path'
    )
    
    response = client.get_object(config['storage']['bucket_name'], data_path)
    df = pd.read_csv(BytesIO(response.read()))
    df['period'] = pd.to_datetime(df['period'])
    
    # Prepare input sequence (last sequence_length hours)
    sequence_length = config['model']['sequence_length']
    input_sequence = df['value'].values[-sequence_length:].reshape(-1, 1)
    
    # Normalize
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
    
    logger.info(f"Predictions saved to {predictions_path}")
    context['task_instance'].xcom_push(key='predictions_path', value=predictions_path)
    
    return predictions_path


def detect_drift(**context):
    """Detect data drift using EvidentlyAI"""
    logger.info("Starting drift detection")
    
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from minio import Minio
    import pandas as pd
    from io import BytesIO
    
    # Load historical (reference) data and current data
    client = Minio(
        config['storage']['minio_endpoint'],
        access_key=config['storage']['minio_access_key'],
        secret_key=config['storage']['minio_secret_key'],
        secure=False
    )
    
    # Get data paths
    data_path = context['task_instance'].xcom_pull(
        task_ids='validate_data',
        key='validated_data_path'
    )
    
    response = client.get_object(config['storage']['bucket_name'], data_path)
    df = pd.read_csv(BytesIO(response.read()))
    df['period'] = pd.to_datetime(df['period'])
    
    # Split into reference and current windows
    reference_days = config['drift_detection']['reference_window_days']
    current_days = config['drift_detection']['current_window_days']
    
    reference_data = df.iloc[:-current_days*24]  # Earlier data
    current_data = df.iloc[-current_days*24:]    # Recent data
    
    # Create Evidently report
    column_mapping = ColumnMapping(
        numerical_features=['value'],
        datetime_features=['period']
    )
    
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset()
    ])
    
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"/tmp/drift_report_{timestamp}.html"
    report.save_html(report_path)
    
    # Upload to MinIO
    with open(report_path, 'rb') as f:
        report_bytes = f.read()
    
    report_buffer = BytesIO(report_bytes)
    minio_report_path = f"drift_reports/drift_report_{timestamp}.html"
    
    client.put_object(
        config['storage']['bucket_name'],
        minio_report_path,
        report_buffer,
        length=len(report_bytes),
        content_type='text/html'
    )
    
    logger.info(f"Drift report saved to {minio_report_path}")
    
    # Check for significant drift
    report_dict = report.as_dict()
    drift_detected = report_dict['metrics'][0]['result']['dataset_drift']
    
    if drift_detected:
        logger.warning("⚠️ Data drift detected!")
        # In production, trigger model retraining or send alerts
    else:
        logger.info("✓ No significant drift detected")
    
    context['task_instance'].xcom_push(key='drift_detected', value=drift_detected)
    context['task_instance'].xcom_push(key='drift_report_path', value=minio_report_path)
    
    return drift_detected


# Define the DAG
with DAG(
    'electricity_load_forecasting',
    default_args=default_args,
    description='End-to-end electricity load forecasting pipeline',
    schedule_interval='0 */6 * * *',  # Run every 6 hours
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['mlops', 'forecasting', 'electricity'],
) as dag:
    
    # Task 1: Extract data from EIA API
    extract_task = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data,
        provide_context=True
    )
    
    # Task 2: Validate data with Pandera
    validate_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
        provide_context=True
    )
    
    # Task 3: Trigger Kubeflow training pipeline
    trigger_training_task = PythonOperator(
        task_id='trigger_kubeflow_training',
        python_callable=trigger_kubeflow_training,
        provide_context=True
    )
    
    # Task 4: Batch inference (runs daily)
    inference_task = PythonOperator(
        task_id='batch_inference',
        python_callable=batch_inference,
        provide_context=True
    )
    
    # Task 5: Detect drift
    drift_task = PythonOperator(
        task_id='detect_drift',
        python_callable=detect_drift,
        provide_context=True
    )
    
    # Define task dependencies
    extract_task >> validate_task >> trigger_training_task >> inference_task >> drift_task


# Helper function for Kubeflow pipeline (referenced in trigger_kubeflow_training)
@dsl.pipeline(
    name='Electricity Load Forecasting Training',
    description='Train LSTM model for electricity load forecasting'
)
def training_pipeline(
    data_path: str,
    model_type: str = 'lstm',
    sequence_length: int = 168,
    prediction_horizon: int = 24,
    hidden_size: int = 128,
    num_layers: int = 2,
    epochs: int = 50
):
    """Kubeflow training pipeline definition"""
    # This will be implemented in the kubeflow_pipeline.py file
    pass
