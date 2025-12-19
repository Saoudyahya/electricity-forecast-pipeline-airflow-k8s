"""
Fixed Airflow DAG for MLOps Pipeline
Proper workflow:
1. Airflow: Extract data from EIA API → save to MinIO raw/
2. Airflow: Validate data with Pandera → save to MinIO processed/
3. Airflow: Compile Kubeflow Pipeline YAML
4. Airflow: Trigger Kubeflow Pipeline with validated data path
5. Kubeflow: Train model, log to MLflow, save artifacts to MinIO
6. Airflow: (Optional) Trigger Katib HPO
7. Daily: Batch inference → save predictions to MinIO predictions/
8. Daily: Drift detection with EvidentlyAI
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


# ==================== JSON ENCODER FOR NUMPY TYPES ====================
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy and Pandas types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return super().default(obj)


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


# ==================== DATA EXTRACTION ====================
def extract_data(**context):
    """Extract electricity data from EIA API and save to MinIO raw/"""
    from minio import Minio

    logger.info("Extracting data from EIA API")

    # Import data extractor
    from data_extraction import EIADataExtractor

    # Initialize extractor
    extractor = EIADataExtractor(config_path=config_path)

    # Fetch data for the specified number of days
    days = 90
    df = extractor.fetch_recent_data(days=days, regions=None)

    if df.empty:
        raise ValueError("No data retrieved from EIA API")

    logger.info(f"✓ Extracted {len(df)} records")
    logger.info(f"  Date range: {df['period'].min()} to {df['period'].max()}")

    if 'respondent' in df.columns:
        logger.info(f"  Regions: {df['respondent'].nunique()}")

    # Save to MinIO raw/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    object_name = f"{config['storage']['raw_data_prefix']}electricity_data_{timestamp}.csv"

    extractor.save_to_minio(df, object_name)

    # Push data info to XCom
    context['task_instance'].xcom_push(key='raw_data_path', value=object_name)
    context['task_instance'].xcom_push(key='num_records', value=int(len(df)))
    context['task_instance'].xcom_push(key='timestamp', value=timestamp)

    logger.info(f"✓ Data saved to MinIO: s3://{config['storage']['bucket_name']}/{object_name}")

    return object_name


# ==================== DATA VALIDATION ====================
def validate_data(**context):
    """Validate data from MinIO and save to processed/"""
    from minio import Minio
    from data_validation import ElectricityDataValidator

    logger.info("Validating data from MinIO")

    # Get raw data path from previous task
    raw_data_path = context['task_instance'].xcom_pull(
        task_ids='extract_data',
        key='raw_data_path'
    )
    timestamp = context['task_instance'].xcom_pull(
        task_ids='extract_data',
        key='timestamp'
    )

    logger.info(f"Loading data from: {raw_data_path}")

    # Initialize MinIO client
    client = Minio(
        config['storage']['minio_endpoint'],
        access_key=config['storage']['minio_access_key'],
        secret_key=config['storage']['minio_secret_key'],
        secure=False
    )

    # Download data from MinIO
    response = client.get_object(
        config['storage']['bucket_name'],
        raw_data_path
    )
    df = pd.read_csv(BytesIO(response.read()))
    df['period'] = pd.to_datetime(df['period'])

    logger.info(f"✓ Loaded {len(df)} records from MinIO")

    # Initialize validator
    validator = ElectricityDataValidator(config_path=config_path)

    # Validate data
    validated_df, report = validator.validate(df)

    logger.info(f"✓ Validation complete")
    logger.info(f"  Valid: {report['is_valid']}")
    logger.info(f"  Errors: {len(report['errors'])}")
    logger.info(f"  Warnings: {len(report['warnings'])}")

    # Log errors and warnings
    if report['errors']:
        for error in report['errors']:
            logger.error(f"  ❌ {error}")

    if report['warnings']:
        for warning in report['warnings']:
            logger.warning(f"  ⚠️ {warning}")

    # Save validation report to MinIO (with proper JSON encoding)
    report_path = f"validation_reports/validation_report_{timestamp}.json"
    report_bytes = json.dumps(report, indent=2, cls=NumpyEncoder).encode('utf-8')
    report_buffer = BytesIO(report_bytes)

    client.put_object(
        config['storage']['bucket_name'],
        report_path,
        report_buffer,
        length=len(report_bytes),
        content_type='application/json'
    )

    logger.info(f"✓ Validation report saved: {report_path}")

    # Save validated data to MinIO processed/
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

    logger.info(f"✓ Validated data saved: s3://{config['storage']['bucket_name']}/{validated_path}")

    # Fail task if validation failed
    if not report['is_valid']:
        raise ValueError(f"Data validation failed with {len(report['errors'])} errors")

    # Push validated data path to XCom
    context['task_instance'].xcom_push(key='validated_data_path', value=validated_path)
    context['task_instance'].xcom_push(key='validation_report_path', value=report_path)
    context['task_instance'].xcom_push(key='num_validated_records', value=int(len(validated_df)))

    return validated_path


# ==================== COMPILE KUBEFLOW PIPELINE ====================
def compile_kubeflow_pipeline(**context):
    """Compile Kubeflow pipeline from Python to YAML"""
    import subprocess
    import shutil

    logger.info("=" * 80)
    logger.info("Compiling Kubeflow Pipeline")
    logger.info("=" * 80)

    # Paths
    pipeline_script = '/opt/airflow/dags/repo/core/kubeflow_pipeline.py'
    output_yaml = '/opt/airflow/dags/electricity_forecasting_pipeline.yaml'
    temp_yaml = '/opt/airflow/dags/repo/core/electricity_forecasting_pipeline.yaml'

    # Check if pipeline script exists
    if not os.path.exists(pipeline_script):
        logger.error(f"Pipeline script not found: {pipeline_script}")
        raise FileNotFoundError(f"Pipeline script not found: {pipeline_script}")

    logger.info(f"✓ Found pipeline script: {pipeline_script}")

    try:
        # Run the pipeline script to compile it
        logger.info("Running pipeline compilation...")
        result = subprocess.run(
            [sys.executable, pipeline_script],
            capture_output=True,
            text=True,
            timeout=120,
            cwd='/opt/airflow/dags/repo/core'
        )

        if result.returncode != 0:
            logger.error("=" * 80)
            logger.error("Pipeline compilation failed!")
            logger.error("=" * 80)
            logger.error(f"Return code: {result.returncode}")
            logger.error(f"STDOUT:\n{result.stdout}")
            logger.error(f"STDERR:\n{result.stderr}")
            raise Exception(f"Pipeline compilation failed with return code {result.returncode}")

        # Log compilation output
        logger.info("Compilation output:")
        for line in result.stdout.split('\n'):
            if line.strip():
                logger.info(f"  {line}")

        # Check if YAML was created in the script directory
        if os.path.exists(temp_yaml):
            # Move/copy to DAGs folder
            shutil.copy(temp_yaml, output_yaml)
            logger.info(f"✓ Pipeline YAML copied from: {temp_yaml}")
            logger.info(f"✓ Pipeline YAML available at: {output_yaml}")
        elif os.path.exists(output_yaml):
            logger.info(f"✓ Pipeline YAML already at: {output_yaml}")
        else:
            # List files to debug
            logger.error("Pipeline YAML not found. Checking directories...")
            logger.error(f"Files in {os.path.dirname(temp_yaml)}:")
            for f in os.listdir(os.path.dirname(temp_yaml)):
                logger.error(f"  - {f}")
            raise FileNotFoundError("Pipeline YAML was not created by compilation")

        # Verify the YAML file
        file_size = os.path.getsize(output_yaml)
        logger.info(f"✓ Pipeline YAML size: {file_size:,} bytes")

        # Read and validate it's proper YAML
        with open(output_yaml, 'r') as f:
            pipeline_yaml = yaml.safe_load(f)

        logger.info(f"✓ Pipeline YAML is valid")
        logger.info(f"  Pipeline name: {pipeline_yaml.get('pipelineInfo', {}).get('name', 'unknown')}")

        logger.info("=" * 80)
        logger.info("✅ Pipeline compilation successful!")
        logger.info("=" * 80)

        # Push to XCom
        context['task_instance'].xcom_push(key='pipeline_yaml_path', value=output_yaml)

        return output_yaml

    except subprocess.TimeoutExpired:
        logger.error("=" * 80)
        logger.error("❌ Pipeline compilation timed out (>120s)")
        logger.error("=" * 80)
        raise
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ Error during pipeline compilation")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise


# ==================== KUBEFLOW PIPELINE ====================
def trigger_kubeflow_pipeline(**context):
    """Trigger Kubeflow Pipeline for model training with validated data from MinIO"""
    import kfp

    logger.info("Triggering Kubeflow Pipeline for model training")

    # Get validated data path from previous task
    validated_data_path = context['task_instance'].xcom_pull(
        task_ids='validate_data',
        key='validated_data_path'
    )
    num_records = context['task_instance'].xcom_pull(
        task_ids='validate_data',
        key='num_validated_records'
    )

    # Get pipeline YAML path from compilation task
    pipeline_yaml_path = context['task_instance'].xcom_pull(
        task_ids='compile_pipeline',
        key='pipeline_yaml_path'
    )

    logger.info(f"Using validated data: {validated_data_path}")
    logger.info(f"Training on {num_records} validated records")
    logger.info(f"Using pipeline: {pipeline_yaml_path}")

    # Kubeflow namespace (default user namespace)
    kf_namespace = 'kubeflow-user-example-com'

    # Connect to Kubeflow Pipelines with authentication
    try:
        # Method 1: Try using service account token
        token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'

        if os.path.exists(token_path):
            logger.info("Using Kubernetes service account token for authentication")
            with open(token_path, 'r') as f:
                token = f.read().strip()

            # Create KFP client with token
            kfp_client = kfp.Client(
                host=config['kubeflow']['pipeline_host'],
                existing_token=token,
                namespace=kf_namespace
            )
            logger.info("✓ Connected to Kubeflow with service account token")

        else:
            # Method 2: Try without authentication (if auth is disabled)
            logger.warning("No service account token found, trying without authentication")
            kfp_client = kfp.Client(
                host=config['kubeflow']['pipeline_host'],
                namespace=kf_namespace
            )
            logger.info("✓ Connected to Kubeflow without authentication")

    except Exception as e:
        logger.error(f"Failed to connect to Kubeflow: {e}")
        logger.error("Possible solutions:")
        logger.error("1. Disable authentication in Kubeflow (for testing)")
        logger.error("2. Create proper ServiceAccount with RBAC permissions")
        logger.error("3. Use port-forward and localhost connection")
        raise

    # Pipeline parameters - pass the MinIO path of validated data
    pipeline_params = {
        'input_object_name': validated_data_path,
        'minio_endpoint': config['storage']['minio_endpoint'],
        'minio_access_key': config['storage']['minio_access_key'],
        'minio_secret_key': config['storage']['minio_secret_key'],
        'bucket_name': config['storage']['bucket_name'],
        'mlflow_tracking_uri': config['mlflow']['tracking_uri'],
        'experiment_name': config['mlflow']['experiment_name'],
        'model_type': 'lstm',
        'hidden_size': config['model']['hidden_size'],
        'num_layers': config['model']['num_layers'],
        'dropout': config['model']['dropout'],
        'learning_rate': config['model']['learning_rate'],
        'batch_size': config['model']['batch_size'],
        'epochs': config['model']['epochs'],
        'sequence_length': config['model']['sequence_length'],
        'prediction_horizon': config['model']['prediction_horizon']
    }

    # Create run name with timestamp
    run_name = f"training-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Submit pipeline
    try:
        logger.info(f"Submitting pipeline run: {run_name}")
        logger.info(f"Experiment: {config['mlflow']['experiment_name']}")
        logger.info(f"Namespace: {kf_namespace}")

        # Check if pipeline file exists
        if not os.path.exists(pipeline_yaml_path):
            logger.error(f"Pipeline file not found: {pipeline_yaml_path}")
            raise FileNotFoundError(f"Pipeline YAML not found at {pipeline_yaml_path}")

        logger.info(f"Using pipeline file: {pipeline_yaml_path}")

        # Create or get experiment
        try:
            experiment = kfp_client.get_experiment(experiment_name=config['mlflow']['experiment_name'])
            logger.info(f"✓ Using existing experiment: {experiment.experiment_id}")
        except Exception as exp_error:
            logger.info(f"Creating new experiment: {config['mlflow']['experiment_name']}")
            try:
                experiment = kfp_client.create_experiment(
                    name=config['mlflow']['experiment_name'],
                    namespace=kf_namespace
                )
                logger.info(f"✓ Created experiment: {experiment.experiment_id}")
            except Exception as create_error:
                logger.warning(f"Could not create experiment: {create_error}")
                logger.info("Proceeding without explicit experiment...")
                experiment = None

        # Submit the run
        run = kfp_client.create_run_from_pipeline_package(
            pipeline_file=pipeline_yaml_path,
            arguments=pipeline_params,
            run_name=run_name,
            experiment_name=config['mlflow']['experiment_name'] if experiment else None,
            namespace=kf_namespace
        )

        logger.info("=" * 80)
        logger.info("✅ Kubeflow pipeline submitted successfully!")
        logger.info("=" * 80)
        logger.info(f"  Run ID: {run.run_id}")
        logger.info(f"  Run Name: {run_name}")
        logger.info(f"  Namespace: {kf_namespace}")
        logger.info(f"  Data Source: {validated_data_path}")
        logger.info(f"  Records: {num_records:,}")
        logger.info("=" * 80)

        # Optional: Monitor pipeline status
        logger.info("Note: Pipeline is running asynchronously")
        logger.info("Check Kubeflow Pipelines UI for progress")
        logger.info(f"URL: http://localhost:8080 (via port-forward to istio-ingressgateway)")

        # Push run info to XCom
        context['task_instance'].xcom_push(key='pipeline_run_id', value=run.run_id)
        context['task_instance'].xcom_push(key='pipeline_run_name', value=run_name)
        context['task_instance'].xcom_push(key='kf_namespace', value=kf_namespace)

        return run.run_id

    except Exception as e:
        logger.error("=" * 80)
        logger.error("❌ Failed to submit Kubeflow pipeline")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")

        # Provide helpful debugging info
        if "401" in str(e) or "Unauthorized" in str(e):
            logger.error("\n🔐 AUTHENTICATION ERROR")
            logger.error("=" * 80)
            logger.error("Kubeflow requires authentication. Try one of these:")
            logger.error("")
            logger.error("Option 1 (Quick): Disable auth in Kubeflow")
            logger.error("  kubectl edit deployment ml-pipeline -n kubeflow")
            logger.error("  Add: KUBEFLOW_USERID_HEADER: \"\"")
            logger.error("")
            logger.error("Option 2: Create ServiceAccount with proper RBAC")
            logger.error("  See documentation for creating kubeflow-pipeline-runner role")
            logger.error("")
            logger.error("Option 3: Use port-forward")
            logger.error("  kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888")
            logger.error("  Update config.yaml: pipeline_host: http://localhost:8888")
            logger.error("=" * 80)

        elif "404" in str(e) or "Not Found" in str(e):
            logger.error("\n📁 PIPELINE FILE ERROR")
            logger.error("=" * 80)
            logger.error(f"Pipeline file not found: {pipeline_yaml_path}")
            logger.error("Make sure the compiled pipeline YAML is in the DAGs folder")
            logger.error("=" * 80)

        elif "Connection" in str(e) or "timeout" in str(e).lower():
            logger.error("\n🌐 CONNECTION ERROR")
            logger.error("=" * 80)
            logger.error(f"Cannot connect to: {config['kubeflow']['pipeline_host']}")
            logger.error("Check if Kubeflow Pipelines is running:")
            logger.error("  kubectl get pods -n kubeflow | grep ml-pipeline")
            logger.error("=" * 80)

        raise


# ==================== KATIB HPO ====================
def trigger_katib_hpo(**context):
    """Trigger Katib HPO experiment for hyperparameter optimization"""
    import subprocess
    import time

    logger.info("Triggering Katib HPO experiment")

    # Get validated data path to pass to Katib
    validated_data_path = context['task_instance'].xcom_pull(
        task_ids='validate_data',
        key='validated_data_path'
    )

    logger.info(f"HPO will use data: {validated_data_path}")

    # Apply Katib experiment
    try:
        result = subprocess.run(
            ['kubectl', 'apply', '-f', '/opt/airflow/dags/katib-experiment.yaml'],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            logger.error(f"Failed to create Katib experiment: {result.stderr}")
            raise Exception(f"Katib experiment creation failed: {result.stderr}")

        logger.info("✓ Katib experiment created successfully")

        # Wait a bit for experiment to initialize
        time.sleep(10)

        # Get experiment status
        status_result = subprocess.run(
            ['kubectl', 'get', 'experiment', 'electricity-forecast-hpo',
             '-n', config['kubeflow']['namespace'], '-o', 'json'],
            capture_output=True,
            text=True,
            timeout=30
        )

        if status_result.returncode == 0:
            experiment_status = json.loads(status_result.stdout)
            status = experiment_status.get('status', {})
            logger.info(f"✓ Experiment status: {status.get('conditions', [])}")

        # Push to XCom
        context['task_instance'].xcom_push(key='katib_experiment', value='electricity-forecast-hpo')

        return "Katib HPO experiment triggered successfully"

    except subprocess.TimeoutExpired:
        logger.error("Kubectl command timed out")
        raise
    except Exception as e:
        logger.error(f"Error triggering Katib: {e}")
        raise


# ==================== BATCH INFERENCE ====================
def get_latest_model_from_mlflow(**context):
    """Get the latest production model from MLflow"""
    import mlflow
    from mlflow.tracking import MlflowClient

    logger.info("Fetching latest model from MLflow")

    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    client = MlflowClient()

    model_name = "electricity-load-forecaster"

    try:
        # Search for model versions
        versions = client.search_model_versions(f"name='{model_name}'")

        if not versions:
            logger.warning("⚠️ No model versions found in MLflow")
            return None

        # Prefer production models, otherwise use latest
        production_versions = [v for v in versions if v.current_stage == 'Production']

        if production_versions:
            latest_version = max(production_versions, key=lambda x: int(x.version))
            logger.info(f"✓ Using Production model version {latest_version.version}")
        else:
            latest_version = max(versions, key=lambda x: int(x.version))
            logger.info(f"✓ Using latest model version {latest_version.version} (Stage: {latest_version.current_stage})")

        # Create model URI
        model_uri = f"models:/{model_name}/{latest_version.version}"

        logger.info(f"✓ Model URI: {model_uri}")

        # Push to XCom
        context['task_instance'].xcom_push(key='model_version', value=str(latest_version.version))
        context['task_instance'].xcom_push(key='model_uri', value=model_uri)
        context['task_instance'].xcom_push(key='model_stage', value=latest_version.current_stage)

        return model_uri

    except Exception as e:
        logger.error(f"❌ Error fetching model from MLflow: {e}")
        raise


def batch_inference(**context):
    """Run batch inference and save predictions to MinIO predictions/"""
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
    model_version = context['task_instance'].xcom_pull(
        task_ids='get_latest_model',
        key='model_version'
    )

    if model_uri is None:
        logger.warning("⚠️ No model available, skipping inference")
        return "No model available"

    logger.info(f"Using model: {model_uri} (v{model_version})")

    # Load model from MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])

    try:
        model = mlflow.pytorch.load_model(model_uri)
        logger.info("✓ Model loaded from MLflow")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

    # Initialize MinIO client
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
        logger.warning("⚠️ No processed data found in MinIO")
        return "No data available"

    # Get most recent data
    latest_object = max(objects_list, key=lambda x: x.last_modified)
    logger.info(f"Using data: {latest_object.object_name}")

    response = client.get_object(
        config['storage']['bucket_name'],
        latest_object.object_name
    )
    df = pd.read_csv(BytesIO(response.read()))
    df['period'] = pd.to_datetime(df['period'])

    # Select region (use region with most data)
    if 'respondent' in df.columns:
        region_counts = df['respondent'].value_counts()
        selected_region = region_counts.index[0]
        df = df[df['respondent'] == selected_region].copy()
        logger.info(f"Selected region: {selected_region}")

    df = df.sort_values('period').reset_index(drop=True)

    # Prepare input (last sequence_length hours)
    sequence_length = config['model']['sequence_length']

    if len(df) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} records, got {len(df)}")

    input_data = df['value'].values[-sequence_length:].reshape(-1, 1)

    # Scale data
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
        'predicted_load_MW': predictions_scaled,
        'model_version': model_version,
        'region': selected_region if 'respondent' in df.columns else 'unknown'
    })

    # Save predictions to MinIO predictions/
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

    logger.info(f"✓ Predictions saved to MinIO: s3://{config['storage']['bucket_name']}/{predictions_path}")
    logger.info(f"  Predicted {len(predictions_df)} hours ahead")
    logger.info(f"  Mean predicted load: {predictions_scaled.mean():.2f} MW")

    # Push to XCom (convert to Python types)
    context['task_instance'].xcom_push(key='predictions_path', value=predictions_path)
    context['task_instance'].xcom_push(key='num_predictions', value=int(len(predictions_df)))
    context['task_instance'].xcom_push(key='mean_prediction', value=float(predictions_scaled.mean()))

    return predictions_path


# ==================== DRIFT DETECTION ====================
def detect_drift(**context):
    """Detect data and model drift using EvidentlyAI"""
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesMetric
    from minio import Minio

    logger.info("Starting drift detection")

    client = Minio(
        config['storage']['minio_endpoint'],
        access_key=config['storage']['minio_access_key'],
        secret_key=config['storage']['minio_secret_key'],
        secure=False
    )

    # Get processed data
    objects = list(client.list_objects(
        config['storage']['bucket_name'],
        prefix=config['storage']['processed_data_prefix']
    ))

    if len(objects) < 2:
        logger.warning("⚠️ Not enough data for drift detection (need at least 2 datasets)")
        return "Insufficient data"

    # Sort by date
    objects_sorted = sorted(objects, key=lambda x: x.last_modified, reverse=True)

    # Get current data (most recent)
    current_object = objects_sorted[0]
    response = client.get_object(config['storage']['bucket_name'], current_object.object_name)
    current_df = pd.read_csv(BytesIO(response.read()))
    current_df['period'] = pd.to_datetime(current_df['period'])
    logger.info(f"Current data: {current_object.object_name} ({len(current_df)} records)")

    # Get reference data (oldest available)
    reference_object = objects_sorted[-1]
    response = client.get_object(config['storage']['bucket_name'], reference_object.object_name)
    reference_df = pd.read_csv(BytesIO(response.read()))
    reference_df['period'] = pd.to_datetime(reference_df['period'])
    logger.info(f"Reference data: {reference_object.object_name} ({len(reference_df)} records)")

    # Filter to same region
    if 'respondent' in current_df.columns:
        region = current_df['respondent'].value_counts().index[0]
        current_df = current_df[current_df['respondent'] == region]
        reference_df = reference_df[reference_df['respondent'] == region]
        logger.info(f"Analyzing region: {region}")

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

    # Save HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"/tmp/drift_report_{timestamp}.html"
    drift_report.save_html(report_path)

    # Get drift metrics
    report_dict = drift_report.as_dict()

    # Parse drift results
    drift_detected = False
    drift_share = 0.0

    try:
        dataset_drift = report_dict['metrics'][1]['result']
        drift_detected = dataset_drift.get('dataset_drift', False)
        drift_share = dataset_drift.get('drift_share', 0.0)
    except (KeyError, IndexError) as e:
        logger.warning(f"Could not parse drift metrics: {e}")

    # Save HTML report to MinIO
    with open(report_path, 'rb') as f:
        file_size = os.path.getsize(report_path)
        client.put_object(
            config['storage']['bucket_name'],
            f"drift_reports/drift_report_{timestamp}.html",
            f,
            length=file_size,
            content_type='text/html'
        )
    logger.info(f"✓ Drift report saved to MinIO")

    # Save drift metrics JSON (with proper encoding)
    drift_metrics = {
        'timestamp': timestamp,
        'drift_detected': bool(drift_detected),
        'drift_share': float(drift_share),
        'threshold': float(config['drift_detection']['drift_threshold']),
        'reference_data': {
            'source': reference_object.object_name,
            'size': int(len(reference_df)),
            'date_range': f"{reference_df['period'].min()} to {reference_df['period'].max()}"
        },
        'current_data': {
            'source': current_object.object_name,
            'size': int(len(current_df)),
            'date_range': f"{current_df['period'].min()} to {current_df['period'].max()}"
        }
    }

    metrics_bytes = json.dumps(drift_metrics, indent=2, cls=NumpyEncoder).encode('utf-8')
    metrics_buffer = BytesIO(metrics_bytes)

    client.put_object(
        config['storage']['bucket_name'],
        f"drift_reports/drift_metrics_{timestamp}.json",
        metrics_buffer,
        length=len(metrics_bytes),
        content_type='application/json'
    )

    logger.info(f"✓ Drift detection complete")
    logger.info(f"  Drift detected: {drift_detected}")
    logger.info(f"  Drift share: {drift_share:.2%}")
    logger.info(f"  Threshold: {config['drift_detection']['drift_threshold']:.2%}")

    # Push to XCom
    context['task_instance'].xcom_push(key='drift_detected', value=bool(drift_detected))
    context['task_instance'].xcom_push(key='drift_share', value=float(drift_share))

    # Alert if drift detected
    if drift_detected or drift_share > config['drift_detection']['drift_threshold']:
        logger.warning("=" * 60)
        logger.warning("⚠️ DRIFT ALERT!")
        logger.warning(f"   Data drift detected with {drift_share:.2%} drift share")
        logger.warning(f"   Threshold: {config['drift_detection']['drift_threshold']:.2%}")
        logger.warning("   Consider retraining the model")
        logger.warning("=" * 60)

    return drift_metrics


# ===================================================================
# DAG DEFINITIONS
# ===================================================================

# DAG 1: Weekly Training Pipeline
with DAG(
    'electricity_training_pipeline',
    default_args=default_args,
    description='Weekly ML pipeline: Extract → Validate → Compile → Train with Kubeflow → Log to MLflow',
    schedule='0 0 * * 0',  # Every Sunday at midnight
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['mlops', 'training', 'kubeflow', 'mlflow', 'minio'],
) as training_dag:

    # Task 1: Extract data from EIA API → save to MinIO raw/
    extract_task = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data,
    )

    # Task 2: Validate data → save to MinIO processed/
    validate_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
    )

    # Task 3: Compile Kubeflow Pipeline to YAML
    compile_pipeline_task = PythonOperator(
        task_id='compile_pipeline',
        python_callable=compile_kubeflow_pipeline,
    )

    # Task 4: Trigger Kubeflow Pipeline (trains model, logs to MLflow)
    trigger_kfp = PythonOperator(
        task_id='trigger_kubeflow_pipeline',
        python_callable=trigger_kubeflow_pipeline,
    )

    # Define task dependencies
    extract_task >> validate_task >> compile_pipeline_task >> trigger_kfp


# DAG 2: Daily Inference and Monitoring
with DAG(
    'electricity_daily_inference',
    default_args=default_args,
    description='Daily: Batch inference → Save predictions to MinIO → Drift detection',
    schedule='0 2 * * *',  # Every day at 2 AM
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['mlops', 'inference', 'drift-detection', 'minio', 'mlflow'],
) as inference_dag:

    # Task 1: Get latest model from MLflow
    get_model = PythonOperator(
        task_id='get_latest_model',
        python_callable=get_latest_model_from_mlflow,
    )

    # Task 2: Run batch inference → save to MinIO predictions/
    inference = PythonOperator(
        task_id='batch_inference',
        python_callable=batch_inference,
    )

    # Task 3: Detect drift with EvidentlyAI
    drift_detection = PythonOperator(
        task_id='detect_drift',
        python_callable=detect_drift,
    )

    # Define task dependencies
    get_model >> inference >> drift_detection


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("✓ MLOps Pipeline DAGs Loaded Successfully!")
    print("=" * 80)
    print("\n📊 DAG 1: electricity_training_pipeline (Weekly)")
    print("-" * 80)
    print("Schedule: Every Sunday at midnight")
    print("Tasks:")
    print("  1. extract_data           - Fetch from EIA API → MinIO raw/")
    print("  2. validate_data          - Validate with Pandera → MinIO processed/")
    print("  3. compile_pipeline       - Compile Kubeflow Pipeline Python → YAML")
    print("  4. trigger_kubeflow       - Train model → Log to MLflow")
    print("\n🔮 DAG 2: electricity_daily_inference (Daily)")
    print("-" * 80)
    print("Schedule: Every day at 2 AM")
    print("Tasks:")
    print("  1. get_latest_model       - Fetch from MLflow Model Registry")
    print("  2. batch_inference        - Predict → MinIO predictions/")
    print("  3. detect_drift           - Monitor with EvidentlyAI")
    print("\n💾 Data Storage (MinIO):")
    print("  • raw/                    - Raw data from EIA API")
    print("  • processed/              - Validated data ready for training")
    print("  • predictions/            - Model predictions")
    print("  • drift_reports/          - Drift detection reports")
    print("  • validation_reports/     - Data validation reports")
    print("\n📈 Model Registry (MLflow):")
    print("  • Experiment tracking     - All training runs logged")
    print("  • Model versioning        - Automatic version management")
    print("  • Model artifacts         - Stored in MinIO via MLflow")
    print("=" * 80)