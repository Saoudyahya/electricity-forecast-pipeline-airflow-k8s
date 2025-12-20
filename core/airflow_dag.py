"""
Fixed Airflow DAG for MLOps Pipeline
Simplified workflow:
1. Airflow: Extract data from EIA API → save to MinIO raw/
2. Airflow: Validate data with Pandera → save to MinIO processed/
3. Airflow: Compile Kubeflow Pipeline YAML → save to MinIO pipelines/
4. Manual: User triggers Kubeflow Pipeline via UI using compiled YAML from MinIO
5. Kubeflow: Train model, log to MLflow, save artifacts to MinIO
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
def compile_and_upload_pipeline(**context):
    """Compile Kubeflow pipeline from Python to YAML and upload to MinIO"""
    import subprocess
    from minio import Minio

    logger.info("=" * 80)
    logger.info("Compiling Kubeflow Pipeline")
    logger.info("=" * 80)

    # Paths
    pipeline_script = '/opt/airflow/dags/repo/core/kubeflow_pipeline.py'
    output_yaml = '/tmp/electricity_forecasting_pipeline.yaml'

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
            cwd='/tmp'
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

        # Check if YAML was created
        if not os.path.exists(output_yaml):
            logger.error("Pipeline YAML not found. Checking /tmp/:")
            for f in os.listdir('/tmp'):
                if 'pipeline' in f.lower() or 'yaml' in f.lower():
                    logger.error(f"  - {f}")
            raise FileNotFoundError("Pipeline YAML was not created by compilation")

        logger.info(f"✓ Pipeline YAML created at: {output_yaml}")

        # Verify the YAML file
        file_size = os.path.getsize(output_yaml)
        logger.info(f"✓ Pipeline YAML size: {file_size:,} bytes")

        # Read and validate it's proper YAML
        with open(output_yaml, 'r') as f:
            pipeline_yaml = yaml.safe_load(f)

        logger.info(f"✓ Pipeline YAML is valid")
        logger.info(f"  Pipeline name: {pipeline_yaml.get('pipelineInfo', {}).get('name', 'unknown')}")

        # Upload to MinIO
        logger.info("\nUploading pipeline YAML to MinIO...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        minio_object_name = f"{config['storage']['pipeline_prefix']}electricity_forecasting_pipeline_{timestamp}.yaml"

        # Initialize MinIO client
        client = Minio(
            config['storage']['minio_endpoint'],
            access_key=config['storage']['minio_access_key'],
            secret_key=config['storage']['minio_secret_key'],
            secure=False
        )

        # Upload to MinIO
        with open(output_yaml, 'rb') as f:
            client.put_object(
                config['storage']['bucket_name'],
                minio_object_name,
                f,
                length=file_size,
                content_type='application/x-yaml'
            )

        logger.info(f"✓ Pipeline YAML uploaded to MinIO: s3://{config['storage']['bucket_name']}/{minio_object_name}")

        logger.info("=" * 80)
        logger.info("✅ Pipeline compilation and upload successful!")
        logger.info("=" * 80)
        logger.info("\n📋 Next Steps:")
        logger.info("  1. Navigate to Kubeflow Pipelines UI")
        logger.info("  2. Click 'Upload Pipeline'")
        logger.info(f"  3. Download pipeline from MinIO: {minio_object_name}")
        logger.info("  4. Upload the YAML file to Kubeflow")
        logger.info("  5. Create a run with the validated data path")
        logger.info("=" * 80)

        # Push to XCom
        context['task_instance'].xcom_push(key='pipeline_yaml_minio_path', value=minio_object_name)
        context['task_instance'].xcom_push(key='pipeline_yaml_local_path', value=output_yaml)

        # Get validated data path for reference
        validated_data_path = context['task_instance'].xcom_pull(
            task_ids='validate_data',
            key='validated_data_path'
        )

        logger.info("\n📊 Pipeline Input Parameters:")
        logger.info(f"  input_object_name: {validated_data_path}")
        logger.info(f"  bucket_name: {config['storage']['bucket_name']}")
        logger.info(f"  minio_endpoint: {config['storage']['minio_endpoint']}")

        return minio_object_name

    except subprocess.TimeoutExpired:
        logger.error("=" * 80)
        logger.error("❌ Pipeline compilation timed out (>120s)")
        logger.error("=" * 80)
        raise
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ Error during pipeline compilation/upload")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise


# ===================================================================
# DAG DEFINITION
# ===================================================================

# Weekly Training Pipeline Preparation
with DAG(
    'electricity_pipeline_preparation',
    default_args=default_args,
    description='Weekly pipeline: Extract → Validate → Compile Pipeline → Store in MinIO',
    schedule='0 0 * * 0',  # Every Sunday at midnight
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['mlops', 'data-preparation', 'kubeflow', 'minio'],
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

    # Task 3: Compile Kubeflow Pipeline YAML → save to MinIO pipelines/
    compile_and_upload_task = PythonOperator(
        task_id='compile_and_upload_pipeline',
        python_callable=compile_and_upload_pipeline,
    )

    # Define task dependencies
    extract_task >> validate_task >> compile_and_upload_task


# ===================================================================
# MAIN
# ===================================================================
if __name__ == '__main__':
    import os

    # Write to /tmp/ first (writable), then copy to /opt/airflow/dags/
    output_path = '/tmp/electricity_forecasting_pipeline.yaml'

    # Compile pipeline
    compiler.Compiler().compile(
        pipeline_func=electricity_training_pipeline,
        package_path=output_path  # Write to /tmp/
    )

    print("=" * 80)
    print("✅ Kubeflow Pipeline Compiled Successfully!")
    print("=" * 80)
    print(f"📄 Output: {output_path}")

    # Copy to DAGs folder (writable location)
    import shutil

    final_path = '/opt/airflow/dags/electricity_forecasting_pipeline.yaml'
    try:
        shutil.copy(output_path, final_path)
        print(f"✓ Copied to: {final_path}")
    except Exception as e:
        print(f"⚠️ Could not copy to {final_path}: {e}")
        print(f"   Pipeline available at: {output_path}")

    print("\n🔧 Pipeline Architecture:")
    print("  Input:  Pre-validated data from MinIO (provided by Airflow)")
    print("  Task:   Train LSTM/Transformer model")
    print("  Output: Model logged to MLflow + artifacts in MinIO")
    print("\n📊 Integration Flow:")
    print("  1. Airflow extracts data from EIA → MinIO raw/")
    print("  2. Airflow validates data → MinIO processed/")
    print("  3. Airflow compiles & stores pipeline YAML → MinIO pipelines/")
    print("  4. User triggers Kubeflow pipeline manually via UI")
    print("  5. Kubeflow trains model → MLflow + MinIO")
    print("  6. Model ready for use!")
    print("\n💾 Storage:")
    print("  • Training data:  MinIO (processed/)")
    print("  • Pipeline YAML:  MinIO (pipelines/)")
    print("  • Model artifacts: MinIO (via MLflow)")
    print("  • Experiments:     MLflow tracking server")
    print("  • Model registry:  MLflow Model Registry")
    print("=" * 80)