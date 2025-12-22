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


# ==================== ENSURE MINIO BUCKETS ====================
def ensure_minio_buckets(**context):
    """Ensure all required MinIO buckets exist"""
    from minio import Minio

    logger.info("Checking MinIO buckets...")

    # Initialize MinIO client
    client = Minio(
        config['storage']['minio_endpoint'],
        access_key=config['storage']['minio_access_key'],
        secret_key=config['storage']['minio_secret_key'],
        secure=False
    )

    # Required buckets
    required_buckets = [
        config['storage']['bucket_name'],
        'mlflow-artifacts',
        config['storage']['pipeline_bucket']  # Use pipeline_bucket from config
    ]

    created_buckets = []
    existing_buckets = []

    for bucket_name in required_buckets:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            created_buckets.append(bucket_name)
            logger.info(f"✓ Created bucket: {bucket_name}")
        else:
            existing_buckets.append(bucket_name)
            logger.info(f"✓ Bucket exists: {bucket_name}")

    # Push results to XCom
    context['task_instance'].xcom_push(key='created_buckets', value=created_buckets)
    context['task_instance'].xcom_push(key='existing_buckets', value=existing_buckets)

    logger.info(f"✓ All {len(required_buckets)} buckets ready")

    return {
        'created': created_buckets,
        'existing': existing_buckets
    }


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

        # Get the pipeline bucket name
        pipeline_bucket = config['storage']['pipeline_bucket']

        # Ensure pipeline bucket exists
        if not client.bucket_exists(pipeline_bucket):
            logger.info(f"Creating pipeline bucket: {pipeline_bucket}")
            client.make_bucket(pipeline_bucket)

        # Upload to MinIO (pipeline bucket, not data bucket)
        with open(output_yaml, 'rb') as f:
            client.put_object(
                pipeline_bucket,  # Use dedicated pipeline bucket
                minio_object_name,
                f,
                length=file_size,
                content_type='application/x-yaml'
            )

        logger.info(f"✓ Pipeline YAML uploaded to MinIO: s3://{pipeline_bucket}/{minio_object_name}")

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
        context['task_instance'].xcom_push(key='pipeline_bucket', value=pipeline_bucket)

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


# ==================== DATA QUALITY SUMMARY ====================
def generate_data_quality_summary(**context):
    """Generate comprehensive data quality summary report"""
    from minio import Minio

    logger.info("Generating data quality summary")

    # Get validation report path
    validation_report_path = context['task_instance'].xcom_pull(
        task_ids='validate_data',
        key='validation_report_path'
    )

    # Initialize MinIO client
    client = Minio(
        config['storage']['minio_endpoint'],
        access_key=config['storage']['minio_access_key'],
        secret_key=config['storage']['minio_secret_key'],
        secure=False
    )

    # Load validation report
    response = client.get_object(
        config['storage']['bucket_name'],
        validation_report_path
    )
    report = json.loads(response.read())

    # Generate summary
    summary = {
        'pipeline_run': datetime.now().isoformat(),
        'data_quality': {
            'is_valid': report['is_valid'],
            'total_records': report['stats']['total_records'],
            'unique_regions': report['stats']['unique_regions'],
            'error_count': len(report['errors']),
            'warning_count': len(report['warnings']),
        },
        'data_statistics': report['stats']['value_stats'],
        'date_range': report['stats']['date_range'],
        'regions': report['stats'].get('regions', [])
    }

    # Log summary
    logger.info("=" * 60)
    logger.info("DATA QUALITY SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Status: {'✅ VALID' if summary['data_quality']['is_valid'] else '❌ INVALID'}")
    logger.info(f"Records: {summary['data_quality']['total_records']:,}")
    logger.info(f"Regions: {summary['data_quality']['unique_regions']}")
    logger.info(f"Errors: {summary['data_quality']['error_count']}")
    logger.info(f"Warnings: {summary['data_quality']['warning_count']}")
    logger.info(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    logger.info("=" * 60)

    # Push summary to XCom
    context['task_instance'].xcom_push(key='data_quality_summary', value=summary)

    return summary


# ==================== GENERATE PIPELINE PARAMETERS ====================
def generate_pipeline_parameters(**context):
    """Generate ready-to-use parameters for Kubeflow pipeline"""

    logger.info("Generating Kubeflow pipeline parameters")

    # Get validated data path
    validated_data_path = context['task_instance'].xcom_pull(
        task_ids='validate_data',
        key='validated_data_path'
    )

    # Get pipeline YAML path
    pipeline_yaml_path = context['task_instance'].xcom_pull(
        task_ids='compile_and_upload_pipeline',
        key='pipeline_yaml_minio_path'
    )

    # Get pipeline bucket
    pipeline_bucket = context['task_instance'].xcom_pull(
        task_ids='compile_and_upload_pipeline',
        key='pipeline_bucket'
    )

    # Generate parameters
    parameters = {
        "input_object_name": validated_data_path,
        "minio_endpoint": config['storage']['minio_endpoint'],
        "minio_access_key": config['storage']['minio_access_key'],
        "minio_secret_key": config['storage']['minio_secret_key'],
        "bucket_name": config['storage']['bucket_name'],
        "mlflow_tracking_uri": config['mlflow']['tracking_uri'],
        "experiment_name": config['mlflow']['experiment_name'],
        "model_type": "lstm",
        "hidden_size": config['model']['hidden_size'],
        "num_layers": config['model']['num_layers'],
        "dropout": config['model']['dropout'],
        "learning_rate": config['model']['learning_rate'],
        "batch_size": config['model']['batch_size'],
        "epochs": config['model']['epochs'],
        "sequence_length": config['model']['sequence_length'],
        "prediction_horizon": config['model']['prediction_horizon']
    }

    # Save parameters to MinIO
    timestamp = context['task_instance'].xcom_pull(
        task_ids='extract_data',
        key='timestamp'
    )

    from minio import Minio
    client = Minio(
        config['storage']['minio_endpoint'],
        access_key=config['storage']['minio_access_key'],
        secret_key=config['storage']['minio_secret_key'],
        secure=False
    )

    params_path = f"pipeline_parameters/parameters_{timestamp}.json"
    params_bytes = json.dumps(parameters, indent=2).encode('utf-8')
    params_buffer = BytesIO(params_bytes)

    client.put_object(
        config['storage']['bucket_name'],
        params_path,
        params_buffer,
        length=len(params_bytes),
        content_type='application/json'
    )

    logger.info("=" * 60)
    logger.info("KUBEFLOW PIPELINE PARAMETERS")
    logger.info("=" * 60)
    logger.info(f"Pipeline YAML location: s3://{pipeline_bucket}/{pipeline_yaml_path}")
    logger.info(f"Parameters file: {params_path}")
    logger.info("")
    logger.info("Copy these parameters when creating Kubeflow run:")
    logger.info("-" * 60)
    for key, value in parameters.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)

    # Push to XCom
    context['task_instance'].xcom_push(key='pipeline_parameters', value=parameters)
    context['task_instance'].xcom_push(key='pipeline_parameters_path', value=params_path)

    return params_path


# ==================== CLEANUP OLD FILES ====================
def cleanup_old_files(**context):
    """Clean up old files from MinIO (keep last 10 of each type)"""
    from minio import Minio

    logger.info("Cleaning up old files from MinIO")

    client = Minio(
        config['storage']['minio_endpoint'],
        access_key=config['storage']['minio_access_key'],
        secret_key=config['storage']['minio_secret_key'],
        secure=False
    )

    # Format: (bucket_name, prefix)
    prefixes_to_clean = [
        (config['storage']['bucket_name'], 'raw/'),
        (config['storage']['bucket_name'], 'processed/'),
        (config['storage']['bucket_name'], 'validation_reports/'),
        (config['storage']['pipeline_bucket'], 'compiled/'),  # Correct bucket for pipelines
        (config['storage']['bucket_name'], 'pipeline_parameters/')
    ]

    keep_count = 10
    total_deleted = 0

    for bucket_name, prefix in prefixes_to_clean:
        # Check if bucket exists
        if not client.bucket_exists(bucket_name):
            logger.warning(f"  {bucket_name}: Bucket does not exist, skipping")
            continue

        # List all objects with this prefix
        objects = list(client.list_objects(
            bucket_name,
            prefix=prefix
        ))

        if len(objects) <= keep_count:
            logger.info(f"  {bucket_name}/{prefix}: {len(objects)} files (keeping all)")
            continue

        # Sort by last modified (oldest first)
        objects_sorted = sorted(objects, key=lambda x: x.last_modified)

        # Delete oldest files
        to_delete = objects_sorted[:-keep_count]

        for obj in to_delete:
            try:
                client.remove_object(
                    bucket_name,
                    obj.object_name
                )
                total_deleted += 1
            except Exception as e:
                logger.warning(f"  Failed to delete {obj.object_name}: {e}")

        logger.info(f"  {bucket_name}/{prefix}: Deleted {len(to_delete)} old files (kept {keep_count} newest)")

    logger.info(f"✓ Cleanup complete: {total_deleted} files deleted")

    return total_deleted


# ==================== SEND SUCCESS NOTIFICATION ====================
def send_success_notification(**context):
    """Log success notification with pipeline details"""

    # Get all relevant info from XCom
    timestamp = context['task_instance'].xcom_pull(
        task_ids='extract_data',
        key='timestamp'
    )

    num_records = context['task_instance'].xcom_pull(
        task_ids='validate_data',
        key='num_validated_records'
    )

    validated_data_path = context['task_instance'].xcom_pull(
        task_ids='validate_data',
        key='validated_data_path'
    )

    pipeline_yaml_path = context['task_instance'].xcom_pull(
        task_ids='compile_and_upload_pipeline',
        key='pipeline_yaml_minio_path'
    )

    pipeline_bucket = context['task_instance'].xcom_pull(
        task_ids='compile_and_upload_pipeline',
        key='pipeline_bucket'
    )

    pipeline_params_path = context['task_instance'].xcom_pull(
        task_ids='generate_pipeline_parameters',
        key='pipeline_parameters_path'
    )

    data_quality = context['task_instance'].xcom_pull(
        task_ids='data_quality_summary',
        key='data_quality_summary'
    )

    # Log comprehensive success message
    logger.info("")
    logger.info("=" * 80)
    logger.info("🎉 PIPELINE PREPARATION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("📊 DATA SUMMARY:")
    logger.info(f"  • Timestamp: {timestamp}")
    logger.info(f"  • Records validated: {num_records:,}")
    logger.info(f"  • Regions: {data_quality['data_quality']['unique_regions']}")
    logger.info(f"  • Date range: {data_quality['date_range']['start']} to {data_quality['date_range']['end']}")
    logger.info("")
    logger.info("📁 MINIO ARTIFACTS:")
    logger.info(f"  • Validated data: s3://{config['storage']['bucket_name']}/{validated_data_path}")
    logger.info(f"  • Pipeline YAML: s3://{pipeline_bucket}/{pipeline_yaml_path}")
    logger.info(f"  • Pipeline parameters: s3://{config['storage']['bucket_name']}/{pipeline_params_path}")
    logger.info("")
    logger.info("🚀 NEXT STEPS:")
    logger.info("  1. Access MinIO UI: kubectl port-forward -n minio svc/minio 9001:9001")
    logger.info(f"  2. Navigate to '{pipeline_bucket}' bucket and download: {pipeline_yaml_path}")
    logger.info("  3. Access Kubeflow UI: kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80")
    logger.info("  4. Upload pipeline and create run")
    logger.info(f"  5. Use parameters from: {pipeline_params_path}")
    logger.info("")
    logger.info("💡 QUICK START:")
    logger.info("  MinIO: http://localhost:9001 (minioadmin/minioadmin)")
    logger.info("  Kubeflow: http://localhost:8080")
    logger.info("  MLflow: kubectl port-forward -n mlflow svc/mlflow 5000:5000")
    logger.info("")
    logger.info("=" * 80)

    # Create notification summary
    notification = {
        'status': 'SUCCESS',
        'timestamp': timestamp,
        'records': num_records,
        'artifacts': {
            'data': f"s3://{config['storage']['bucket_name']}/{validated_data_path}",
            'pipeline': f"s3://{pipeline_bucket}/{pipeline_yaml_path}",
            'parameters': f"s3://{config['storage']['bucket_name']}/{pipeline_params_path}"
        }
    }

    return notification


# ===================================================================
# DAG DEFINITION
# ===================================================================

# Weekly Training Pipeline Preparation
with DAG(
    'electricity_pipeline_preparation',
    default_args=default_args,
    description='Weekly pipeline: Extract → Validate → Compile → Generate Parameters → Cleanup',
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

    # Task 3: Generate data quality summary
    quality_summary_task = PythonOperator(
        task_id='data_quality_summary',
        python_callable=generate_data_quality_summary,
    )

    # Task 4: Compile Kubeflow Pipeline YAML → save to MinIO pipelines/
    compile_and_upload_task = PythonOperator(
        task_id='compile_and_upload_pipeline',
        python_callable=compile_and_upload_pipeline,
    )

    # Task 5: Generate pipeline parameters → save to MinIO
    generate_params_task = PythonOperator(
        task_id='generate_pipeline_parameters',
        python_callable=generate_pipeline_parameters,
    )

    # Task 6: Cleanup old files from MinIO
    cleanup_task = PythonOperator(
        task_id='cleanup_old_files',
        python_callable=cleanup_old_files,
    )

    # Task 7: Send success notification
    notification_task = PythonOperator(
        task_id='send_success_notification',
        python_callable=send_success_notification,
    )

    # Define task dependencies
    extract_task >> validate_task >> quality_summary_task >> compile_and_upload_task >> generate_params_task >> cleanup_task >> notification_task


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