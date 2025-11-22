#!/usr/bin/env python3
"""
Test script for data validation using Pandera
Run this after test_extraction.py
"""

import logging
import pandas as pd
import sys
import os

# Add parent directory to path so we can import from core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.data_validation import ElectricityDataValidator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Testing Data Validation with Pandera")
    logger.info("=" * 60)

    # Check if data file exists
    data_file = "electricity_data.csv"
    if not os.path.exists(data_file):
        # Try parent directory
        data_file = "../data/electricity_data.csv"
        if not os.path.exists(data_file):
            logger.error("❌ electricity_data.csv not found")
            logger.error("Run test_extraction.py first to generate the data")
            sys.exit(1)

    try:
        # Load data
        logger.info(f"\n1. Loading data from {data_file}...")
        df = pd.read_csv(data_file)
        df['period'] = pd.to_datetime(df['period'])
        logger.info(f"✓ Loaded {len(df)} records")

        if 'respondent' in df.columns:
            logger.info(f"✓ Found {df['respondent'].nunique()} unique regions")
            logger.info(f"   Regions: {sorted(df['respondent'].unique().tolist())[:10]}...")

        # Initialize validator
        logger.info("\n2. Initializing validator...")
        # Try to find config file
        config_path = "../config.yaml"
        if not os.path.exists(config_path):
            config_path = "config.yaml"

        if os.path.exists(config_path):
            validator = ElectricityDataValidator(config_path=config_path)
        else:
            logger.warning("⚠️ config.yaml not found, using default settings")
            # Create a minimal validator without config
            validator = ElectricityDataValidator.__new__(ElectricityDataValidator)
            validator.schema = validator._create_schema()

        logger.info("✓ Validator initialized")

        # Run validation
        logger.info("\n3. Running validation...")
        validated_df, report = validator.validate(df)

        # Display results
        logger.info("\n4. Validation Results:")
        logger.info("-" * 60)

        if report['is_valid']:
            logger.info("✅ Status: VALID")
        else:
            logger.warning("⚠️ Status: INVALID")

        logger.info(f"   Errors: {len(report['errors'])}")
        logger.info(f"   Warnings: {len(report['warnings'])}")

        # Display errors
        if report['errors']:
            logger.info("\n5. Errors:")
            for i, error in enumerate(report['errors'], 1):
                logger.error(f"   {i}. {error}")

        # Display warnings
        if report['warnings']:
            logger.info("\n6. Warnings:")
            for i, warning in enumerate(report['warnings'], 1):
                logger.warning(f"   {i}. {warning}")

        # Display statistics
        logger.info("\n7. Data Statistics:")
        stats = report['stats']
        logger.info(f"   Total records: {stats['total_records']}")
        logger.info(f"   Missing values: {stats['missing_values']}")
        logger.info(f"   Unique regions: {stats['unique_regions']}")

        if 'regions' in stats:
            logger.info(f"   Regions: {', '.join(stats['regions'][:5])}{'...' if len(stats['regions']) > 5 else ''}")

        logger.info(f"   Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")

        logger.info("\n8. Value Statistics:")
        value_stats = stats['value_stats']
        logger.info(f"   Mean: {value_stats['mean']:.2f}")
        logger.info(f"   Std: {value_stats['std']:.2f}")
        logger.info(f"   Min: {value_stats['min']:.2f}")
        logger.info(f"   Max: {value_stats['max']:.2f}")
        logger.info(f"   Median: {value_stats['median']:.2f}")

        # Display per-region stats
        if 'per_region' in stats:
            logger.info("\n9. Per-Region Statistics (first 5 regions):")
            for i, (region, region_stats) in enumerate(list(stats['per_region'].items())[:5], 1):
                logger.info(f"   {region}:")
                logger.info(f"      Records: {region_stats['records']}")
                logger.info(f"      Mean: {region_stats['mean']:.2f}")
                logger.info(f"      Range: {region_stats['min']:.2f} - {region_stats['max']:.2f}")

        if 'outliers' in stats:
            logger.info("\n10. Outlier Detection:")
            outliers = stats['outliers']
            logger.info(f"   Total count: {outliers['total_count']}")
            logger.info(f"   Overall percentage: {outliers['overall_percentage']:.2f}%")

        # Save validation report
        logger.info("\n11. Saving validation report...")
        validator.save_validation_report(report, "validation_report.json")
        logger.info("✓ Report saved to validation_report.json")

        # Save validated data
        logger.info("\n12. Saving validated data...")
        validated_df.to_csv("validated_data.csv", index=False)
        logger.info("✓ Validated data saved to validated_data.csv")

        logger.info("\n" + "=" * 60)
        if report['is_valid']:
            logger.info("✅ ALL VALIDATION TESTS PASSED!")
        else:
            logger.warning("⚠️ VALIDATION COMPLETED WITH ERRORS")
            logger.warning("Review the errors above before proceeding")
        logger.info("=" * 60)
        logger.info("\nFiles created:")
        logger.info("  - validation_report.json")
        logger.info("  - validated_data.csv")
        logger.info("\nNext step: Run test_model_training.py")

    except Exception as e:
        logger.error("\n" + "=" * 60)
        logger.error(f"❌ TEST FAILED: {str(e)}")
        logger.error("=" * 60)
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()