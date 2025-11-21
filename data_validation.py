"""
Data Validation Module using Pandera
Validates electricity load data quality and schema
"""

import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema
from typing import Tuple
import logging
import yaml

logger = logging.getLogger(__name__)


class ElectricityDataValidator:
    """Validate electricity load data"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.schema = self._create_schema()
    
    def _create_schema(self) -> DataFrameSchema:
        """Create Pandera schema for electricity data"""
        
        schema = DataFrameSchema(
            columns={
                "period": Column(
                    pa.DateTime,
                    checks=[
                        Check(lambda s: s.notna().all(), 
                              error="period cannot contain null values"),
                        Check(lambda s: s.is_monotonic_increasing, 
                              error="period must be monotonically increasing")
                    ],
                    nullable=False,
                    coerce=True
                ),
                "respondent": Column(
                    pa.String,
                    checks=[
                        Check(lambda s: s.notna().all(),
                              error="respondent cannot be null")
                    ],
                    nullable=False
                ),
                "type": Column(
                    pa.String,
                    checks=[
                        Check.isin(["D", "NG", "TI", "ID"]),  # Common EIA data types
                    ],
                    nullable=False
                ),
                "value": Column(
                    pa.Float,
                    checks=[
                        Check.greater_than_or_equal_to(0, 
                              error="value must be non-negative"),
                        Check(lambda s: s.notna().all(),
                              error="value cannot be null"),
                        Check(lambda s: (s < s.quantile(0.99) * 3).all(),
                              error="Detected extreme outliers")
                    ],
                    nullable=False,
                    coerce=True
                ),
            },
            strict=False,  # Allow additional columns
            coerce=True
        )
        
        return schema
    
    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Validate DataFrame against schema
        
        Returns:
            Tuple of (validated_df, validation_report)
        """
        validation_report = {
            "is_valid": False,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        try:
            # Validate schema
            validated_df = self.schema.validate(df, lazy=True)
            
            # Additional custom validations
            self._check_data_continuity(validated_df, validation_report)
            self._check_data_quality(validated_df, validation_report)
            self._check_outliers(validated_df, validation_report)
            
            if not validation_report["errors"]:
                validation_report["is_valid"] = True
                logger.info("Data validation passed!")
            else:
                logger.warning(f"Data validation failed with {len(validation_report['errors'])} errors")
            
            return validated_df, validation_report
        
        except pa.errors.SchemaErrors as e:
            logger.error(f"Schema validation failed: {e}")
            validation_report["errors"].append(str(e))
            return df, validation_report
    
    def _check_data_continuity(self, df: pd.DataFrame, report: dict):
        """Check for gaps in time series"""
        if len(df) < 2:
            return
        
        # Check hourly continuity
        df_sorted = df.sort_values('period')
        time_diffs = df_sorted['period'].diff()
        
        # Expected: 1 hour between consecutive records
        expected_diff = pd.Timedelta(hours=1)
        gaps = time_diffs[time_diffs > expected_diff * 1.5]  # Allow 50% tolerance
        
        if len(gaps) > 0:
            report["warnings"].append(
                f"Found {len(gaps)} time gaps in data. "
                f"Largest gap: {gaps.max()}"
            )
            logger.warning(f"Found {len(gaps)} time gaps")
    
    def _check_data_quality(self, df: pd.DataFrame, report: dict):
        """Check data quality metrics"""
        
        # Check for missing values
        missing_pct = (df['value'].isna().sum() / len(df)) * 100
        if missing_pct > 5:
            report["errors"].append(
                f"Too many missing values: {missing_pct:.2f}% (threshold: 5%)"
            )
        elif missing_pct > 0:
            report["warnings"].append(
                f"Found {missing_pct:.2f}% missing values"
            )
        
        # Check for duplicate timestamps
        duplicates = df['period'].duplicated().sum()
        if duplicates > 0:
            report["errors"].append(
                f"Found {duplicates} duplicate timestamps"
            )
        
        # Check data statistics
        report["stats"] = {
            "total_records": len(df),
            "missing_values": df['value'].isna().sum(),
            "unique_regions": df['respondent'].nunique(),
            "date_range": {
                "start": df['period'].min().isoformat(),
                "end": df['period'].max().isoformat()
            },
            "value_stats": {
                "mean": float(df['value'].mean()),
                "std": float(df['value'].std()),
                "min": float(df['value'].min()),
                "max": float(df['value'].max()),
                "median": float(df['value'].median())
            }
        }
    
    def _check_outliers(self, df: pd.DataFrame, report: dict):
        """Detect outliers using IQR method"""
        Q1 = df['value'].quantile(0.25)
        Q3 = df['value'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
        outlier_pct = (len(outliers) / len(df)) * 100
        
        if outlier_pct > 2:
            report["warnings"].append(
                f"Found {len(outliers)} outliers ({outlier_pct:.2f}% of data)"
            )
        
        report["stats"]["outliers"] = {
            "count": len(outliers),
            "percentage": outlier_pct,
            "bounds": {
                "lower": float(lower_bound),
                "upper": float(upper_bound)
            }
        }
    
    def save_validation_report(self, report: dict, filepath: str):
        """Save validation report to JSON"""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load sample data
    df = pd.read_csv("electricity_data.csv")
    df['period'] = pd.to_datetime(df['period'])
    
    # Validate
    validator = ElectricityDataValidator()
    validated_df, report = validator.validate(df)
    
    print("\n=== Validation Report ===")
    print(f"Valid: {report['is_valid']}")
    print(f"Errors: {len(report['errors'])}")
    print(f"Warnings: {len(report['warnings'])}")
    print(f"\nStats: {report['stats']}")
    
    if report['errors']:
        print("\nErrors:")
        for error in report['errors']:
            print(f"  - {error}")
    
    if report['warnings']:
        print("\nWarnings:")
        for warning in report['warnings']:
            print(f"  - {warning}")
