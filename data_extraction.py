"""
Data Extraction Module for EIA Electricity API
Fetches hourly electricity load data from different regions
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import yaml
from io import BytesIO

logger = logging.getLogger(__name__)


class EIADataExtractor:
    """Extract electricity load data from EIA API"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.api_key = self.config['api']['eia_api_key']
        self.base_url = self.config['api']['eia_base_url']
        self.endpoint = self.config['api']['endpoint']
    
    def fetch_electricity_data(
        self,
        start_date: str,
        end_date: str,
        regions: Optional[List[str]] = None,
        data_type: str = "D"  # D = Demand, NG = Net Generation
    ) -> pd.DataFrame:
        """
        Fetch electricity data from EIA API
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            regions: List of region codes (e.g., ['CAL', 'MIDA', 'TEX'])
            data_type: Data type code (D=Demand, NG=Net Generation)
        
        Returns:
            DataFrame with electricity data
        """
        url = f"{self.base_url}{self.endpoint}"
        
        params = {
            "api_key": self.api_key,
            "frequency": "hourly",
            "data[]": "value",
            "facets[type][]": data_type,
            "start": start_date,
            "end": end_date,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000  # Max records per request
        }
        
        if regions:
            for region in regions:
                params[f"facets[respondent][]"] = region
        
        all_data = []
        
        while True:
            logger.info(f"Fetching data with offset {params['offset']}")
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                logger.error(f"API request failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                raise Exception(f"API request failed: {response.status_code}")
            
            data = response.json()
            
            if 'response' not in data or 'data' not in data['response']:
                logger.warning("No data in response")
                break
            
            records = data['response']['data']
            if not records:
                break
            
            all_data.extend(records)
            
            # Check if there are more pages
            total = data['response'].get('total', 0)
            if params['offset'] + params['length'] >= total:
                break
            
            params['offset'] += params['length']
        
        if not all_data:
            logger.warning("No data retrieved from API")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Process datetime
        df['period'] = pd.to_datetime(df['period'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Sort by time
        df = df.sort_values('period').reset_index(drop=True)
        
        logger.info(f"Fetched {len(df)} records from {df['period'].min()} to {df['period'].max()}")
        
        return df
    
    def fetch_recent_data(self, days: int = 30, regions: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch recent electricity data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.fetch_electricity_data(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            regions=regions
        )
    
    def save_to_csv(self, df: pd.DataFrame, filepath: str):
        """Save DataFrame to CSV"""
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
    
    def save_to_minio(self, df: pd.DataFrame, object_name: str):
        """Save DataFrame to MinIO"""
        from minio import Minio
        
        # Initialize MinIO client
        client = Minio(
            self.config['storage']['minio_endpoint'],
            access_key=self.config['storage']['minio_access_key'],
            secret_key=self.config['storage']['minio_secret_key'],
            secure=False
        )
        
        bucket_name = self.config['storage']['bucket_name']
        
        # Create bucket if it doesn't exist
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logger.info(f"Created bucket: {bucket_name}")
        
        # Convert DataFrame to CSV bytes
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        csv_buffer = BytesIO(csv_bytes)
        
        # Upload to MinIO
        client.put_object(
            bucket_name,
            object_name,
            csv_buffer,
            length=len(csv_bytes),
            content_type='text/csv'
        )
        
        logger.info(f"Data uploaded to MinIO: s3://{bucket_name}/{object_name}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    extractor = EIADataExtractor()
    
    # Fetch data for California (CAL) region
    df = extractor.fetch_recent_data(days=90, regions=['CAL'])
    
    print(f"Data shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData statistics:\n{df['value'].describe()}")
    
    # Save locally
    extractor.save_to_csv(df, "electricity_data.csv")
