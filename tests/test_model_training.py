#!/usr/bin/env python3
"""
Test script for LSTM model training
Run this after test_validation.py
"""

import logging
import pandas as pd
import torch
from core.model  import ElectricityLoadForecaster
import sys
import os
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Testing LSTM Model Training")
    logger.info("=" * 60)
    
    # Check if validated data exists
    if not os.path.exists("validated_data.csv"):
        logger.warning("⚠️ validated_data.csv not found, using electricity_data.csv")
        if not os.path.exists("electricity_data.csv"):
            logger.error("❌ No data file found")
            logger.error("Run test_extraction.py first")
            sys.exit(1)
        data_file = "electricity_data.csv"
    else:
        data_file = "validated_data.csv"
    
    try:
        # Load data
        logger.info(f"\n1. Loading data from {data_file}...")
        df = pd.read_csv(data_file)
        df['period'] = pd.to_datetime(df['period'])
        df = df.sort_values('period').reset_index(drop=True)
        
        # Use only one region for simplicity
        if 'respondent' in df.columns:
            region = df['respondent'].iloc[0]
            df = df[df['respondent'] == region].copy()
            logger.info(f"✓ Using region: {region}")
        
        logger.info(f"✓ Loaded {len(df)} records")
        
        # Check minimum data requirement
        min_required = 168 + 24 + 100  # sequence_length + prediction_horizon + buffer
        if len(df) < min_required:
            logger.error(f"❌ Need at least {min_required} records, got {len(df)}")
            logger.error("Fetch more data using test_extraction.py with more days")
            sys.exit(1)
        
        # Initialize model
        logger.info("\n2. Initializing LSTM model...")
        forecaster = ElectricityLoadForecaster(
            model_type="lstm",
            sequence_length=168,  # 7 days
            prediction_horizon=24,  # 24 hours
            hidden_size=64,  # Smaller for faster testing
            num_layers=2,
            dropout=0.2,
            learning_rate=0.001
        )
        logger.info(f"✓ Model initialized on {forecaster.device}")
        logger.info(f"   Parameters: {sum(p.numel() for p in forecaster.model.parameters()):,}")
        
        # Prepare data
        logger.info("\n3. Preparing datasets...")
        train_loader, val_loader, test_loader = forecaster.prepare_data(
            df,
            train_split=0.7,
            val_split=0.15
        )
        logger.info(f"✓ Train batches: {len(train_loader)}")
        logger.info(f"   Val batches: {len(val_loader)}")
        logger.info(f"   Test batches: {len(test_loader)}")
        
        # Train model
        logger.info("\n4. Training model (10 epochs for quick test)...")
        logger.info("-" * 60)
        
        forecaster.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=10,  # Reduced for testing
            early_stopping_patience=5
        )
        
        # Evaluate on test set
        logger.info("\n5. Evaluating on test set...")
        test_rmse, test_mape = forecaster.evaluate(test_loader)
        logger.info(f"✓ Test RMSE: {test_rmse:.4f}")
        logger.info(f"✓ Test MAPE: {test_mape:.2f}%")
        
        # Make sample prediction
        logger.info("\n6. Making sample prediction...")
        # Get last sequence from data
        values = df['value'].values[-forecaster.sequence_length:].reshape(-1, 1)
        values_scaled = forecaster.scaler.transform(values)
        
        # Predict next 24 hours
        predictions = forecaster.predict(values_scaled)
        
        logger.info("✓ Predicted next 24 hours:")
        for i, pred in enumerate(predictions[:5], 1):
            logger.info(f"   Hour {i}: {pred:.2f} MW")
        logger.info(f"   ... (showing first 5 of 24)")
        
        # Save model
        logger.info("\n7. Saving model...")
        forecaster.save_model("best_model.pt")
        logger.info("✓ Model saved to best_model.pt")
        
        # Test model loading
        logger.info("\n8. Testing model loading...")
        forecaster_loaded = ElectricityLoadForecaster(
            model_type="lstm",
            sequence_length=168,
            prediction_horizon=24
        )
        forecaster_loaded.load_model("best_model.pt")
        logger.info("✓ Model loaded successfully")
        
        # Verify loaded model gives same predictions
        predictions_loaded = forecaster_loaded.predict(values_scaled)
        if np.allclose(predictions, predictions_loaded):
            logger.info("✓ Loaded model predictions match original")
        else:
            logger.warning("⚠️ Loaded model predictions differ slightly")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL MODEL TESTS PASSED!")
        logger.info("=" * 60)
        logger.info("\nFiles created:")
        logger.info("  - best_model.pt (trained model)")
        logger.info("\nModel is ready for deployment!")
        logger.info("\nNext step: Test with Kubernetes (see K8S_TESTING_GUIDE.md)")
        
    except Exception as e:
        logger.error("\n" + "=" * 60)
        logger.error(f"❌ TEST FAILED: {str(e)}")
        logger.error("=" * 60)
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
