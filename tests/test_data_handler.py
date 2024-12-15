import pandas as pd
import numpy as np
import pytest

from src.proc.data_handler import DataHandler

def test_load_market_data():
    """Test comprehensive data loading and validation from sp500_data.csv"""
    print("\nTesting load_market_data...")
    
    try:
        # Initialize handler and load data
        handler = DataHandler()
        df = handler.load_market_data('data/sp500_data.csv', min_market_cap=2000)
        
        # 1. Basic DataFrame Validations
        assert isinstance(df, pd.DataFrame), "Result should be a pandas DataFrame"
        assert not df.empty, "DataFrame should not be empty"
        assert len(df) > 0, "DataFrame should contain rows"
        
        # 2. Required Columns Present
        required_columns = [
            'Name', 'Full Ticker', 'price', 'market_cap', 'croic',
            'operating_income_margin', 'asset_turnover', 'net_debt_to_ebitda',
            'interest_coverage_ratio', 'current_ratio', 'fair_value',
            'fair_value_analyst', 'analyst_target_high', 'analyst_target_low'
        ]
        for col in required_columns:
            assert col in df.columns, f"Required column {col} missing from DataFrame"
        
        # 3. Data Type Validations
        numeric_columns = [
            'price', 'market_cap', 'croic', 'operating_income_margin',
            'asset_turnover', 'net_debt_to_ebitda', 'interest_coverage_ratio',
            'current_ratio', 'fair_value', 'fair_value_analyst',
            'analyst_target_high', 'analyst_target_low'
        ]
        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} should be numeric"
            
        # 4. Market Cap Filter Validation
        assert (df['market_cap'] >= 2000).all(), "All market caps should be above 2000M"
        
        # 5. Missing Value Handling
        assert not df.isnull().any().any(), "DataFrame should not contain any null values after processing"
        
        # 6. String Column Validations
        assert df['Name'].str.len().gt(0).all(), "All company names should be non-empty"
        assert df['Full Ticker'].str.contains(':').all(), "All tickers should be in format with ':'"
        
        # 7. Data Range Validations
        assert (df['price'] > 0).all(), "All prices should be positive"
        assert (df['market_cap'] > 0).all(), "All market caps should be positive"
        
        # 8. Sample Data Validation (if we know expected values)
        # Comment these in if we have known values to check against
        # sample_ticker = "NYSE:AAPL"  # Example
        # if sample_ticker in df['Full Ticker'].values:
        #     sample_row = df[df['Full Ticker'] == sample_ticker].iloc[0]
        #     assert sample_row['market_cap'] > 1000000, "Apple's market cap should be very large"
        
        print("✓ load_market_data test passed")
        print(f"Loaded {len(df)} companies")
        print(f"Columns validated: {', '.join(required_columns)}")
        print(f"Average market cap: ${df['market_cap'].mean():,.2f}M")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise

def test_load_market_data_invalid_path():
    """Test handling of invalid file path"""
    handler = DataHandler()
    with pytest.raises(Exception):
        handler.load_market_data('nonexistent_file.csv', min_market_cap=2000)

def test_calculate_quality_metrics():
    """Test quality metrics calculation and standardization"""
    print("\nTesting calculate_quality_metrics...")
    
    try:
        # Initialize handler and load test data
        handler = DataHandler()
        df = handler.load_market_data('data/sp500_data.csv', min_market_cap=2000)
        
        # Calculate quality metrics
        quality_df = handler.calculate_quality_metrics(df)
        
        # 1. Verify required columns exist
        required_columns = ['croic_z', 'margin_z', 'turnover_z', 'quality_score']
        for col in required_columns:
            assert col in quality_df.columns, f"Required column {col} missing"
            
        # 2. Validate standardization
        z_score_columns = ['croic_z', 'margin_z', 'turnover_z']
        for col in z_score_columns:
            assert abs(quality_df[col].mean()) < 0.0001, f"{col} mean should be close to 0"
            assert abs(quality_df[col].std() - 1) < 0.0001, f"{col} std should be close to 1"
            
        # 3. Verify quality score calculation
        calculated_score = quality_df[z_score_columns].mean(axis=1)
        quality_score_diff = (quality_df['quality_score'] - calculated_score).abs()
        assert (quality_score_diff < 0.0001).all(), "Quality score calculation mismatch"
        
        # 4. Test edge cases
        # Verify no infinite values
        assert not quality_df[required_columns].isin([np.inf, -np.inf]).any().any(), "Infinite values found"
        
        # Verify handling of extreme values
        max_z_score = quality_df[z_score_columns].abs().max().max()
        assert max_z_score < 10, f"Extreme z-scores detected: {max_z_score}"
        
        print("✓ calculate_quality_metrics test passed")
        print(f"Processed {len(quality_df)} companies")
        print(f"Z-score ranges:")
        for col in z_score_columns:
            print(f"{col}: [{quality_df[col].min():.2f}, {quality_df[col].max():.2f}]")
        print(f"Quality score range: [{quality_df['quality_score'].min():.2f}, {quality_df['quality_score'].max():.2f}]")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise
