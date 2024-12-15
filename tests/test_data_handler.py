import pandas as pd
import numpy as np
import pytest
from src.proc.data_handler import DataHandler

def test_load_market_data():
    """Test comprehensive data loading and validation"""
    print("\nTesting load_market_data...")
    
    try:
        handler = DataHandler()
        df = handler.load_market_data('data/sp500_data.csv', min_market_cap=2000)
        
        # Basic DataFrame Validations
        assert isinstance(df, pd.DataFrame), "Result should be a pandas DataFrame"
        assert not df.empty, "DataFrame should not be empty"
        assert len(df) > 0, "DataFrame should contain rows"
        assert all(df.columns), "All columns should have non-null headers"
        assert df.columns.is_unique, "Column headers should be unique"
        
        # Dynamic Column Type Detection
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        string_columns = df.select_dtypes(include=['object', 'string']).columns
        
        # Data Type Validations
        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} should be numeric"
            assert not df[col].isin([np.inf, -np.inf]).any(), f"Column {col} contains infinite values"
        
        # String Column Validations
        if 'Name' in df.columns:
            assert df['Name'].str.len().gt(0).all(), "All company names should be non-empty"
        if 'Full Ticker' in df.columns:
            assert df['Full Ticker'].str.contains(':').all(), "All tickers should be in format with ':'"
        
        # Basic Range Validations - Only test logical bounds
        assert (df['price'] > 0).all(), "All prices should be positive"
        assert (df['market_cap'] >= 2000).all(), "All market caps should be above minimum threshold"
        
        # Ratio Validations - Only test logical bounds
        if 'current_ratio' in df.columns:
            assert (df['current_ratio'] >= 0).all(), "Current ratio cannot be negative"
        
        if 'operating_income_margin' in df.columns:
            assert (df['operating_income_margin'] >= -100).all(), "Operating margin cannot be below -100%"
            assert (df['operating_income_margin'] <= 100).all(), "Operating margin cannot exceed 100%"
        
        # Analyst Target Consistency
        if 'analyst_target_high' in df.columns and 'analyst_target_low' in df.columns:
            assert (df['analyst_target_high'] >= df['analyst_target_low']).all(), "Target high should not be less than target low"
        
        # Print Summary Statistics
        print(f"\n✓ Data Validation Summary:")
        print(f"Total companies: {len(df)}")
        print(f"Market cap range: ${df['market_cap'].min():,.0f}M - ${df['market_cap'].max():,.0f}M")
        print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
        
        if 'operating_income_margin' in df.columns:
            print(f"Operating margin range: {df['operating_income_margin'].min():.1f}% - {df['operating_income_margin'].max():.1f}%")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise