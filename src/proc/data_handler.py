import duckdb
import pandas as pd
import numpy as np
import logging
from typing import Optional

class DataHandler:
    """
    Handles data loading and preprocessing for HRP portfolio optimization.
    
    This class manages:
    - Data loading from CSV files using DuckDB
    - Data validation and quality checks
    - Calculation of quality metrics
    - Financial strength scoring
    - Value scoring
    - Composite score generation
    """
    
    def __init__(self):
        """Initialize DataHandler with database connection and logger."""
        self.con = None
        self.logger = logging.getLogger('DataHandler')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
    def _validate_market_data(self, df: pd.DataFrame) -> None:
        """
        Validates loaded market data with different severity levels.
        
        Critical issues raise errors, potential issues generate warnings.
        """
        # CRITICAL CHECKS (raise errors)
        # Check for zero/negative market caps
        bad_mcap = df[df['market_cap'] <= 0]
        if not bad_mcap.empty:
            tickers = bad_mcap['Full Ticker'].tolist()
            raise ValueError(f"Found zero or negative market cap values for tickers: {tickers}")
            
        # Check for missing fundamental data beyond threshold
        missing_threshold = 0.20
        for col in ['croic', 'operating_income_margin', 'asset_turnover', 
                   'net_debt_to_ebitda', 'interest_coverage_ratio', 'current_ratio']:
            missing_pct = df[col].isna().mean()
            if missing_pct > missing_threshold:
                raise ValueError(f"Column {col} missing {missing_pct:.1%} of values, exceeding {missing_threshold:.1%} threshold")
        
        # WARNING CHECKS (log warnings)
        # Check for unusual operating margins
        bad_margins = df[df['operating_income_margin'].abs() > 1]
        if not bad_margins.empty:
            tickers = bad_margins['Full Ticker'].tolist()
            values = bad_margins['operating_income_margin'].tolist()
            self.logger.warning(f"Unusual operating margins (>±100%) found for:\nTickers: {tickers}\nValues: {values}")
            
        # Check for unusual current ratios
        bad_current = df[df['current_ratio'] <= 0]
        if not bad_current.empty:
            tickers = bad_current['Full Ticker'].tolist()
            self.logger.warning(f"Zero or negative current ratios found for tickers: {tickers}")
            
        # Check for unusual interest coverage ratios
        bad_coverage = df[df['interest_coverage_ratio'] <= -100]
        if not bad_coverage.empty:
            tickers = bad_coverage['Full Ticker'].tolist()
            values = df[df['interest_coverage_ratio'] <= -100]['interest_coverage_ratio'].tolist()
            self.logger.warning(f"Unusual interest coverage ratios (<-100) found for:\nTickers: {tickers}\nValues: {values}")
    
    def _validate_returns(self, returns: pd.DataFrame) -> None:
        """
        Validates return data for anomalies.
        
        Checks:
        - Unrealistic daily returns
        - Perfect correlations
        - Excessive missing data
        
        Raises:
        -------
        ValueError: If return data appears problematic
        """
        # Check for unrealistic daily returns (e.g., >50% daily move)
        max_daily_return = 0.50
        if (returns.abs() > max_daily_return).any().any():
            raise ValueError(f"Found daily returns exceeding ±{max_daily_return:.0%}")
        
        # Check for perfect correlations that might indicate data issues
        corr_matrix = returns.corr()
        np.fill_diagonal(corr_matrix.values, np.nan)  # Ignore self-correlations
        if (corr_matrix.abs() > 0.9999).any().any():
            raise ValueError("Found perfect correlations between different assets, possible data quality issue")
        
        # Check for excessive missing data
        missing_threshold = 0.10
        missing_pct = returns.isna().mean()
        if (missing_pct > missing_threshold).any():
            raise ValueError(f"Some assets missing more than {missing_threshold:.1%} of return data")
    
    def load_market_data(self, csv_file_path: str, min_market_cap: float = 2000) -> pd.DataFrame:
        """
        Load and preprocess market data from CSV file.
        
        Parameters:
        -----------
        csv_file_path : str
            Path to the CSV file containing market data
        min_market_cap : float
            Minimum market cap filter in millions
            
        Returns:
        --------
        pd.DataFrame
            Processed market data with converted types
        """
        self.con = duckdb.connect()
        
        # Create table from CSV
        self.con.execute(f"""
        CREATE TABLE tickers AS
        SELECT *
        FROM read_csv_auto('{csv_file_path}');
        """)
        
        # Transform and query data
        query = f"""
        SELECT 
            "Name", 
            "Full Ticker", 
            try_cast("Price, Current" AS DOUBLE) AS price, 
            try_cast("Market Cap" AS DOUBLE) AS market_cap,
            try_cast("Cash Return On Invested Capital (CROIC)" AS DOUBLE) AS croic,
            try_cast("Operating Income Margin" AS DOUBLE) AS operating_income_margin,
            try_cast("Asset Turnover" AS DOUBLE) AS asset_turnover,
            try_cast("Net Debt / EBITDA" AS DOUBLE) AS net_debt_to_ebitda,
            try_cast("Interest Coverage Ratio" AS DOUBLE) AS interest_coverage_ratio,
            try_cast("Current Ratio" AS DOUBLE) AS current_ratio,
            try_cast("Fair Value" AS DOUBLE) AS fair_value,
            try_cast("Fair Value (Analyst Targets)" AS DOUBLE) AS fair_value_analyst,
            try_cast("Analyst Price Target High" AS DOUBLE) AS analyst_target_high,
            try_cast("Analyst Price Target Low" AS DOUBLE) AS analyst_target_low
        FROM tickers
        WHERE try_cast("Market Cap" AS DOUBLE) > {min_market_cap};
        """
        
        df = self.con.execute(query).df()
        self.con.close()
        self.con = None
        
        df = df.dropna()
        
        # Validate loaded data
        self._validate_market_data(df)
        
        return df
    
    def prepare_hrp_data(self, df: pd.DataFrame, n_stocks: int) -> pd.DataFrame:
        """Prepare final DataFrame for HRP optimization."""
        df = self.calculate_quality_metrics(df)
        df = self.calculate_financial_strength(df)
        df = self.calculate_value_metrics(df)
        df = self.calculate_composite_score(df)
        
        top_n_df = df.head(n_stocks)
        
        hrp_df = top_n_df[['Full Ticker', 'market_cap']].copy()
        hrp_df['ticker'] = hrp_df['Full Ticker'].str.split(':').str[1]
        
        return hrp_df[['ticker', 'market_cap']]

    def process_data(self, csv_file_path: str, n_stocks: int = 50) -> pd.DataFrame:
        """Complete data processing pipeline."""
        df = self.load_market_data(csv_file_path)
        return self.prepare_hrp_data(df, n_stocks)
    
    def calculate_quality_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate quality metrics including CROIC, operating margin, and asset turnover.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with required financial metrics
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added quality metrics
        """
        df = df.copy()
        
        # Standardize components
        df['croic_z'] = (df['croic'] - df['croic'].mean()) / df['croic'].std()
        df['margin_z'] = (df['operating_income_margin'] - df['operating_income_margin'].mean()) / df['operating_income_margin'].std()
        df['turnover_z'] = (df['asset_turnover'] - df['asset_turnover'].mean()) / df['asset_turnover'].std()
        
        # Calculate quality score
        df['quality_score'] = df[['croic_z', 'margin_z', 'turnover_z']].mean(axis=1)
        
        return df
    
    def calculate_financial_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate financial strength metrics including debt ratios and coverage.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with required financial metrics
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added financial strength metrics
        """
        df = df.copy()
        
        # Standardize components
        df['debt_z'] = (-df['net_debt_to_ebitda'] - (-df['net_debt_to_ebitda']).mean()) / df['net_debt_to_ebitda'].std()
        df['coverage_z'] = (df['interest_coverage_ratio'] - df['interest_coverage_ratio'].mean()) / df['interest_coverage_ratio'].std()
        df['current_z'] = (df['current_ratio'] - df['current_ratio'].mean()) / df['current_ratio'].std()
        
        # Calculate weighted financial strength score
        df['financial_strength_score'] = (
            0.4 * df['debt_z'] +
            0.4 * df['coverage_z'] +
            0.2 * df['current_z']
        )
        
        return df
    
    def calculate_value_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate value metrics including fair value and analyst target ratios.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with required price and valuation metrics
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added value metrics
        """
        df = df.copy()
        
        # Calculate upside ratios
        df['fair_value_upside'] = (df['fair_value'] / df['price'] - 1)
        df['analyst_value_upside'] = (df['fair_value_analyst'] / df['price'] - 1)
        df['analyst_target_upside'] = (df['analyst_target_high'] / df['price'] - 1)
        
        # Standardize components
        df['fair_z'] = (df['fair_value_upside'] - df['fair_value_upside'].mean()) / df['fair_value_upside'].std()
        df['analyst_z'] = (df['analyst_value_upside'] - df['analyst_value_upside'].mean()) / df['analyst_value_upside'].std()
        df['target_z'] = (df['analyst_target_upside'] - df['analyst_target_upside'].mean()) / df['analyst_target_upside'].std()
        
        # Calculate weighted value score
        df['value_score'] = (
            0.4 * df['fair_z'] +
            0.4 * df['analyst_z'] +
            0.2 * df['target_z']
        )
        
        return df
    
    def calculate_composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate final composite score from quality, financial strength, and value metrics.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with all component scores
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added composite score, sorted by score
        """
        df = df.copy()
        
        df['composite_score'] = (
            df['quality_score'] +
            df['financial_strength_score'] +
            df['value_score']
        ) / 3
        
        return df.sort_values('composite_score', ascending=False)
    
    def prepare_hrp_data(self, df: pd.DataFrame, n_stocks: int = 400) -> pd.DataFrame:
        """
        Prepare final DataFrame for HRP optimization.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with composite scores
        n_stocks : int
            Number of top stocks to include
            
        Returns:
        --------
        pd.DataFrame
            DataFrame ready for HRP optimization
        """
        top_n_df = df.head(n_stocks)
        
        hrp_df = top_n_df.copy()[['Full Ticker', 'market_cap']]
        hrp_df['ticker'] = hrp_df['Full Ticker'].apply(lambda x: x.split(':')[1])
        
        return hrp_df[['ticker', 'market_cap']]
    
    def process_data(self, csv_file_path: str, n_stocks: int = 100) -> pd.DataFrame:
        """
        Complete data processing pipeline.
        
        Parameters:
        -----------
        csv_file_path : str
            Path to input CSV file
        n_stocks : int
            Number of top stocks to include
            
        Returns:
        --------
        pd.DataFrame
            Final DataFrame ready for HRP optimization
        """
        df = self.load_market_data(csv_file_path)
        df = self.calculate_quality_metrics(df)
        df = self.calculate_financial_strength(df)
        df = self.calculate_value_metrics(df)
        df = self.calculate_composite_score(df)
        return self.prepare_hrp_data(df, n_stocks)