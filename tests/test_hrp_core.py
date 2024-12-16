import pytest
import pandas as pd
import numpy as np
from src.core.hrp_core import create_hrp_portfolio
from src.proc.data_handler import DataHandler
from datetime import datetime, timedelta

# Global test parameters
START_DATE = '2019-05-01'
END_DATE = '2024-12-10'

@pytest.fixture
def sp500_data():
    handler = DataHandler()
    df = handler.process_data('data/sp500_data.csv')
    return df['ticker'].tolist(), df['market_cap'].values

def test_sp500_portfolio_creation():
    tickers, market_caps = sp500_data()    
    weights, linkage, returns = create_hrp_portfolio(
        tickers=tickers,
        market_caps=market_caps,
        start_date=START_DATE,
        end_date=END_DATE,
        num_clusters=8
    )
    
    assert len(weights) == len(tickers)
    assert np.isclose(weights.sum(), 1.0, rtol=1e-5)
    assert (weights >= 0).all()
    assert (weights[weights > 0] >= 0.01).all()
    assert returns.shape[1] == len(tickers)
    assert not returns.isna().any().any()
    assert linkage.shape[0] == len(tickers) - 1

def test_sp500_data_dates():
    tickers, market_caps = sp500_data()
    
    start = datetime.strptime(START_DATE, '%Y-%m-%d')
    end = datetime.strptime(END_DATE, '%Y-%m-%d')
    mid_point = start + (end - start) / 2
    
    test_periods = [
        (start.strftime('%Y-%m-%d'), mid_point.strftime('%Y-%m-%d')),
        (mid_point.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')),
        (START_DATE, END_DATE)
    ]
    
    for period_start, period_end in test_periods:
        weights, linkage, returns = create_hrp_portfolio(
            tickers=tickers,
            market_caps=market_caps,
            start_date=period_start,
            end_date=period_end,
            num_clusters=8
        )
        
        assert returns.index[0].strftime('%Y-%m-%d') >= period_start
        assert returns.index[-1].strftime('%Y-%m-%d') <= period_end
        assert len(returns) > 0

def test_sp500_cluster_counts():
    tickers, market_caps = sp500_data()
    
    for num_clusters in [5, 8, 10]:
        weights, linkage, returns = create_hrp_portfolio(
            tickers=tickers,
            market_caps=market_caps,
            start_date=START_DATE,
            end_date=END_DATE,
            num_clusters=num_clusters
        )
        
        assert np.isclose(weights.sum(), 1.0, rtol=1e-5)
        assert len(weights[weights > 0]) >= num_clusters