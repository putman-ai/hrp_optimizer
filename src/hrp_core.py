import yfinance as yf
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from typing import List, Dict, Tuple
import logging

def create_hrp_portfolio(
    tickers: List[str],
    market_caps: np.ndarray,
    start_date: str,
    end_date: str,
    num_clusters: int = 3
) -> Tuple[pd.Series, np.ndarray, pd.DataFrame]:
    """
    Creates a Hierarchical Risk Parity portfolio with market cap weighting.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        if len(tickers) != len(market_caps):
            raise ValueError("Number of tickers must match number of market caps")
        
        market_cap_dict = dict(zip(tickers, market_caps))
        total_market_cap = sum(market_caps)
        market_cap_dict = {k: v/total_market_cap for k, v in market_cap_dict.items()}
        
        logger.info("Downloading historical data...")
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        
        data = data.dropna(axis=1)
        if data.empty:
            raise ValueError("No valid data after removing NA values")
            
        returns = data.pct_change().dropna()
        
        logger.info("Calculating correlation and distance matrices...")
        corr_matrix = returns.corr()
        dist_matrix = np.sqrt(np.clip((1 - corr_matrix) / 2, 0, 1))
        
        logger.info("Performing hierarchical clustering...")
        linkage_matrix = linkage(pdist(dist_matrix), method='ward')
        clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
        
        logger.info("Calculating cluster risk metrics...")
        cluster_inv_variances: Dict[int, float] = {}
        for i in np.unique(clusters):
            cluster_returns = returns.iloc[:, clusters == i].mean(axis=1)
            cluster_variance = np.var(cluster_returns)
            if cluster_variance == 0:
                logger.warning(f"Cluster {i} has zero variance, setting to small positive number")
                cluster_variance = 1e-8
            cluster_inv_variances[i] = 1 / cluster_variance
            
        total_inv_variance = sum(cluster_inv_variances.values())
        weights = pd.Series(index=data.columns, dtype=float)
        for i in np.unique(clusters):
            cluster_weight = cluster_inv_variances[i] / total_inv_variance
            cluster_assets = clusters == i
            weights[cluster_assets] = cluster_weight / np.sum(cluster_assets)
            
        logger.info("Applying market cap constraints...")
        available_market_caps = pd.Series({ticker: market_cap_dict[ticker] 
                                         for ticker in data.columns 
                                         if ticker in market_cap_dict})
        
        weights = weights * available_market_caps
        weights = weights / weights.sum()
        
        min_weight = 0.01
        weights[weights < min_weight] = 0
        weights = weights / weights.sum()
        
        logger.info("Portfolio optimization complete")
        return weights, linkage_matrix, returns
        
    except Exception as e:
        logger.error(f"Error in portfolio creation: {e}")
        raise

