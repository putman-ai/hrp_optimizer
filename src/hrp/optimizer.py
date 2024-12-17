import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging
import yfinance as yf

@dataclass
class PortfolioMetrics:
    weights: pd.Series
    linkage_matrix: np.ndarray
    returns: pd.DataFrame
    sharpe_ratio: float
    volatility: float
    expected_return: float
    clusters: np.ndarray
    max_weight: float

class HRPPortfolio:
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        min_weight: float = 0.01,
        max_weight: float = 0.15
    ):
        self.risk_free_rate = risk_free_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _calculate_ewma_covariance(self, returns: pd.DataFrame, lambda_decay: float = 0.94) -> pd.DataFrame:
        """
        Calculate exponentially weighted covariance matrix.
        """
        n_periods = len(returns)
        weights = np.array([(1 - lambda_decay) * lambda_decay ** i for i in range(n_periods-1, -1, -1)])
        weights = weights / weights.sum()
        
        demean_returns = returns - returns.mean()
        weighted_returns = demean_returns * np.sqrt(weights[:, np.newaxis])
        ewma_covar = weighted_returns.T @ weighted_returns
        
        return ewma_covar

    def _calculate_ewma_correlation(self, returns: pd.DataFrame, lambda_decay: float = 0.94) -> pd.DataFrame:
        """
        Calculate exponentially weighted correlation matrix.
        """
        ewma_covar = self._calculate_ewma_covariance(returns, lambda_decay)
        std_dev = np.sqrt(np.diag(ewma_covar))
        ewma_corr = ewma_covar / np.outer(std_dev, std_dev)
        
        ewma_corr = pd.DataFrame(
            ewma_corr, 
            index=returns.columns, 
            columns=returns.columns
        )
        
        return ewma_corr

    def _calculate_distance_matrix(self, returns: pd.DataFrame, risk_measure: str, **kwargs) -> np.ndarray:
        """
        Calculate distance matrix based on specified risk measure.
        """
        if risk_measure == "ewma":
            lambda_decay = kwargs.get('lambda_decay', 0.94)
            corr_matrix = self._calculate_ewma_correlation(returns, lambda_decay)
            dist_matrix = np.sqrt(np.clip((1 - corr_matrix) / 2, 0, 1))
            return pdist(dist_matrix)
            
        elif risk_measure == "variance":
            corr_matrix = returns.corr()
            dist_matrix = np.sqrt(np.clip((1 - corr_matrix) / 2, 0, 1))
            return pdist(dist_matrix)
            
        elif risk_measure == "mad":
            mad_matrix = returns.apply(lambda x: returns.apply(lambda y: np.mean(np.abs(x - y))))
            dist_matrix = mad_matrix / mad_matrix.max().max()
            return pdist(dist_matrix)
            
        elif risk_measure == "correlation":
            dist_matrix = returns.corr().abs()
            return pdist(dist_matrix)
            
        raise ValueError(f"Unsupported risk measure: {risk_measure}")

    def _calculate_optimal_weights(
        self,
        returns: pd.DataFrame,
        clusters: np.ndarray,
        market_caps: pd.Series
    ) -> pd.Series:
        """Calculate optimal weights using HRP with market cap weighting within clusters."""
        cluster_weights = self._get_cluster_weights(returns, clusters)
        weights = pd.Series(0, index=returns.columns)
        
        for cluster in np.unique(clusters):
            cluster_assets = clusters == cluster
            cluster_market_caps = market_caps[returns.columns[cluster_assets]]
            normalized_caps = cluster_market_caps / cluster_market_caps.sum()
            weights[cluster_assets] = cluster_weights[cluster] * normalized_caps
        
        weights = self._apply_position_limits(weights)
        return weights

    def _get_cluster_weights(self, returns: pd.DataFrame, clusters: np.ndarray) -> Dict[int, float]:
        cluster_variances = {}
        for i in np.unique(clusters):
            cluster_returns = returns.iloc[:, clusters == i].mean(axis=1)
            cluster_variances[i] = np.var(cluster_returns) or 1e-8
            
        total_inv_variance = sum(1/v for v in cluster_variances.values())
        return {i: (1/v)/total_inv_variance for i, v in cluster_variances.items()}

    def _apply_position_limits(self, weights: pd.Series) -> pd.Series:
        weights[weights < self.min_weight] = 0
        weights = weights / weights.sum()
        return weights

    def _calculate_portfolio_metrics(self, returns: pd.DataFrame, weights: pd.Series) -> Dict[str, float]:
        portfolio_returns = (returns * weights).sum(axis=1)
        expected_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'expected_return': expected_return
        }

    def optimize_portfolio(
        self,
        tickers: List[str],
        market_caps: np.ndarray,
        start_date: str,
        end_date: str,
        num_clusters: int = 3,
        risk_measure: str = "variance",
        max_weight: float = 0.10,
        lambda_decay: float = 0.94
    ) -> PortfolioMetrics:
        """Create optimized portfolio based on HRP algorithm."""
        try:
            if len(tickers) != len(market_caps):
                raise ValueError("Number of tickers must match number of market caps")
            
            market_cap_dict = dict(zip(tickers, market_caps))
            total_market_cap = sum(market_caps)
            market_cap_dict = {k: v/total_market_cap for k, v in market_cap_dict.items()}
            
            self.logger.info("Downloading historical data...")
            data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
            
            data = data.dropna(axis=1)
            if data.empty:
                raise ValueError("No valid data after removing NA values")
                
            returns = data.pct_change().dropna()
            
            self.logger.info("Calculating distance matrix...")
            distance_matrix = self._calculate_distance_matrix(
                returns, 
                risk_measure,
                lambda_decay=lambda_decay
            )
            
            self.logger.info("Performing hierarchical clustering...")
            linkage_matrix = linkage(distance_matrix, method='ward')
            clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
            
            available_market_caps = pd.Series({ticker: market_cap_dict[ticker] 
                                             for ticker in data.columns 
                                             if ticker in market_cap_dict})
            
            weights = self._calculate_optimal_weights(returns, clusters, available_market_caps)
            
            # Apply maximum weight constraint
            while weights.max() > max_weight:
                excess = weights[weights > max_weight] - max_weight
                weights[weights > max_weight] = max_weight
                
                remaining_assets = weights[weights < max_weight].index
                if len(remaining_assets) > 0:
                    total_excess = excess.sum()
                    current_remaining_weights = weights[remaining_assets]
                    weights[remaining_assets] += (total_excess * 
                        (current_remaining_weights / current_remaining_weights.sum()))
            
            weights = weights / weights.sum()
            
            metrics = self._calculate_portfolio_metrics(returns, weights)
            
            return PortfolioMetrics(
                weights=weights,
                linkage_matrix=linkage_matrix,
                returns=returns,
                sharpe_ratio=metrics['sharpe_ratio'],
                volatility=metrics['volatility'],
                expected_return=metrics['expected_return'],
                clusters=clusters,
                max_weight=max_weight
            )
            
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {e}")
            raise

    def _calculate_optimal_weights(
        self,
        returns: pd.DataFrame,
        clusters: np.ndarray,
        market_caps: pd.Series
    ) -> pd.Series:
        """Calculate optimal weights using HRP with market cap weighting within clusters."""
        cluster_weights = self._get_cluster_weights(returns, clusters)
        weights = pd.Series(0, index=returns.columns)
        
        for cluster in np.unique(clusters):
            cluster_assets = clusters == cluster
            cluster_market_caps = market_caps[returns.columns[cluster_assets]]
            normalized_caps = cluster_market_caps / cluster_market_caps.sum()
            weights[cluster_assets] = cluster_weights[cluster] * normalized_caps
        
        weights = self._apply_position_limits(weights)
        return weights

    def _get_cluster_weights(self, returns: pd.DataFrame, clusters: np.ndarray) -> Dict[int, float]:
        cluster_variances = {}
        for i in np.unique(clusters):
            cluster_returns = returns.iloc[:, clusters == i].mean(axis=1)
            cluster_variances[i] = np.var(cluster_returns) or 1e-8
            
        total_inv_variance = sum(1/v for v in cluster_variances.values())
        return {i: (1/v)/total_inv_variance for i, v in cluster_variances.items()}

    def _apply_position_limits(self, weights: pd.Series) -> pd.Series:
        weights[weights < self.min_weight] = 0
        weights = weights / weights.sum()
        return weights

    def _calculate_portfolio_metrics(self, returns: pd.DataFrame, weights: pd.Series) -> Dict[str, float]:
        portfolio_returns = (returns * weights).sum(axis=1)
        expected_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'expected_return': expected_return
        }

    @staticmethod
    def _setup_logger() -> logging.Logger:
        logger = logging.getLogger('HRPPortfolio')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger