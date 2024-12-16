import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from typing import List, Dict
from dataclasses import dataclass
import logging

@dataclass
class PortfolioMetrics:
    weights: pd.Series
    linkage_matrix: np.ndarray
    returns: pd.DataFrame
    sharpe_ratio: float
    volatility: float
    expected_return: float
    clusters: np.ndarray

class HRPPortfolio:
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        min_weight: float = 0.01,
        max_weight: float = 0.25
    ):
        self.risk_free_rate = risk_free_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.logger = self._setup_logger()

    def optimize_portfolio(
        self,
        data: pd.DataFrame,
        market_caps: np.ndarray,
        num_clusters: int = 3,
        risk_measure: str = "variance"
    ) -> PortfolioMetrics:
        returns = data.pct_change().dropna()
        
        distance_matrix = self._calculate_distance_matrix(returns, risk_measure)
        self.linkage_matrix = linkage(distance_matrix, method='ward')
        clusters = fcluster(self.linkage_matrix, num_clusters, criterion='maxclust')
        
        weights = self._calculate_optimal_weights(returns, clusters, market_caps)
        metrics = self._calculate_portfolio_metrics(returns, weights)
        
        return PortfolioMetrics(
            weights=weights,
            linkage_matrix=self.linkage_matrix,
            returns=returns,
            sharpe_ratio=metrics['sharpe_ratio'],
            volatility=metrics['volatility'],
            expected_return=metrics['expected_return'],
            clusters=clusters
        )

    def _calculate_distance_matrix(self, returns: pd.DataFrame, risk_measure: str) -> np.ndarray:
        if risk_measure == "variance":
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
        market_caps: np.ndarray
    ) -> pd.Series:
        cluster_weights = self._get_cluster_weights(returns, clusters)
        weights = pd.Series(0, index=returns.columns)
        
        for cluster in np.unique(clusters):
            cluster_assets = clusters == cluster
            cluster_market_caps = market_caps[cluster_assets]
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
        weights[weights > self.max_weight] = self.max_weight
        return weights / weights.sum()

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