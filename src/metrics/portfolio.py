import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PortfolioAnalytics:
    """Container for comprehensive portfolio analytics"""
    returns: pd.Series
    rolling_metrics: pd.DataFrame
    risk_metrics: Dict[str, float]
    distribution_metrics: Dict[str, float]
    risk_adjusted_metrics: Dict[str, float]

class PortfolioMetrics:
    def __init__(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        risk_free_rate: float = 0.02,
        rolling_window: int = 252
    ):
        """
        Initialize portfolio metrics calculator.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Daily returns for each asset
        weights : pd.Series
            Portfolio weights
        risk_free_rate : float
            Annual risk-free rate
        rolling_window : int
            Window size for rolling calculations (default: 252 days)
        """
        self.returns = returns
        self.weights = weights
        self.risk_free_rate = risk_free_rate
        self.rolling_window = rolling_window
        self.daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
        self.portfolio_returns = (returns * weights).sum(axis=1)

    def calculate_drawdown_series(self) -> pd.Series:
        """Calculate the drawdown series for the portfolio."""
        wealth_index = (1 + self.portfolio_returns).cumprod()
        previous_peaks = wealth_index.expanding().max()
        drawdown_series = (wealth_index - previous_peaks) / previous_peaks
        return drawdown_series

    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate various risk metrics for the portfolio."""
        annual_vol = self.portfolio_returns.std() * np.sqrt(252)
        
        # Calculate drawdown series and max drawdown
        drawdown_series = self.calculate_drawdown_series()
        max_drawdown = drawdown_series.min()
        
        # Calculate VaR and CVaR
        var_95 = np.percentile(self.portfolio_returns, 5)
        cvar_95 = self.portfolio_returns[self.portfolio_returns <= var_95].mean()
        
        # Calculate downside deviation (for Sortino ratio)
        negative_returns = self.portfolio_returns[self.portfolio_returns < 0]
        downside_deviation = np.sqrt(252) * np.sqrt(np.mean(negative_returns**2))
        
        return {
            'annual_volatility': annual_vol,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'downside_deviation': downside_deviation
        }

    def calculate_distribution_metrics(self) -> Dict[str, float]:
        """Calculate distribution metrics for the portfolio returns."""
        returns = self.portfolio_returns
        
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        positive_months = (returns.resample('M').last().pct_change() > 0).mean()
        
        # Calculate various quantiles
        quantiles = returns.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'positive_months_pct': positive_months,
            'percentile_5': quantiles[0.05],
            'percentile_95': quantiles[0.95],
            'median': quantiles[0.5]
        }

    def calculate_risk_adjusted_metrics(self) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics."""
        # Annualized return and excess return
        annual_return = self.portfolio_returns.mean() * 252
        excess_return = annual_return - self.risk_free_rate
        
        # Get risk metrics
        risk_metrics = self.calculate_risk_metrics()
        
        # Sharpe Ratio
        sharpe_ratio = excess_return / risk_metrics['annual_volatility']
        
        # Sortino Ratio
        sortino_ratio = excess_return / risk_metrics['downside_deviation']
        
        # Calmar Ratio
        calmar_ratio = -annual_return / risk_metrics['max_drawdown']
        
        return {
            'annual_return': annual_return,
            'excess_return': excess_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }

    def calculate_rolling_metrics(self) -> pd.DataFrame:
        """Calculate rolling performance metrics."""
        rolling_returns = self.portfolio_returns.rolling(window=self.rolling_window)
        
        # Rolling annualized return
        rolling_annual_return = rolling_returns.mean() * 252
        
        # Rolling volatility
        rolling_vol = rolling_returns.std() * np.sqrt(252)
        
        # Rolling Sharpe Ratio
        rolling_excess_return = rolling_annual_return - self.risk_free_rate
        rolling_sharpe = rolling_excess_return / rolling_vol
        
        # Rolling drawdown
        rolling_dd = self.calculate_drawdown_series().rolling(window=self.rolling_window).min()
        
        return pd.DataFrame({
            'rolling_return': rolling_annual_return,
            'rolling_volatility': rolling_vol,
            'rolling_sharpe': rolling_sharpe,
            'rolling_drawdown': rolling_dd
        })

    def calculate_all_metrics(self) -> PortfolioAnalytics:
        """Calculate all portfolio metrics."""
        risk_metrics = self.calculate_risk_metrics()
        distribution_metrics = self.calculate_distribution_metrics()
        risk_adjusted_metrics = self.calculate_risk_adjusted_metrics()
        rolling_metrics = self.calculate_rolling_metrics()
        
        return PortfolioAnalytics(
            returns=self.portfolio_returns,
            rolling_metrics=rolling_metrics,
            risk_metrics=risk_metrics,
            distribution_metrics=distribution_metrics,
            risk_adjusted_metrics=risk_adjusted_metrics
        )

    def perform_factor_analysis(
        self,
        factor_returns: pd.DataFrame
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Perform factor analysis using regression.
        
        Parameters:
        -----------
        factor_returns : pd.DataFrame
            Returns of factor portfolios
            
        Returns:
        --------
        Tuple[pd.Series, pd.DataFrame]
            Factor betas and analysis results
        """
        import statsmodels.api as sm
        
        # Prepare data
        Y = self.portfolio_returns
        X = sm.add_constant(factor_returns)
        
        # Run regression
        model = sm.OLS(Y, X).fit()
        
        # Extract factor betas
        factor_betas = model.params[1:]  # Exclude constant
        
        # Calculate factor contribution
        factor_contribution = factor_returns.multiply(factor_betas, axis=1)
        
        return factor_betas, factor_contribution

    def calculate_risk_contribution(self) -> pd.Series:
        """Calculate risk contribution of each position."""
        # Calculate covariance matrix
        cov_matrix = self.returns.cov() * 252
        
        # Portfolio volatility
        port_vol = np.sqrt(self.weights @ cov_matrix @ self.weights)
        
        # Marginal contribution to risk
        mcr = cov_matrix @ self.weights
        
        # Component contribution to risk
        ccr = self.weights * mcr
        
        # Percentage contribution to risk
        pcr = ccr / port_vol
        
        return pd.Series(pcr, index=self.weights.index)