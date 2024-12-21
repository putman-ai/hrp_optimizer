import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Tuple

from scipy.cluster.hierarchy import dendrogram
from scipy import stats
from src.metrics.portfolio import PortfolioAnalytics

def plot_portfolio_analysis(returns, weights, linkage_matrix, figsize=(12, 12)):
    """
    Creates visualization of the portfolio analysis
    """
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 2)
    
    # Correlation matrix heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    corr_matrix = returns.corr()
    sns.heatmap(corr_matrix, 
                annot=False,
                cmap='coolwarm', 
                center=0,
                ax=ax1)
    ax1.set_title('Asset Correlation Matrix')
    
    ax2 = fig.add_subplot(gs[0, 1])
    dendrogram(linkage_matrix, 
               labels=returns.columns,
               leaf_rotation=90)
    ax2.set_title('Hierarchical Clustering Dendrogram')
    
    # Filter out zero weights before plotting
    nonzero_weights = weights[weights > 0].sort_values(ascending=True)
    
    ax3 = fig.add_subplot(gs[1, :])
    nonzero_weights.plot(kind='barh', ax=ax3)
    ax3.set_title('Portfolio Weights (Non-Zero Positions)')
    ax3.set_xlabel('Weight')
    
    portfolio_return = (returns * weights).sum(axis=1)
    annual_return = portfolio_return.mean() * 252
    annual_vol = portfolio_return.std() * np.sqrt(252)
    pseudo_sharpe = annual_return / annual_vol
    
    metrics_text = f'Annual Return: {annual_return:.2%}\n'
    metrics_text += f'Annual Volatility: {annual_vol:.2%}\n'
    metrics_text += f'Pseudo Sharpe: {pseudo_sharpe:.2f}'
    
    # Changed position to lower right (0.75, 0.02)
    plt.figtext(0.75, 0.02, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_enhanced_metrics(portfolio_analytics: PortfolioAnalytics) -> Tuple[plt.Figure, plt.Figure]:
    """
    Create enhanced visualizations of portfolio metrics.
    
    Parameters:
    -----------
    portfolio_analytics : PortfolioAnalytics
        Portfolio analytics from PortfolioMetrics calculator
        
    Returns:
    --------
    Tuple[plt.Figure, plt.Figure]
        Figures containing rolling metrics and return distribution plots
    """
    # Rolling metrics plot
    fig_rolling, axes1 = plt.subplots(2, 2, figsize=(15, 10))
    fig_rolling.suptitle('Rolling Portfolio Metrics', fontsize=14)
    
    portfolio_analytics.rolling_metrics['rolling_return'].plot(ax=axes1[0,0])
    axes1[0,0].set_title('Rolling Annual Returns')
    axes1[0,0].grid(True)
    
    portfolio_analytics.rolling_metrics['rolling_volatility'].plot(ax=axes1[0,1])
    axes1[0,1].set_title('Rolling Volatility')
    axes1[0,1].grid(True)
    
    portfolio_analytics.rolling_metrics['rolling_sharpe'].plot(ax=axes1[1,0])
    axes1[1,0].set_title('Rolling Sharpe Ratio')
    axes1[1,0].grid(True)
    
    portfolio_analytics.rolling_metrics['rolling_drawdown'].plot(ax=axes1[1,1])
    axes1[1,1].set_title('Rolling Drawdown')
    axes1[1,1].grid(True)
    
    # Distribution analysis plot
    fig_dist, axes2 = plt.subplots(2, 2, figsize=(15, 10))
    fig_dist.suptitle('Portfolio Return Distribution Analysis', fontsize=14)
    
    sns.histplot(portfolio_analytics.returns, kde=True, ax=axes2[0,0])
    axes2[0,0].set_title('Return Distribution')
    
    stats.probplot(portfolio_analytics.returns, dist="norm", plot=axes2[0,1])
    axes2[0,1].set_title('Q-Q Plot')
    
    drawdown_series = portfolio_analytics.rolling_metrics['rolling_drawdown']
    drawdown_series.plot(ax=axes2[1,0])
    axes2[1,0].set_title('Drawdown')
    axes2[1,0].grid(True)
    
    returns_series = portfolio_analytics.returns
    lag_returns = returns_series.shift(1)
    axes2[1,1].scatter(lag_returns, returns_series, alpha=0.5)
    axes2[1,1].set_title('Return Scatter (t vs t-1)')
    axes2[1,1].set_xlabel('Previous Return')
    axes2[1,1].set_ylabel('Current Return')
    axes2[1,1].grid(True)
    
    plt.tight_layout()
    return fig_rolling, fig_dist