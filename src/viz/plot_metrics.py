import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.metrics.portfolio import PortfolioAnalytics

def plot_performance_metrics(analytics: PortfolioAnalytics, figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    analytics.rolling_metrics[['rolling_return', 'rolling_volatility']].plot(ax=ax)
    ax.set_title('Rolling Annual Performance')
    ax.set_ylabel('Percentage')
    plt.tight_layout()
    return fig

def plot_risk_metrics(analytics: PortfolioAnalytics, figsize=(12, 6)):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    risk_data = pd.Series(analytics.risk_metrics)
    risk_data.plot(kind='bar', ax=ax1)
    ax1.set_title('Risk Metrics')
    ax1.tick_params(axis='x', rotation=45)
    
    dist_data = pd.Series(analytics.distribution_metrics)
    dist_data.plot(kind='bar', ax=ax2)
    ax2.set_title('Distribution Metrics')
    ax2.tick_params(axis='x', rotation=45)
    
    risk_adj_data = pd.Series(analytics.risk_adjusted_metrics)
    risk_adj_data.plot(kind='bar', ax=ax3)
    ax3.set_title('Risk-Adjusted Metrics')
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def plot_factor_analysis(factor_betas, risk_contribution, analytics: PortfolioAnalytics, figsize=(15, 5)):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Factor exposures
    factor_betas.plot(kind='bar', ax=ax1)
    ax1.set_title('Factor Exposures')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.tick_params(axis='x', rotation=45)
    
    # Risk contribution
    risk_contribution.sort_values(ascending=True).tail(10).plot(kind='barh', ax=ax2)
    ax2.set_title('Top 10 Risk Contributors')
    
    # Cumulative performance
    cumulative_return = (1 + analytics.returns).cumprod()
    cumulative_return.plot(ax=ax3)
    ax3.set_title('Cumulative Portfolio Return')
    ax3.set_ylabel('Growth of $1')
    
    plt.tight_layout()
    return fig