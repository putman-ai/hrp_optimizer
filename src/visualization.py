import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import dendrogram

def plot_portfolio_analysis(returns, weights, linkage_matrix, figsize=(15, 15)):
    """
    Creates visualization of the portfolio analysis
    """
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    corr_matrix = returns.corr()
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0, 
                fmt='.2f',
                ax=ax1)
    ax1.set_title('Asset Correlation Matrix')
    
    ax2 = fig.add_subplot(gs[0, 1])
    dendrogram(linkage_matrix, 
               labels=returns.columns,
               leaf_rotation=90)
    ax2.set_title('Hierarchical Clustering Dendrogram')
    
    ax3 = fig.add_subplot(gs[1, :])
    weights.sort_values(ascending=True).plot(kind='barh', ax=ax3)
    ax3.set_title('Portfolio Weights')
    ax3.set_xlabel('Weight')
    
    portfolio_return = (returns * weights).sum(axis=1)
    annual_return = portfolio_return.mean() * 252
    annual_vol = portfolio_return.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol
    
    metrics_text = f'Annual Return: {annual_return:.2%}\n'
    metrics_text += f'Annual Volatility: {annual_vol:.2%}\n'
    metrics_text += f'Sharpe Ratio: {sharpe:.2f}'
    
    plt.figtext(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig