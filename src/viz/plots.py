import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import dendrogram

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
    
    plt.tight_layout()
    return fig