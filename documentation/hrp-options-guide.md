# HRP Portfolio Optimization Configuration Guide

## Risk Measures

### Available Options
- `variance`: Standard correlation-based distance (default)
- `mad`: Mean Absolute Deviation-based distance
- `correlation`: Direct correlation distance
- `ewma`: Exponentially weighted correlation distance (planned)

### When to Use Each

#### Variance (Default)
- Best for: Normal market conditions, well-behaved return distributions
- Advantages: 
  - Well-understood, standard approach
  - Computationally efficient
  - Good for long-term strategic allocation
- Disadvantages:
  - Sensitive to outliers
  - Assumes normal distribution
  - May underestimate tail risks

#### MAD
- Best for: Portfolios with frequent outliers or high kurtosis
- Advantages:
  - More robust to outliers
  - No distribution assumptions
  - Better for heavy-tailed assets
- Disadvantages:
  - More computationally intensive
  - May overweight defensive assets
  - Less efficient in normal market conditions

#### Direct Correlation
- Best for: Focus on pure diversification without risk scaling
- Advantages:
  - Clearer diversification focus
  - Scale-independent
  - Good for assets with similar volatilities
- Disadvantages:
  - Ignores magnitude of risks
  - May lead to suboptimal risk allocation
  - Not suitable for assets with very different risk profiles

#### EWMA (Planned)
- Best for: Tactical allocation, regime-aware portfolios
- Advantages:
  - More responsive to recent market conditions
  - Better adapts to changing correlations
  - Good for tactical allocation
- Disadvantages:
  - More parameters to tune
  - May lead to higher turnover
  - Requires careful decay factor selection

## Clustering Methods

### Available Options
- `ward`: Minimum variance clustering (default)
- `complete`: Maximum distance clustering (planned)
- `average`: Average distance clustering (planned)
- `single`: Minimum distance clustering (planned)

### Impact on Portfolio Construction
- `ward`: Tends to create balanced clusters, good for risk parity
- `complete`: Creates more distinct clusters, good for strong differentiation
- `average`: Balanced approach, good for stable hierarchies
- `single`: Sensitive to outliers, good for identifying unique assets

### Cluster Number Determination
- Fixed: User-specified number (current)
- Dynamic: Based on gap statistic (planned)
- Adaptive: Based on risk contribution (planned)

## Weight Calculation Methods

### Within Clusters
- Market cap weighted (current)
- Risk parity (planned)
- Minimum variance (planned)
- Hybrid approaches (planned)

### Cross-Cluster
- Inverse variance (current)
- Equal risk contribution (planned)
- CVaR optimization (planned)

## Implementation Examples

```python
# Basic usage with defaults
portfolio = HRPPortfolio()
results = portfolio.optimize_portfolio(
    tickers=tickers,
    market_caps=market_caps,
    start_date=start_date,
    end_date=end_date
)

# Advanced configuration
portfolio = HRPPortfolio(
    risk_free_rate=0.02,
    min_weight=0.01,
    max_weight=0.10
)
results = portfolio.optimize_portfolio(
    tickers=tickers,
    market_caps=market_caps,
    start_date=start_date,
    end_date=end_date,
    num_clusters=5,
    risk_measure="mad",
    clustering_method="ward"
)
```

## Performance Implications

### Processing Time
- Variance: Base case (1x)
- MAD: ~1.5x slower
- EWMA: ~1.2x slower
- Dynamic clustering: ~2x slower

### Memory Usage
- Scales with O(n²) for n assets
- EWMA requires additional storage for time series
- Dynamic clustering requires additional temporary storage

## Best Practices

1. Start with default settings for initial analysis
2. Use MAD for portfolios with known outlier issues
3. Consider EWMA for tactical portfolios with shorter horizons
4. Use dynamic clustering for new portfolios without strong priors
5. Monitor turnover when using more responsive methods

## Planned Enhancements

1. Exponentially weighted correlations
2. Dynamic cluster number determination
3. Enhanced risk parity within clusters
4. CVaR optimization
5. Alternative clustering methods

## Warning Signs

Watch for these indicators that configuration changes may be needed:
- High concentration in specific clusters
- Rapid weight changes between rebalances
- Excessive sensitivity to parameter changes
- Unexpectedly high turnover
