# Hierarchical Risk Parity Portfolio Optimizer with Fundamental Integration

## Project Overview
This project implements an enhanced version of the Hierarchical Risk Parity (HRP) portfolio optimization algorithm, extending beyond the basic framework by incorporating fundamental analysis, market capitalization weighting, and sophisticated risk management features. HRP provides an alternative to traditional Mean-Variance Optimization by using hierarchical clustering to construct diversified portfolios without requiring explicit return forecasting.

## Key Features

### Core Implementation
- Hierarchical clustering using Ward's linkage method
- Multiple risk measures (variance, MAD, correlation)
- Exponentially weighted covariance for responsive risk estimation
- Market capitalization integration within clusters
- Position-level risk management with weight constraints
- Comprehensive error handling and validation

### Fundamental Analysis Integration
The portfolio construction begins with a multi-factor screening model evaluating:

1. Quality Metrics: CROIC, Operating Margin, Asset Turnover
2. Financial Strength: Debt/EBITDA, Interest Coverage, Current Ratio
3. Value Metrics: Fair Value estimates, Analyst Targets, Relative Value

Each component is standardized and weighted to create a composite score, tilting the portfolio toward higher-quality companies while maintaining HRP's diversification benefits.

### Risk Management
- Maximum position size limits (5%)
- Market cap-based cluster weighting
- Outlier detection and handling
- Missing data management
- Robust error validation

## Important Disclosures
- Point-in-Time Analysis: Uses current fundamental data, not accounting for historical changes
- Forward-Looking Elements: Incorporates analyst projections and estimates
- Pseudo-Sharpe Ratio: Historical calculation, not predictive of future performance

## Development Framework
- Comprehensive test coverage
- Clear component separation
- Detailed documentation
- Separate public/private implementations
- Python 3.8+ with NumPy, Pandas, SciPy

## References
Based on research by Marcos Lopez de Prado (2016), with significant enhancements for fundamental integration and risk management.
