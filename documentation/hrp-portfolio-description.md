# Hierarchical Risk Parity Portfolio Optimizer with Fundamental Integration

## Project Overview
This project implements an enhanced version of the Hierarchical Risk Parity (HRP) portfolio optimization algorithm, originally introduced by Marcos Lopez de Prado. The implementation extends beyond the basic HRP framework by incorporating fundamental analysis, market capitalization weighting, and sophisticated risk management features.

## Technical Implementation

### Core HRP Algorithm
The HRP algorithm provides an alternative to traditional Mean-Variance Optimization by using hierarchical clustering to construct diversified portfolios without requiring explicit expected return assumptions. This approach helps avoid the well-documented issues with return forecasting and covariance matrix instability that plague traditional optimization methods.

Key technical features include:
- Hierarchical clustering using Ward's linkage method
- Distance matrix calculation with multiple risk measures (variance, MAD, correlation)
- Exponentially weighted covariance implementation for more responsive risk estimation
- Market capitalization integration within the clustering framework
- Position-level risk management through weight constraints

### Fundamental Screening and Ranking
The portfolio construction process begins with a comprehensive fundamental screening model that evaluates securities across three primary dimensions:

1. Quality Metrics
   - Cash Return on Invested Capital (CROIC)
   - Operating Income Margin
   - Asset Turnover

2. Financial Strength
   - Net Debt to EBITDA ratio (40% weight)
   - Interest Coverage Ratio (40% weight)
   - Current Ratio (20% weight)

3. Value Metrics
   - Fair Value estimates (40% weight)
   - Analyst Price Targets (40% weight)
   - Relative Value metrics (20% weight)

Each component is standardized and weighted to create category scores, which are then combined into a final composite score. This fundamental overlay helps ensure that the portfolio tilts toward higher-quality companies while maintaining the risk-based diversification benefits of the HRP algorithm.

### Risk Management Features
The implementation includes several risk management enhancements:
- Maximum position size limits (default 5%)
- Minimum position size thresholds to prevent excessive fragmentation
- Market capitalization-based weighting within clusters
- Robust handling of missing data and outliers
- Comprehensive error checking and validation

## Important Disclosures

### Point-in-Time Analysis
The current implementation uses point-in-time fundamental data, which means that the analysis is based on the most recent available data. This approach does not account for historical changes in fundamental metrics or potential forward-looking biases in analyst estimates.

### Forward-Looking Metrics
The portfolio construction process incorporates forward-looking elements such as analyst price targets and fair value estimates. These projections are inherently uncertain and may not reflect future realized values.

### Pseudo-Sharpe Ratio
The performance metrics include a Pseudo-Sharpe ratio, which is calculated using historical data and should not be interpreted as predictive of future performance. This metric serves as a relative measure of risk-adjusted return potential but does not account for changes in market conditions or fundamental factors.

## Development Process
The project follows a robust development methodology with:
- Comprehensive test coverage for all core components
- Clear separation between data handling, optimization, and visualization
- Detailed documentation and logging
- Version control with separate public and private implementations

## Future Enhancements
Planned improvements include:
- Cross-validation for optimal cluster number determination
- Transaction cost modeling
- More sophisticated turnover constraints
- Enhanced risk factor analysis
- Parallel processing for large-scale optimization

## Technical Requirements
- Python 3.8+
- Key dependencies: NumPy, Pandas, SciPy, yfinance
- Development tools: pytest, logging, duckdb

## References
- López de Prado, Marcos. "Building Diversified Portfolios that Outperform Out of Sample." The Journal of Portfolio Management, 2016.
- Bailey, David H., and Marcos López de Prado. "Balanced baskets: A new approach to trading and hedging risks." Journal of Investment Strategies, 2012.
