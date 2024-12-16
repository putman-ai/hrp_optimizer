# Hierarchical Risk Parity Portfolio Optimizer

## Overview
This repository implements a Hierarchical Risk Parity (HRP) portfolio optimization algorithm, as introduced by Marcos Lopez de Prado. HRP provides an alternative to traditional Mean-Variance Optimization by using graph theory and machine learning techniques to construct diversified portfolios without the need for explicit expected return assumptions.

## Key Features
- Implementation of core HRP algorithm with market cap weighting
- Fundamental screening and ranking methodology
- Data preprocessing and handling of financial time series
- Position-level risk management through weight constraints
- Visualization tools for portfolio analysis
- Robust error handling and logging

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start
```python
from src.proc.data_handler import DataHandler
from src.hrp.core import create_hrp_portfolio
from src.viz.plots import plot_portfolio_analysis

# Initialize data handler
handler = DataHandler()

# Load and prepare data
df = handler.load_market_data('data/your_data.csv', min_market_cap=2000)
df = handler.calculate_quality_metrics(df)
hrp_df = handler.prepare_hrp_data(df)

# Create portfolio
weights, linkage_matrix, returns = create_hrp_portfolio(
    tickers=hrp_df['ticker'].tolist(),
    market_caps=hrp_df['market_cap'].tolist(),
    start_date="2023-01-01",
    end_date="2023-12-31".
    num_clusters = 5,
    max_weight = 0.05
)

# Visualize results
plot_portfolio_analysis(returns, weights, linkage_matrix)
```

## Implementation Details

### Fundamental Screening and Ranking
Before applying the HRP algorithm, securities are filtered and ranked based on a composite score incorporating multiple fundamental factors:

#### Quality Metrics
- Cash Return on Invested Capital (CROIC)
- Operating Income Margin
- Asset Turnover

#### Financial Strength
- Net Debt to EBITDA ratio
- Interest Coverage Ratio
- Current Ratio

#### Value Metrics
- Fair Value estimates
- Analyst Price Targets
- Relative Value metrics

Each component is standardized and weighted to create category scores:
1. Quality Score = Equal-weighted average of standardized quality metrics
2. Financial Strength Score = Weighted average (40% Debt/EBITDA, 40% Interest Coverage, 20% Current Ratio)
3. Value Score = Weighted average (40% Fair Value, 40% Analyst Estimates, 20% Price Targets)

The final composite score is an equal-weighted average of these three category scores. Only the top-ranked securities (by default, top 50) are passed to the HRP optimization process.

### Risk Management
The implementation includes position-level risk management through a 5% maximum weight constraint per asset. While this may result in a slightly lower Pseudo-Sharpe ratio compared to an unconstrained portfolio, it provides protection against concentration risk and aligns with common institutional portfolio management practices.

### Data Processing
The data handler includes:
- Market cap filtering
- Quality metrics calculation
- Financial strength assessment
- Missing data handling

### Portfolio Construction
The core algorithm:
1. Calculates correlation matrix from historical returns
2. Performs hierarchical clustering
3. Applies market cap weighting within clusters
4. Implements position limits
5. Generates final portfolio weights

### Performance Metrics
- Returns are annualized (252 trading days)
- Volatility is calculated using sample standard deviation
- Pseudo-Sharpe ratio is provided (note: point-in-time calculation)

## Important Notes

### Data Requirements
The implementation expects the following data points for fundamental screening:
- Market capitalization
- CROIC
- Operating margins
- Asset turnover ratios
- Debt metrics (Net Debt/EBITDA, Interest Coverage)
- Current ratio
- Fair value estimates
- Analyst price targets

### Limitations
- Point-in-time analysis only
- No transaction cost consideration
- Basic risk management implementation
- Limited to single period optimization
- Requires comprehensive fundamental data

### Assumptions
- Market data is sufficiently clean and representative
- Correlations are relatively stable
- Market caps are current and accurate
- Fundamental data is point-in-time accurate

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## References
- López de Prado, Marcos. "Building Diversified Portfolios that Outperform Out of Sample." The Journal of Portfolio Management, 2016.
- Bailey, David H., and Marcos López de Prado. "Balanced baskets: A new approach to trading and hedging risks." Journal of Investment Strategies, 2012.

## Contact
For questions or feedback, please open an issue in the repository.
