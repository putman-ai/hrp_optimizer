# System imports
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Third-party imports
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# Local imports
from src.proc.data_handler import DataHandler
from src.hrp.optimizer import HRPPortfolio
from src.viz.plots import plot_portfolio_analysis
from src.metrics.portfolio import PortfolioMetrics
from src.viz.plot_metrics import (
    plot_performance_metrics, 
    plot_risk_metrics, 
    plot_factor_analysis
)

class PortfolioManager:
    def __init__(self, risk_free_rate: float = 0.02):
        self.data_handler = DataHandler()
        self.portfolio = HRPPortfolio()
        self.risk_free_rate = risk_free_rate
        
    def run_portfolio_optimization(
        self,
        data_path: str,
        min_market_cap: int,
        start_date: str,
        end_date: str,
        num_clusters: int = 3,
        risk_measure: str = "variance",
        n_stocks: int = 50
    ):
        # Load and process data
        df = self.data_handler.load_market_data(data_path, min_market_cap)
        df = self.data_handler.calculate_quality_metrics(df)
        df = self.data_handler.calculate_financial_strength(df)
        df = self.data_handler.calculate_value_metrics(df)
        df = self.data_handler.calculate_composite_score(df)
        
        top_n = df.head(n_stocks)
        hrp_df = pd.DataFrame({
            'ticker': top_n['Full Ticker'].str.split(':').str[1],
            'market_cap': top_n['market_cap']
        })
        
        # Optimize portfolio
        portfolio_metrics = self.portfolio.optimize_portfolio(
            tickers=hrp_df['ticker'].tolist(),
            market_caps=hrp_df['market_cap'].values,
            start_date=start_date,
            end_date=end_date,
            num_clusters=num_clusters,
            risk_measure=risk_measure,
            max_weight=.05
        )
        
        # Calculate portfolio analytics
        metrics = PortfolioMetrics(
            returns=portfolio_metrics.returns,
            weights=portfolio_metrics.weights,
            risk_free_rate=self.risk_free_rate
        )
        analytics = metrics.calculate_all_metrics()
        
        # Download factor data and calculate factor analysis
        factor_tickers = ['SPY', 'IWM', 'VTV']  # Market, Size, Value factors
        factor_data = yf.download(factor_tickers, start=start_date, end=end_date)['Adj Close']
        factor_returns = factor_data.pct_change().dropna()
        factor_returns.columns = ['Market', 'Size', 'Value']
        
        factor_betas, factor_contribution = metrics.perform_factor_analysis(factor_returns)
        risk_contribution = metrics.calculate_risk_contribution()
        
        # Generate results DataFrame
        results = pd.DataFrame({
            'Weight': portfolio_metrics.weights,
            'Cluster': portfolio_metrics.clusters,
            'Risk_Contribution': risk_contribution
        })
        results['Market_Cap'] = hrp_df.set_index('ticker')['market_cap']
        
        # Create visualizations
        # Portfolio composition plot
        fig1 = plot_portfolio_analysis(
            portfolio_metrics.returns,
            portfolio_metrics.weights,
            portfolio_metrics.linkage_matrix
        )
        plt.show()
        
        # Performance metrics
        fig2 = plot_performance_metrics(analytics)
        plt.show()
        
        # Risk metrics
        fig3 = plot_risk_metrics(analytics)
        plt.show()
        
        # Factor analysis and additional metrics
        fig4 = plot_factor_analysis(factor_betas, risk_contribution, analytics)
        plt.show()
        
        # Print summary metrics and factor analysis
        print("\nPortfolio Risk Metrics:")
        for metric, value in analytics.risk_metrics.items():
            print(f"{metric}: {value:.4f}")
            
        print("\nRisk-Adjusted Metrics:")
        for metric, value in analytics.risk_adjusted_metrics.items():
            print(f"{metric}: {value:.4f}")
            
        print("\nFactor Exposures:")
        for factor, beta in factor_betas.items():
            print(f"{factor}: {beta:.3f}")
        
        return results, analytics, factor_betas, risk_contribution

if __name__ == "__main__":
    manager = PortfolioManager(risk_free_rate=0.02)
    results, analytics, factor_betas, risk_contribution = manager.run_portfolio_optimization(
        data_path='data/sp500_data.csv',
        min_market_cap=2000,
        start_date="2019-05-01",
        end_date="2024-12-10"
    )
    
    if results is not None:
        print("\nTop Portfolio Positions:")
        print(results.sort_values('Weight', ascending=False).head(10))