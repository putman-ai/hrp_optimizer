from src.proc.data_handler import DataHandler
from src.hrp.optimizer import HRPPortfolio
from src.viz.plots import plot_portfolio_analysis
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

class PortfolioManager:
    def __init__(self, risk_free_rate: float = 0.02):
        self.data_handler = DataHandler()
        self.portfolio = HRPPortfolio(risk_free_rate=risk_free_rate)
        
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
        
        portfolio_metrics = self.portfolio.optimize_portfolio(
            tickers=hrp_df['ticker'].tolist(),
            market_caps=hrp_df['market_cap'].values,
            start_date=start_date,
            end_date=end_date,
            num_clusters=num_clusters,
            risk_measure=risk_measure,
            max_weight = .05
        )
        
        results = pd.DataFrame({
            'Weight': portfolio_metrics.weights,
            'Cluster': portfolio_metrics.clusters
        })
        results['Market_Cap'] = hrp_df.set_index('ticker')['market_cap']
        
        fig = plot_portfolio_analysis(
            portfolio_metrics.returns,
            portfolio_metrics.weights,
            portfolio_metrics.linkage_matrix
        )
        plt.show()
        
        return results

if __name__ == "__main__":
    manager = PortfolioManager(risk_free_rate=0.02)
    results = manager.run_portfolio_optimization(
        data_path='data/sp500_data.csv',
        min_market_cap=2000,
        start_date="2019-05-01",
        end_date="2024-12-10"
    )
    
    if results is not None:
        print("\nPortfolio Summary:")
        print(results.sort_values('Weight', ascending=False).head(10))