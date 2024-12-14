from src.data_handler import DataHandler
from src.hrp_core import create_hrp_portfolio
from src.visualization import plot_portfolio_analysis
import matplotlib.pyplot as plt

if __name__ == "__main__":
    handler = DataHandler()
    
    df = handler.load_market_data('data/sp500_data.csv', 2000) # data source Finbox and/or EODHistorical
    df = handler.calculate_quality_metrics(df)
    hrp_df = handler.prepare_hrp_data(df)
    
    try:
        weights, linkage_matrix, returns = create_hrp_portfolio(
            tickers=hrp_df['ticker'].tolist(),
            market_caps=hrp_df['market_cap'].tolist(),
            start_date="2019-05-01",
            end_date="2024-12-10"
        )
        
        print("\nFinal Portfolio Allocation:")
        print(weights.round(4))
        
        fig = plot_portfolio_analysis(returns, weights, linkage_matrix)
        plt.show()
        
    except Exception as e:
        print(f"Failed to create portfolio: {e}")