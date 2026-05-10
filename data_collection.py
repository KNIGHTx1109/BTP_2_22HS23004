import yfinance as yf
import pandas as pd
import numpy as np
import os

def download_and_preprocess_data(start_date="2019-01-02", end_date="2021-12-31"):
    """
    Downloads historical adjusted closing prices for specified indices,
    handles non-trading days via forward-filling, and calculates log returns.
    """
    tickers = {
        'Sensex': '^BSESN',
        'S&P 500': '^GSPC',
        'Nikkei 225': '^N225',
        'DAX': '^GDAXI',
        'SSE Composite': '000001.SS'
    }
    
    print("Downloading data...")
    raw_data = {}
    for name, ticker in tickers.items():
        df = yf.download(ticker, start=start_date, end=end_date)
        
        # Handle MultiIndex columns (yfinance >= 0.2.40)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        if 'Adj Close' in df.columns and not df['Adj Close'].isna().all():
            raw_data[name] = df['Adj Close']
        else:
            raw_data[name] = df['Close']
    
    print("Merging and handling non-trading days...")
    # Combine all series into a single DataFrame using an outer join (default for pd.concat on axis=1)
    combined_df = pd.concat(raw_data, axis=1)
    
    # Drop rows where ALL values are NaN (e.g., global weekends)
    combined_df.dropna(how='all', inplace=True)
    
    # Forward-fill to handle non-trading days for individual markets
    combined_df.ffill(inplace=True)
    
    # Drop any remaining NaNs at the very beginning
    combined_df.dropna(inplace=True)
    
    print("Calculating log returns...")
    # Calculate continuous daily log returns
    log_returns = np.log(combined_df / combined_df.shift(1))
    
    # Drop the first row (which is NaN after taking returns)
    log_returns.dropna(inplace=True)
    
    # Create outputs directory
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    combined_df.to_csv('data/adjusted_close_prices.csv')
    log_returns.to_csv('data/log_returns.csv')
    
    print(f"Data processing complete. {len(log_returns)} observations saved.")
    return combined_df, log_returns

if __name__ == "__main__":
    download_and_preprocess_data()
