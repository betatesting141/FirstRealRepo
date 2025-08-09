# Save as sma_roc_correlation.py and run: python sma_roc_correlation.py
# Requires: pip install yfinance pandas numpy scipy

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from datetime import datetime, timedelta

TICKERS = ["AMZN","TSLA","PLTR","NVDA","KTOS","AAPL","SPY"]
END = datetime(2025, 8, 9)   # set to the date you want as 'today'
START = END - timedelta(days=140)  # ~90 trading days; adjust if you want more/less
LOOKBACKS = [10, 20, 50]

def fetch_close(ticker):
    df = yf.download(ticker, start=START.strftime("%Y-%m-%d"), end=(END + timedelta(days=1)).strftime("%Y-%m-%d"), progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    return df['Close'].rename(ticker)

def sma(series, n):
    return series.rolling(window=n, min_periods=n).mean()

def roc(series, n):
    return 100.0 * (series - series.shift(n)) / series.shift(n)

def sma_slope(series, n):
    # slope defined as SMA_t - SMA_{t-n}
    s = sma(series, n)
    return s - s.shift(n)

def per_ticker_correlation(close, n):
    # produce aligned vectors for correlation
    s_slope = sma_slope(close, n)
    r = roc(close, n)
    df = pd.concat([s_slope, r], axis=1).dropna()
    if df.shape[0] < 10:
        return np.nan, df.shape[0]
    corr, p = pearsonr(df.iloc[:,0], df.iloc[:,1])
    return corr, df.shape[0]

def run():
    results = []
    # download all tickers into one DataFrame (aligned by date)
    all_close = []
    for t in TICKERS:
        try:
            c = fetch_close(t)
            all_close.append(c)
        except Exception as e:
            print("Warning:", e)
    closes = pd.concat(all_close, axis=1).dropna(how='all')
    for t in closes.columns:
        for n in LOOKBACKS:
            corr, rows = per_ticker_correlation(closes[t], n)
            results.append({
                "ticker": t,
                "lookback": n,
                "pearson_r": None if np.isnan(corr) else round(corr, 4),
                "data_points": rows
            })
    res_df = pd.DataFrame(results).pivot(index='ticker', columns='lookback', values='pearson_r')
    print("\nPearson correlation (SMA_slope vs ROC)\n(lookbacks are columns: 10, 20, 50)\n")
    print(res_df)
    # summary stats
    print("\nSummary (mean absolute correlation by lookback):")
    for n in LOOKBACKS:
        mean_abs = np.nanmean([abs(x) for x in res_df[n].values.astype(float)])
        print(f" {n}-day: mean |r| = {mean_abs:.3f}")
    print("\nNotes:")
    print("- r near 0.6-0.8 = moderate-strong positive relation (in trending regimes).")
    print("- r below ~0.3 = weak relation (likely choppy/range market).")
    print("- Use ROC as an early alert (it reacts faster). Use SMA slope as trend confirmation.")
    print("\nIf you'd like, save the results to CSV by uncommenting the next line:")
    # res_df.to_csv("sma_roc_correlations.csv")

if __name__ == '__main__':
    run()
