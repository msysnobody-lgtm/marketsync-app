import pandas as pd
import streamlit as st
import yfinance as yf

@st.cache_data
def load_market_data(period: str) -> pd.DataFrame:
    """
    株価と為替データを取得し、前処理を行って返す関数。
    """
    # ドル円(JPY=X)を追加
    tickers = {
        "^GSPC": "S&P500", 
        "1306.T": "TOPIX(ETF)",
        "JPY=X": "USDJPY" 
    }
    
    # データを一括取得
    data = yf.download(list(tickers.keys()), period=period)["Close"]
    
    # カラム名を変更
    data = data.rename(columns=tickers)
    
    # 欠損値を埋める
    data = data.ffill().dropna()
    
    return data

# process_lag_data は変更なしなのでそのままでOK
def process_lag_data(
    df: pd.DataFrame, lag_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # (中身は前回のまま)
    df_display = df.copy()
    if lag_days > 0:
        df_display["S&P500"] = df_display["S&P500"].shift(lag_days)
        # ドル円もS&P500と同じタイミング（海外時間）とみなしてシフトさせる場合
        df_display["USDJPY"] = df_display["USDJPY"].shift(lag_days)
        df_display = df_display.dropna()

    df_normalized = df_display / df_display.iloc[0] * 100
    return df_display, df_normalized