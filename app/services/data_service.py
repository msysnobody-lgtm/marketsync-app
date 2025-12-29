import pandas as pd
import streamlit as st
import yfinance as yf

@st.cache_data(ttl=3600) # 1時間ごとにキャッシュクリア
def load_market_data(period: str) -> pd.DataFrame:
    """
    株価データを取得し、前処理を行って返す関数。
    """
    # 取得するティッカーリスト
    tickers = {
        "^GSPC": "S&P500", 
        "1306.T": "TOPIX(ETF)",
        "JPY=X": "USDJPY" 
    }
    
    # yfinanceでデータを取得
    # group_by='column' は新しいバージョンでの挙動を安定させるため
    # auto_adjust=True で分割調整などを自動で行う
    data = yf.download(list(tickers.keys()), period=period, auto_adjust=True)
    
    # データ構造の確認と修正
    # yfinanceのバージョンによってMultiIndex('Close', '^GSPC')になる場合とならない場合があるため調整
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            data = data['Close']
    elif 'Close' in data.columns:
        data = data['Close']
        
    # カラム名をリネーム (存在するものだけ)
    # ダウンロードに失敗した銘柄がある場合のエラー回避
    available_cols = set(data.columns)
    rename_map = {k: v for k, v in tickers.items() if k in available_cols}
    data = data.rename(columns=rename_map)
    
    # 欠損値処理の強化 (ここが修正の肝！)
    # 1. まず前方の値で埋める（休日対策）
    data = data.ffill()
    
    # 2. それでも埋まらない（開始時点のズレなど）場合、後方の値で埋める
    data = data.bfill()
    
    # 3. 最後にどうしてもNaNが残る行だけ消す
    data = data.dropna()
    
    # データが空っぽになってしまった場合の緊急避難
    if data.empty:
        # エラーを出さずに、ダミーデータを返すか、例外を投げてUI側でキャッチさせる
        # ここではエラーメッセージを表示させるために例外を投げる
        raise ValueError("データの取得に失敗しました（データが空です）。RenderのIP制限またはYahoo Financeの一時的な不具合の可能性があります。")

    return data

def process_lag_data(
    df: pd.DataFrame, lag_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    タイムラグ処理と正規化処理を行う
    """
    df_display = df.copy()

    # S&P500のデータを指定した日数分ずらす
    if lag_days > 0 and "S&P500" in df_display.columns:
        df_display["S&P500"] = df_display["S&P500"].shift(lag_days)
        
        # USDJPYも合わせてずらす
        if "USDJPY" in df_display.columns:
            df_display["USDJPY"] = df_display["USDJPY"].shift(lag_days)
            
        df_display = df_display.dropna()

    if df_display.empty:
         raise ValueError("ラグ処理後のデータが空になりました。期間を長くするか、ラグを短くしてください。")

    # 正規化（開始日=100）
    df_normalized = df_display / df_display.iloc[0] * 100

    return df_display, df_normalized