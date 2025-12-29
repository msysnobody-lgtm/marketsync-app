import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def get_features_and_target(raw_df: pd.DataFrame):
    df = raw_df.copy()
    df["Returns"] = df["TOPIX(ETF)"].pct_change()
    df["S&P_Returns"] = df["S&P500"].pct_change()
    df["S&P_Trend"] = df["S&P500"].rolling(5).mean()
    df["Volatility"] = df["TOPIX(ETF)"].rolling(5).std()
    
    # 目的変数: 翌日のTOPIXが上がるか(1)下がるか(0)
    df["Target"] = (df["TOPIX(ETF)"].shift(-1) > df["TOPIX(ETF)"]).astype(int)
    
    df.dropna(inplace=True)
    
    feature_cols = ["Returns", "S&P_Returns", "S&P_Trend", "Volatility"]
    X = df[feature_cols]
    y = df["Target"]
    
    return X, y, df, feature_cols

def train_and_predict(raw_df: pd.DataFrame):
    X, y, _, _ = get_features_and_target(raw_df)
    
    model = RandomForestClassifier(n_estimators=100, min_samples_split=5, random_state=42)
    model.fit(X, y)
    
    latest_data = X.iloc[[-1]]
    prob = model.predict_proba(latest_data)[0][1]
    prediction = "上昇" if prob >= 0.5 else "下落"
    
    # 精度確認（全期間）
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    importance = dict(zip(X.columns, model.feature_importances_))
    
    return {
        "prediction": prediction,
        "probability": prob,
        "accuracy": accuracy,
        "importance": importance
    }

def run_backtest(raw_df: pd.DataFrame, threshold: float = 0.5):
    X, y, data, feature_cols = get_features_and_target(raw_df)
    
    # 1. データを分割 (80%学習 / 20%テスト)
    split_point = int(len(X) * 0.8)
    
    X_train = X.iloc[:split_point]
    y_train = y.iloc[:split_point]
    
    X_test = X.iloc[split_point:]
    # y_test = y.iloc[split_point:] # テスト用正解ラベル（今回はリターン計算に使うので直接は不要）
    
    # 2. モデル学習 (過去データのみで学習)
    model = RandomForestClassifier(n_estimators=100, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    
    # 3. 予測とリターン計算 (テスト期間のみ！)
    probs = model.predict_proba(X_test)[:, 1]
    predictions = (probs >= threshold).astype(int)
    
    # テスト期間の実際の市場リターンを取得
    returns_col = data["TOPIX(ETF)"].pct_change().shift(-1)
    actual_returns = returns_col.loc[X_test.index]
    
    # 4. 資産推移シミュレーション (ここからスタート)
    strategy_assets = [1.0]
    market_assets = [1.0]
    current_strategy = 1.0
    current_market = 1.0
    positions = []
    
    for pred, ret in zip(predictions, actual_returns):
        positions.append(pred)
        if pred == 1:
            current_strategy *= (1 + ret)
        else:
            current_strategy *= 1.0 # 現金ポジション
            
        current_market *= (1 + ret)
        strategy_assets.append(current_strategy)
        market_assets.append(current_market)
        
    results_df = pd.DataFrame({
        "Date": actual_returns.index,
        "AI戦略": strategy_assets[1:],
        "TOPIXガチホ": market_assets[1:],
        "Position": positions
    }).set_index("Date")
    
    final_return_ai = (current_strategy - 1) * 100
    final_return_market = (current_market - 1) * 100
    
    # テスト期間のみなので、日付データは不要
    return results_df, final_return_ai, final_return_market