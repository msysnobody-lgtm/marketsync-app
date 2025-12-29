import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier # ← RFに戻す
from sklearn.metrics import accuracy_score

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    
    # 1. 基本指標
    data["S&P500_Chg"] = data["S&P500"].pct_change()
    data["USDJPY_Chg"] = data["USDJPY"].pct_change()
    
    # 2. トレンド・加熱感
    data["S&P500_MA5"] = data["S&P500"] / data["S&P500"].rolling(window=5).mean() - 1
    data["Volatility"] = data["TOPIX(ETF)"].pct_change().rolling(window=5).std()
    
    # RSI
    diff = data["TOPIX(ETF)"].diff()
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    avg_gain = up.rolling(window=14).mean()
    avg_loss = down.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # 3. カレンダー
    data["Weekday"] = data.index.dayofweek
    data["Is_Month_End"] = data.index.is_month_end.astype(int)
    
    # 4. 短期記憶 (ラグ)
    data["S&P500_Lag1"] = data["S&P500_Chg"].shift(1)
    data["S&P500_Lag2"] = data["S&P500_Chg"].shift(2)
    data["S&P500_Lag3"] = data["S&P500_Chg"].shift(3)

    return data.dropna()

def get_features_and_target(df: pd.DataFrame):
    data = add_technical_indicators(df)
    
    feature_cols = [
        "S&P500_Chg", "USDJPY_Chg", 
        "S&P500_MA5", "Volatility", "RSI",
        "Weekday", "Is_Month_End",
        "S&P500_Lag1", "S&P500_Lag2", "S&P500_Lag3"
    ]
    
    X = data[feature_cols]
    target_returns = data["TOPIX(ETF)"].pct_change().shift(-1)
    y = (target_returns > 0).astype(int)
    
    valid_indices = X.index.intersection(y.index).intersection(target_returns.dropna().index)
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]
    
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    
    return X, y, data, feature_cols

def train_and_predict(raw_df: pd.DataFrame):
    X, y, data, feature_cols = get_features_and_target(raw_df)
    
    # ランダムフォレスト (n_estimators=100)
    model = RandomForestClassifier(n_estimators=100, min_samples_split=5, random_state=42)
    model.fit(X, y)
    
    accuracy = model.score(X, y)

    latest_data = add_technical_indicators(raw_df)
    latest_features = latest_data[feature_cols].iloc[[-1]]
    
    prediction = model.predict(latest_features)[0]
    probs = model.predict_proba(latest_features)[0]
    confidence = probs[prediction]

    importance_dict = dict(zip(feature_cols, model.feature_importances_))

    return {
        "prediction": "上昇" if prediction == 1 else "下落",
        "probability": confidence,
        "accuracy": accuracy,
        "latest_input": latest_features.iloc[0].to_dict(),
        "importance": importance_dict
    }

def run_backtest(raw_df: pd.DataFrame, threshold: float = 0.5):
    X, y, data, feature_cols = get_features_and_target(raw_df)
    
    # 1. 学習用(前半80%) と テスト用(後半20%) に分ける
    # ※ AIの学習自体は、未来を見ないように「過去データのみ」で行います
    split_point = int(len(X) * 0.8)
    X_train = X.iloc[:split_point]
    y_train = y.iloc[:split_point]
    
    # モデル学習 (前半のデータだけで賢くなる)
    model = RandomForestClassifier(n_estimators=100, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    
    # --- 【ここが変更点】 ---
    # 2. 予測は「全期間 (X全体)」に対して行う
    # これにより、グラフが「期間設定」と同じ長さになります
    probs = model.predict_proba(X)[:, 1]
    predictions = (probs >= threshold).astype(int)
    
    # リターン計算も全期間で行う
    returns_col = data["TOPIX(ETF)"].pct_change().shift(-1)
    actual_returns = returns_col.loc[X.index] # X_testではなくX全体
    
    # 資産推移シミュレーション
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
            current_strategy *= 1.0
            
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
    
    return results_df, final_return_ai, final_return_market