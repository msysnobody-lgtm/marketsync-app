import sys
from pathlib import Path

# 1. 外部ライブラリのインポート
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- 2. パス解決 (Path Resolution) ---
# appフォルダをモジュールとして認識させるための設定
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- 3. 自作モジュールのインポート ---
# 必ず sys.path.append の後に記述する
from app.services.data_service import load_market_data, process_lag_data
from app.services.ai_service import train_and_predict, run_backtest

# --- UI設定 ---
st.set_page_config(page_title="MarketSync AI", layout="wide")
st.title("🇺🇸S&P500 vs 🇯🇵TOPIX MarketSync AI")

# --- サイドバー (設定・操作) ---
st.sidebar.header("設定")
period_option = st.sidebar.selectbox("期間を選択", ["1y", "2y", "5y", "10y"], index=1)
lag_days = st.sidebar.slider("S&P500のタイムラグ (日)", 0, 5, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI予測")
run_prediction = st.sidebar.button("明日のTOPIXを予測する")

st.sidebar.markdown("---")
st.sidebar.subheader("💰 バックテスト")

# 【新機能】AIの性格調整スライダー
threshold = st.sidebar.slider(
    "AIの強気度 (買い基準)", 
    min_value=0.3, max_value=0.7, value=0.5, step=0.05,
    help="数値を下げると(0.4など)、自信がなくても積極的に買いに行きます。上げると慎重になります。"
)

run_simulation = st.sidebar.button("収益シミュレーション実行")

# --- メインロジック ---
try:
    # データの読み込み
    raw_df = load_market_data(period_option)
    df_display, df_normalized = process_lag_data(raw_df, lag_days)

    # ==========================================
    # 1. AI予測機能 (Prediction)
    # ==========================================
    if run_prediction:
        with st.spinner('AIが市場データを学習中...'):
            result = train_and_predict(raw_df)
            
        st.success("予測完了！")
        col_pred1, col_pred2, col_pred3 = st.columns(3)
        
        with col_pred1:
            st.metric("AIの予測判定", result["prediction"], 
                      delta=f"確信度: {result['probability']:.1%}")
        
        with col_pred2:
            st.metric("学習モデルの精度", f"{result['accuracy']:.1%}")
            
        with col_pred3:
            feat = result["latest_input"]
            st.caption(f"S&P500変化: {feat['S&P500_Chg']:.2%}")
            st.caption(f"USD/JPY変化: {feat['USDJPY_Chg']:.2%}")

        # AIの判断根拠をグラフ表示
        st.markdown("##### 🧠 AIの判断根拠 (重要度)")
        importance_df = pd.DataFrame(
            list(result["importance"].items()), 
            columns=["要因", "重要度"]
        ).set_index("要因")
        
        st.bar_chart(importance_df, horizontal=True)
        st.divider()

    # ==========================================
    # 2. バックテスト機能 (Simulation)
    # ==========================================
    if run_simulation:
        st.subheader("💰 収益シミュレーション結果")
        with st.spinner(f'AI(強気度:{threshold})が過去データでトレード中...'):
            # test_start_date も受け取る
            res_df, ret_ai, ret_market, test_start_date = run_backtest(raw_df, threshold)
            
        # 結果サマリー
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("🤖 AI戦略 (全期間)", f"{ret_ai:+.2f}%", 
                      delta="注: 左側の網掛け部分は学習データです", delta_color="off")
        with col_res2:
            st.metric("🐻 TOPIXガチホ (全期間)", f"{ret_market:+.2f}%")

        # --- チャート描画 ---
        st.subheader("📊 売買タイミング検証")
        fig = go.Figure()
        
        # 1. 市場平均
        fig.add_trace(go.Scatter(
            x=res_df.index, y=res_df["TOPIXガチホ"],
            mode='lines', name='TOPIXガチホ',
            line=dict(color='gray', dash='dot')
        ))
        
        # 2. AI戦略
        fig.add_trace(go.Scatter(
            x=res_df.index, y=res_df["AI戦略"],
            mode='lines', name='AI戦略',
            line=dict(color='red', width=2)
        ))
        
        # 3. 売買ポイント（マーカー）
        buy_signals = res_df[res_df["Position"].diff() == 1]
        sell_signals = res_df[res_df["Position"].diff() == -1]
        
        fig.add_trace(go.Scatter(
            x=buy_signals.index, y=res_df.loc[buy_signals.index]["AI戦略"],
            mode='markers', name='Buy',
            marker=dict(symbol='triangle-up', size=10, color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=sell_signals.index, y=res_df.loc[sell_signals.index]["AI戦略"],
            mode='markers', name='Sell',
            marker=dict(symbol='triangle-down', size=10, color='orange')
        ))
        
        # --- 【追加】学習期間とテスト期間を分ける線 ---
        fig.add_vline(x=test_start_date, line_width=2, line_dash="dash", line_color="green")
        
        # 学習期間（カンニング期間）をグレーで塗りつぶす
        # 注: Plotlyで日付の範囲指定をする際、データの最初の日付が必要です
        min_date = res_df.index.min()
        fig.add_vrect(
            x0=min_date, x1=test_start_date,
            fillcolor="gray", opacity=0.15,
            layer="below", line_width=0,
            annotation_text="学習期間 (Training)", annotation_position="top left"
        )
        
        # テスト期間の注釈
        fig.add_annotation(
            x=test_start_date, y=1.0,
            text="ここから実力 (Testing) →",
            showarrow=True, arrowhead=1, ax=-10, ay=-40
        )

        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"予期せぬエラーが発生しました: {e}")