import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from services.data_service import load_market_data, process_lag_data
from services.ai_service import train_and_predict, run_backtest

# --- ページ設定 ---
st.set_page_config(page_title="MarketSync AI", layout="wide")

st.title("🇺🇸S&P500 vs 🇯🇵TOPIX MarketSync AI")
st.markdown("米国市場(S&P500)の動きから、翌日の日本市場(TOPIX)を予測するAI")

# --- サイドバー設定 ---
st.sidebar.header("設定")
selected_period = st.sidebar.selectbox("データ期間", ["1y", "2y", "5y", "10y"], index=2)
st.sidebar.caption("※5yを選択すると、直近1年間の成績が表示されます(80:20分割)")
threshold = st.sidebar.slider("AIの強気度判定(しきい値)", 0.4, 0.6, 0.40, 0.01)
run_simulation = st.sidebar.checkbox("収益シミュレーションを実行", value=True)

# --- メイン処理 ---
try:
    # 1. データ取得
    with st.spinner('市場データを取得中...'):
        raw_df = load_market_data(selected_period)
    
    # 直近データの表示
    latest_date = raw_df.index[-1].strftime('%Y-%m-%d')
    st.info(f"データ取得日: {latest_date} (直近の終値データを使用)")

    # 元データのトレンド比較グラフ
    st.subheader("📈 日米株価トレンド比較 (正規化済み)")
    norm_df = raw_df / raw_df.iloc[0]
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(x=norm_df.index, y=norm_df["S&P500"], name="🇺🇸 S&P500", line=dict(color='blue')))
    fig_raw.add_trace(go.Scatter(x=norm_df.index, y=norm_df["TOPIX(ETF)"], name="🇯🇵 TOPIX", line=dict(color='orange')))
    fig_raw.update_layout(height=400, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_raw, use_container_width=True)

    # 2. AI予測
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 AIの予測判定")
        with st.spinner('AIが思考中...'):
            ai_result = train_and_predict(raw_df)
        
        prediction_text = ai_result["prediction"]
        probability = ai_result["probability"]
        
        if prediction_text == "上昇":
            st.success(f"## {prediction_text} 📈")
        else:
            st.error(f"## {prediction_text} 📉")
            
        display_prob = probability if probability >= 0.5 else 1 - probability
        st.write(f"確信度: **{display_prob:.1%}**")
        st.caption(f"モデル精度(Accuracy): {ai_result['accuracy']:.1%}")

    with col2:
        st.subheader("🔑 注目している指標")
        importance = ai_result["importance"]
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        st.json(dict(sorted_importance))

    st.markdown("---")

    # 3. バックテスト結果
    if run_simulation:
        st.subheader("💰 実力テスト結果 (直近20%の期間)")
        with st.spinner(f'AI(強気度:{threshold})が過去データでトレード中...'):
            # 戻り値を3つに変更
            res_df, ret_ai, ret_market = run_backtest(raw_df, threshold)
            
        # 結果サマリー
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("🤖 AI戦略 (実力)", f"{ret_ai:+.2f}%")
        with col_res2:
            st.metric("🐻 TOPIXガチホ (実力)", f"{ret_market:+.2f}%")

        # --- チャート描画 ---
        st.subheader("📊 売買タイミング検証")
        fig = go.Figure()
        
        # 1. 市場平均 (黒線)
        fig.add_trace(go.Scatter(
            x=res_df.index, y=res_df["TOPIXガチホ"],
            mode='lines', name='TOPIXガチホ',
            line=dict(color='black', width=2)
        ))
        
        # 2. AI戦略 (赤線)
        fig.add_trace(go.Scatter(
            x=res_df.index, y=res_df["AI戦略"],
            mode='lines', name='AI戦略',
            line=dict(color='red', width=2)
        ))
        
        # 3. 売買ポイント
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
        
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"予期せぬエラーが発生しました: {e}")