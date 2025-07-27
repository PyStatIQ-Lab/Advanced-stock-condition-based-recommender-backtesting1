import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide", page_title="📊 Multi-Model Stock Recommender & Backtester")

@st.cache_data(ttl=3600)
def get_stock_data(symbol, period='6mo'):
    try:
        stock = yf.Ticker(symbol)
        return stock.history(period=period)
    except:
        return None

def calculate_rsi(prices, window=14):
    delta = np.diff(prices)
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    gain = pd.Series(up).rolling(window).mean()
    loss = pd.Series(down).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values[-1]

def calculate_macd(prices):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    return ema12.iloc[-1] - ema26.iloc[-1]

def calculate_bollinger(prices, window=20):
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return sma.iloc[-1] + 2 * std.iloc[-1], sma.iloc[-1] - 2 * std.iloc[-1]

def calculate_tsi(prices, short=13, long=25):
    diff = np.diff(prices)
    ema_short = pd.Series(diff).ewm(span=short).mean()
    ema_long = ema_short.ewm(span=long).mean()
    abs_diff = np.abs(diff)
    ema_abs_short = pd.Series(abs_diff).ewm(span=short).mean()
    ema_abs_long = ema_abs_short.ewm(span=long).mean()
    return 100 * (ema_long.iloc[-1] / ema_abs_long.iloc[-1]) if ema_abs_long.iloc[-1] != 0 else 0

def detect_support_resistance(prices):
    resistance = prices.rolling(20).max().iloc[-1]
    support = prices.rolling(20).min().iloc[-1]
    current = prices.iloc[-1]
    if current >= resistance * 0.98:
        return "Near Resistance (Bearish)"
    elif current <= support * 1.02:
        return "Near Support (Bullish)"
    return None

def analyze_stock_logic(symbol, hist):
    if hist is None or len(hist) < 30:
        return None

    latest = hist.iloc[-1]
    prev = hist.iloc[-2]
    close = hist['Close']

    confidence = 50
    details = []

    # 1. Open-High/Low Condition
    if latest['Open'] == latest['High']:
        details.append("Open = High (Bearish)")
        confidence -= 20
        rec = "Sell"
    elif latest['Open'] == latest['Low']:
        details.append("Open = Low (Bullish)")
        confidence += 20
        rec = "Buy"
    else:
        details.append("No OH/OL condition")
        rec = "Neutral"

    # 2. RSI
    rsi = calculate_rsi(close.values)
    if rsi < 30:
        details.append("RSI Oversold")
        confidence += 10
    elif rsi > 70:
        details.append("RSI Overbought")
        confidence -= 10

    # 3. MACD
    macd = calculate_macd(close)
    if macd > 0:
        details.append("MACD Bullish")
        confidence += 5
    else:
        details.append("MACD Bearish")
        confidence -= 5

    # 4. Bollinger Bands
    upper, lower = calculate_bollinger(close)
    if latest['Close'] < lower:
        details.append("Bollinger: Oversold")
        confidence += 5
    elif latest['Close'] > upper:
        details.append("Bollinger: Overbought")
        confidence -= 5

    # 5. TSI
    tsi = calculate_tsi(close.values)
    if tsi > 25:
        details.append("TSI Strong Trend Up")
        confidence += 5
    elif tsi < -25:
        details.append("TSI Strong Trend Down")
        confidence -= 5

    # 6. Support/Resistance
    sr = detect_support_resistance(close)
    if sr:
        details.append(f"SR: {sr}")
        if "Support" in sr:
            confidence += 5
        elif "Resistance" in sr:
            confidence -= 5

    if confidence >= 70:
        rec = "Buy"
    elif confidence <= 30:
        rec = "Sell"
    else:
        rec = "Neutral"

    stop = round(latest['Close'] * (0.98 if rec == 'Buy' else 1.02), 2)
    target = round(latest['Close'] * (1.04 if rec == 'Buy' else 0.96), 2)
    change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100

    return {
        "Symbol": symbol,
        "Price": round(latest['Close'], 2),
        "Change %": round(change, 2),
        "Recommendation": rec,
        "Confidence": int(confidence),
        "Stop Loss": stop,
        "Target": target,
        "Primary Condition": details[0],
        "Details": " | ".join(details)
    }

def backtest(symbol, hist):
    results = []
    for i in range(25, len(hist)-5):
        sub_hist = hist.iloc[:i+1]
        result = analyze_stock_logic(symbol, sub_hist)
        if result and result['Recommendation'] != "Neutral":
            future = hist.iloc[i+1:i+6]['Close']
            buy = result['Recommendation'] == "Buy"
            hit_target = any(future >= result['Target']) if buy else any(future <= result['Target'])
            hit_stop = any(future <= result['Stop Loss']) if buy else any(future >= result['Stop Loss'])
            result.update({
                "Date": hist.index[i].strftime('%Y-%m-%d'),
                "Target Hit": hit_target,
                "Stop Loss Hit": hit_stop,
                "Outcome": "Target" if hit_target else "Stop Loss" if hit_stop else "Undecided"
            })
            results.append(result)
    return pd.DataFrame(results)

def generate_html_report(recommend_df, backtest_df):
    return f"""
    <html><head><style>
    body {{ font-family: Arial; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background-color: #f2f2f2; }}
    </style></head><body>
    <h2>📌 Live Recommendations</h2>
    {recommend_df.to_html(index=False, escape=False)}

    <h2>🔁 Backtesting Results</h2>
    {backtest_df.to_html(index=False)}

    <h3>📊 Backtest Summary</h3>
    <ul>
    <li>Total Signals: {len(backtest_df)}</li>
    <li>Target Hits: {(backtest_df['Outcome'] == 'Target').sum()}</li>
    <li>Stop Loss Hits: {(backtest_df['Outcome'] == 'Stop Loss').sum()}</li>
    <li>Undecided: {(backtest_df['Outcome'] == 'Undecided').sum()}</li>
    <li>Win Rate: {(backtest_df['Outcome'] == 'Target').sum() / len(backtest_df) * 100:.2f}%</li>
    </ul>
    </body></html>
    """

def main():
    st.title("📊 Multi-Model Stock Recommender & Backtester")

    try:
        sheets = pd.ExcelFile("stocklist.xlsx").sheet_names
    except FileNotFoundError:
        st.error("Upload 'stocklist.xlsx' with Symbol column.")
        return

    sheet = st.selectbox("Select Stock List", sheets)
    period = st.selectbox("Backtest Period", ["3mo", "6mo", "1y"], index=1)

    if st.button("Run Analysis"):
        df = pd.read_excel("stocklist.xlsx", sheet_name=sheet)
        symbols = df['Symbol'].dropna().unique().tolist()

        rec_results = []
        bt_results = []
        prog = st.progress(0)
        for i, symbol in enumerate(symbols):
            hist = get_stock_data(symbol, period)
            rec = analyze_stock_logic(symbol, hist)
            if rec:
                rec_results.append(rec)

            bt_df = backtest(symbol, hist)
            if not bt_df.empty:
                bt_results.append(bt_df)

            prog.progress((i+1)/len(symbols))

        rec_df = pd.DataFrame(rec_results)
        bt_df_all = pd.concat(bt_results, ignore_index=True)

        st.subheader("📌 Live Recommendations")
        st.dataframe(rec_df)

        st.subheader("🔁 Backtest Results")
        st.dataframe(bt_df_all[['Symbol', 'Date', 'Recommendation', 'Confidence', 'Outcome']])

        html_report = generate_html_report(rec_df, bt_df_all)
        st.download_button("📥 Download HTML Report",
                           data=html_report,
                           file_name=f'stock_analysis_{datetime.now().strftime("%Y%m%d")}.html',
                           mime='text/html')

if __name__ == "__main__":
    main()
