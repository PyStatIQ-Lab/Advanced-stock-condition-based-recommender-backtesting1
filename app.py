import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import io

st.set_page_config(layout="wide", page_title="📈 Stock Recommender & Backtester")

@st.cache_data(ttl=3600)
def get_stock_data(symbol, period='6mo'):
    try:
        stock = yf.Ticker(symbol)
        return stock.history(period=period)
    except Exception as e:
        st.error(f"Error fetching {symbol}: {str(e)}")
        return None

def calculate_rsi(prices, window=14):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window).mean()
    avg_loss = pd.Series(loss).rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values[-1]

def analyze_stock(symbol, hist):
    if hist is None or hist.empty or len(hist) < 20:
        return None

    close_prices = hist['Close']
    latest = hist.iloc[-1]
    prev = hist.iloc[-2]

    rsi = calculate_rsi(close_prices.values)
    macd = close_prices.ewm(span=12).mean().iloc[-1] - close_prices.ewm(span=26).mean().iloc[-1]

    confidence = 50
    if rsi < 30:
        confidence += 15
    elif rsi > 70:
        confidence -= 15

    if macd > 0:
        confidence += 10
    else:
        confidence -= 10

    recommendation = "Buy" if confidence >= 70 else "Sell" if confidence <= 30 else "Neutral"
    price_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100

    return {
        'Symbol': symbol,
        'Date': latest.name.strftime('%Y-%m-%d'),
        'Price': round(latest['Close'], 2),
        'Change (%)': round(price_change, 2),
        'RSI': round(rsi, 2),
        'MACD': round(macd, 2),
        'Confidence': round(confidence),
        'Recommendation': recommendation
    }

def analyze_stock_on_date(symbol, hist, idx):
    if idx >= len(hist) - 5:
        return None

    data = hist.iloc[:idx+1]
    res = analyze_stock(symbol, data)
    if not res or res['Recommendation'] == "Neutral":
        return None

    close = hist.iloc[idx]['Close']
    future = hist.iloc[idx+1:idx+6]['Close']
    target = close * 1.04 if res['Recommendation'] == 'Buy' else close * 0.96
    stop = close * 0.98 if res['Recommendation'] == 'Buy' else close * 1.02

    target_hit = any(price >= target if res['Recommendation'] == 'Buy' else price <= target for price in future)
    stop_hit = any(price <= stop if res['Recommendation'] == 'Buy' else price >= stop for price in future)

    res.update({
        'Backtest Date': hist.index[idx].strftime('%Y-%m-%d'),
        'Target Hit': target_hit,
        'Stop Loss Hit': stop_hit,
        'Outcome': "Target" if target_hit else "Stop Loss" if stop_hit else "Undecided"
    })
    return res

def backtest_stock(symbol, period='6mo'):
    hist = get_stock_data(symbol, period)
    results = []
    if hist is None or len(hist) < 25:
        return pd.DataFrame()
    for i in range(20, len(hist)-5):
        res = analyze_stock_on_date(symbol, hist, i)
        if res:
            results.append(res)
    return pd.DataFrame(results)

def generate_html_report(recommend_df, backtest_df):
    html = f"""
    <html>
    <head><style>
    h2 {{ color: #2e86c1; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background-color: #f2f2f2; }}
    </style></head>
    <body>
    <h2>📌 Stock Recommendations</h2>
    {recommend_df.to_html(index=False)}

    <h2>🔁 Backtest Results</h2>
    {backtest_df.to_html(index=False)}

    <h3>📊 Summary</h3>
    <ul>
    <li>Total Backtests: {len(backtest_df)}</li>
    <li>Target Hits: {(backtest_df['Outcome'] == 'Target').sum()}</li>
    <li>Stop Loss Hits: {(backtest_df['Outcome'] == 'Stop Loss').sum()}</li>
    <li>Undecided: {(backtest_df['Outcome'] == 'Undecided').sum()}</li>
    <li>Win Rate: {(backtest_df['Outcome'] == 'Target').sum() / len(backtest_df) * 100:.2f}%</li>
    </ul>
    </body>
    </html>
    """
    return html

def main():
    st.title("📈 Stock Recommender + Backtester")

    try:
        stock_sheets = pd.ExcelFile('stocklist.xlsx').sheet_names
    except FileNotFoundError:
        st.error("Please upload a valid `stocklist.xlsx` file.")
        return

    selected_sheet = st.selectbox("Select Stock List Sheet", stock_sheets)
    period = st.selectbox("Backtest Period", ['3mo', '6mo', '1y'])

    if st.button("🚀 Run Analysis & Backtest"):
        stock_df = pd.read_excel('stocklist.xlsx', sheet_name=selected_sheet)
        symbols = stock_df['Symbol'].dropna().unique().tolist()

        st.subheader("📌 Live Recommendations")
        rec_results = []
        bt_results = []
        prog = st.progress(0)
        for i, symbol in enumerate(symbols):
            hist = get_stock_data(symbol, '1mo')
            result = analyze_stock(symbol, hist)
            if result:
                rec_results.append(result)
            
            bt_df = backtest_stock(symbol, period)
            if not bt_df.empty:
                bt_results.append(bt_df)
            
            prog.progress((i+1)/len(symbols))

        rec_df = pd.DataFrame(rec_results)
        bt_df_all = pd.concat(bt_results, ignore_index=True)

        if not rec_df.empty:
            st.dataframe(rec_df)
        else:
            st.info("No recommendations found.")

        if not bt_df_all.empty:
            st.subheader("🔁 Backtest Summary")
            st.dataframe(bt_df_all[['Symbol', 'Backtest Date', 'Recommendation', 'Outcome']])
        else:
            st.info("No backtest data found.")

        # Generate HTML report
        html_report = generate_html_report(rec_df, bt_df_all)
        st.download_button("📥 Download HTML Report",
                           data=html_report,
                           file_name=f'stock_report_{datetime.now().strftime("%Y%m%d")}.html',
                           mime='text/html')

if __name__ == "__main__":
    main()
