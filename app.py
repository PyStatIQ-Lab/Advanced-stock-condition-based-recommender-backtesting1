import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide", page_title="📊 Enhanced Stock Recommender")

@st.cache_data(ttl=3600)
def get_stock_data(symbol, period='1y'):
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

def calculate_atr(hist, window=14):
    high_low = hist['High'] - hist['Low']
    high_close = np.abs(hist['High'] - hist['Close'].shift())
    low_close = np.abs(hist['Low'] - hist['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean().iloc[-1]

def calculate_trend(close):
    if len(close) < 200:
        return "Neutral"
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
    return "Bullish" if ema50 > ema200 else "Bearish" if ema50 < ema200 else "Neutral"

def check_fundamentals(symbol):
    try:
        stock = yf.Ticker(symbol)
        pe = stock.info.get('trailingPE', 100)
        debt = stock.info.get('debtToEquity', 1)
        return pe < 25 and debt < 0.8
    except:
        return True

def analyze_stock_logic(symbol, hist):
    if hist is None or len(hist) < 200:
        return None

    latest = hist.iloc[-1]
    prev = hist.iloc[-2]
    close = hist['Close']
    volume = hist['Volume']
    
    # Calculate all indicators first
    rsi = calculate_rsi(close.values) if len(close) >= 15 else 50
    macd = calculate_macd(close)
    upper_bb, lower_bb = calculate_bollinger(close)
    tsi = calculate_tsi(close.values) if len(close) >= 26 else 0
    sr = detect_support_resistance(close)
    trend = calculate_trend(close)
    atr = calculate_atr(hist)
    avg_volume = volume.rolling(20).mean().iloc[-1]
    
    # Initialize conditions
    confidence = 50
    details = []
    bullish_conditions = 0
    bearish_conditions = 0

    # 1. OHLC Pattern (weight=2)
    if latest['Open'] == latest['High']:
        details.append("Open=High (Bearish)")
        bearish_conditions += 2
        confidence -= 15
    elif latest['Open'] == latest['Low']:
        details.append("Open=Low (Bullish)")
        bullish_conditions += 2
        confidence += 15
    else:
        details.append("No OH/OL pattern")

    # 2. RSI with tighter thresholds (weight=2)
    if rsi < 28:
        details.append(f"RSI Oversold ({rsi:.1f})")
        bullish_conditions += 2
        confidence += 10
    elif rsi > 72:
        details.append(f"RSI Overbought ({rsi:.1f})")
        bearish_conditions += 2
        confidence -= 10

    # 3. MACD (weight=1)
    if macd > 0:
        details.append("MACD Bullish")
        bullish_conditions += 1
        confidence += 5
    else:
        details.append("MACD Bearish")
        bearish_conditions += 1
        confidence -= 5

    # 4. Bollinger Bands (weight=1)
    if latest['Close'] < lower_bb:
        details.append("Below Bollinger Lower")
        bullish_conditions += 1
        confidence += 5
    elif latest['Close'] > upper_bb:
        details.append("Above Bollinger Upper")
        bearish_conditions += 1
        confidence -= 5

    # 5. TSI (weight=1)
    if tsi > 25:
        details.append(f"TSI Strong Up ({tsi:.1f})")
        bullish_conditions += 1
        confidence += 5
    elif tsi < -25:
        details.append(f"TSI Strong Down ({tsi:.1f})")
        bearish_conditions += 1
        confidence -= 5

    # 6. Support/Resistance (weight=1.5)
    if sr:
        details.append(sr)
        if "Support" in sr:
            bullish_conditions += 1.5
            confidence += 7
        elif "Resistance" in sr:
            bearish_conditions += 1.5
            confidence -= 7

    # 7. Volume confirmation (required)
    volume_ok = latest['Volume'] > avg_volume * 1.2
    details.append(f"Volume: {'Above Avg' if volume_ok else 'Below Avg'}")

    # Determine initial recommendation
    if bullish_conditions >= 3 and bearish_conditions < 1:
        rec = "Buy"
        confidence = min(95, confidence + 10)
    elif bearish_conditions >= 3 and bullish_conditions < 1:
        rec = "Sell"
        confidence = max(5, confidence - 10)
    else:
        rec = "Neutral"

    # Trend filter (required)
    if rec == "Buy" and trend != "Bullish":
        details.append(f"Trend Filter: {trend} (Blocked Buy)")
        rec = "Neutral"
        confidence = max(40, confidence - 15)
    elif rec == "Sell" and trend != "Bearish":
        details.append(f"Trend Filter: {trend} (Blocked Sell)")
        rec = "Neutral"
        confidence = min(60, confidence + 15)

    # Volume filter (required)
    if rec != "Neutral" and not volume_ok:
        details.append("Volume Filter: Blocked Signal")
        rec = "Neutral"
        confidence = 50

    # Fundamental check
    if rec != "Neutral" and not check_fundamentals(symbol):
        details.append("Fundamentals: Blocked Signal")
        rec = "Neutral"
        confidence = 50

    # Final confidence thresholds
    if confidence >= 75:
        rec = "Buy"
    elif confidence <= 25:
        rec = "Sell"
    else:
        rec = "Neutral"

    # Volatility-adjusted targets
    risk_reward_ratio = 2.0
    if rec == 'Buy' and atr > 0:
        stop = round(latest['Close'] - 1.5 * atr, 2)
        target = round(latest['Close'] + (1.5 * atr * risk_reward_ratio), 2)
    elif rec == 'Sell' and atr > 0:
        stop = round(latest['Close'] + 1.5 * atr, 2)
        target = round(latest['Close'] - (1.5 * atr * risk_reward_ratio), 2)
    else:
        stop = target = None

    change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100

    return {
        "Symbol": symbol,
        "Price": round(latest['Close'], 2),
        "Change %": round(change, 2),
        "Recommendation": rec,
        "Confidence": int(confidence),
        "Stop Loss": stop,
        "Target": target,
        "Trend": trend,
        "ATR": round(atr, 2),
        "Details": " | ".join(details)
    }

def backtest(symbol, hist):
    results = []
    for i in range(200, len(hist)-10):  # Start from 200 for EMA
        sub_hist = hist.iloc[:i+1]
        result = analyze_stock_logic(symbol, sub_hist)
        if result and result['Recommendation'] != "Neutral":
            future = hist.iloc[i+1:i+11]  # 10-day holding period
            entry_price = sub_hist['Close'].iloc[-1]
            buy = result['Recommendation'] == "Buy"
            atr = result['ATR']
            
            # Initialize tracking variables
            hit_target = False
            hit_stop = False
            exit_price = None
            holding_days = 0
            
            # Trailing stop parameters
            trail_stop = entry_price * (0.95 if buy else 1.05)
            max_price = entry_price
            min_price = entry_price
            
            for j in range(len(future)):
                current = future.iloc[j]
                current_close = current['Close']
                
                # Update trailing stop
                if buy:
                    if current_close > max_price:
                        max_price = current_close
                        trail_stop = max(trail_stop, max_price * 0.95)
                    if current_close <= trail_stop:
                        hit_stop = True
                        exit_price = current_close
                        break
                    if current_close >= result['Target']:
                        hit_target = True
                        exit_price = result['Target']
                        break
                else:  # Sell
                    if current_close < min_price:
                        min_price = current_close
                        trail_stop = min(trail_stop, min_price * 1.05)
                    if current_close >= trail_stop:
                        hit_stop = True
                        exit_price = current_close
                        break
                    if current_close <= result['Target']:
                        hit_target = True
                        exit_price = result['Target']
                        break
                
                holding_days = j + 1
            
            # Determine outcome
            if hit_target:
                outcome = "Target"
            elif hit_stop:
                outcome = "Stop Loss"
            else:
                outcome = "Undecided"
                exit_price = future['Close'].iloc[-1]
            
            # Calculate PnL
            if buy:
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100
            
            result.update({
                "Entry Date": hist.index[i].strftime('%Y-%m-%d'),
                "Exit Date": future.index[j].strftime('%Y-%m-%d') if outcome != "Undecided" else future.index[-1].strftime('%Y-%m-%d'),
                "Holding Days": holding_days,
                "Exit Price": round(exit_price, 2),
                "PnL %": round(pnl_pct, 2),
                "Outcome": outcome
            })
            results.append(result)
    return pd.DataFrame(results)

def generate_html_report(recommend_df, backtest_df):
    if len(backtest_df) > 0:
        win_rate = (backtest_df['Outcome'] == 'Target').sum() / len(backtest_df) * 100
        avg_pnl = backtest_df['PnL %'].mean()
    else:
        win_rate = 0
        avg_pnl = 0
        
    return f"""
    <html><head><style>
    body {{ font-family: Arial; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background-color: #f2f2f2; }}
    .buy {{ background-color: #e6f7ff; }}
    .sell {{ background-color: #ffe6e6; }}
    </style></head><body>
    <h2>📌 Enhanced Stock Recommendations</h2>
    {recommend_df.to_html(index=False, escape=False, classes='recommend')}

    <h2>🔁 Backtesting Performance</h2>
    {backtest_df.to_html(index=False, classes='backtest')}

    <h3>📊 Performance Summary</h3>
    <ul>
    <li>Total Signals: {len(backtest_df)}</li>
    <li>Target Hits: {(backtest_df['Outcome'] == 'Target').sum()}</li>
    <li>Stop Loss Hits: {(backtest_df['Outcome'] == 'Stop Loss').sum()}</li>
    <li>Undecided: {(backtest_df['Outcome'] == 'Undecided').sum()}</li>
    <li>Win Rate: {win_rate:.2f}%</li>
    <li>Average PnL: {avg_pnl:.2f}%</li>
    <li>Profit Factor: {backtest_df[backtest_df['PnL %'] > 0]['PnL %'].sum() / abs(backtest_df[backtest_df['PnL %'] < 0]['PnL %'].sum()):.2f}</li>
    </ul>
    </body></html>
    """

def main():
    st.title("📊 Enhanced Stock Analysis System")
    
    # Upload stock list
    uploaded_file = st.file_uploader("Upload Stock List (Excel with 'Symbol' column)", type="xlsx")
    if not uploaded_file:
        st.info("Please upload a stock list file to begin")
        return

    try:
        sheets = pd.ExcelFile(uploaded_file).sheet_names
    except:
        st.error("Invalid file format. Please upload a valid Excel file.")
        return

    sheet = st.selectbox("Select Sheet", sheets)
    period = st.selectbox("Backtest Period", ["1y", "2y", "3y"], index=0)
    min_confidence = st.slider("Minimum Confidence", 0, 100, 75)
    
    if st.button("Run Enhanced Analysis"):
        df = pd.read_excel(uploaded_file, sheet_name=sheet)
        symbols = df['Symbol'].dropna().unique().tolist()
        
        if not symbols:
            st.error("No valid symbols found in the 'Symbol' column")
            return

        rec_results = []
        bt_results = []
        prog_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(symbols):
            status_text.text(f"Processing {symbol} ({i+1}/{len(symbols)})...")
            hist = get_stock_data(symbol, period)
            
            if hist is None or len(hist) < 200:
                continue
                
            rec = analyze_stock_logic(symbol, hist)
            if rec and rec['Confidence'] >= min_confidence:
                rec_results.append(rec)
                
            bt_df = backtest(symbol, hist)
            if not bt_df.empty:
                bt_results.append(bt_df)
                
            prog_bar.progress((i+1)/len(symbols))
        
        if not rec_results:
            st.warning("No recommendations met the confidence threshold")
            return
            
        rec_df = pd.DataFrame(rec_results)
        bt_df_all = pd.concat(bt_results, ignore_index=True) if bt_results else pd.DataFrame()

        st.subheader("🔥 Top Recommendations")
        st.dataframe(rec_df.sort_values('Confidence', ascending=False))

        if not bt_df_all.empty:
            st.subheader("📈 Backtest Performance")
            st.dataframe(bt_df_all)
            
            # Performance metrics
            win_rate = (bt_df_all['Outcome'] == 'Target').sum() / len(bt_df_all) * 100
            avg_hold = bt_df_all['Holding Days'].mean()
            st.metric("Win Rate", f"{win_rate:.2f}%")
            st.metric("Average Holding Days", f"{avg_hold:.1f} days")
            
            # Download report
            html_report = generate_html_report(rec_df, bt_df_all)
            st.download_button("📥 Download Full Report",
                               data=html_report,
                               file_name=f'enhanced_stock_report_{datetime.now().strftime("%Y%m%d")}.html',
                               mime='text/html')
        else:
            st.warning("No backtest results available")

if __name__ == "__main__":
    main()
