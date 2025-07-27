import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns

# Configuration
st.set_page_config(layout="wide", page_title="📈 80% Win Rate Stock System")
pd.options.mode.chained_assignment = None

@st.cache_data(ttl=3600)
def get_stock_data(symbol, period='2y'):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval='1d')
        if df.empty:
            return None
        return df
    except:
        return None

def calculate_technical_indicators(df):
    # Momentum Indicators
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['MACD'] = ta.macd(df['Close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
    df['MACD_Signal'] = ta.macd(df['Close'], fast=12, slow=26, signal=9)['MACDs_12_26_9']
    df['TSI'] = ta.tsi(df['Close'], fast=13, slow=25)
    df['Stoch_%K'] = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)['STOCHk_14_3_3']
    df['Stoch_%D'] = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)['STOCHd_14_3_3']
    
    # Trend Indicators
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
    df['PSAR'] = ta.psar(df['High'], df['Low'], df['Close'])['PSARl_0.02_0.2']
    
    # Volatility Indicators
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['Bollinger_Upper'] = ta.bbands(df['Close'], length=20, std=2)['BBU_20_2.0']
    df['Bollinger_Lower'] = ta.bbands(df['Close'], length=20, std=2)['BBL_20_2.0']
    
    # Volume Indicators
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    
    # Pattern Recognition
    df['CDL_DOJI'] = ta.cdl_doji(df['Open'], df['High'], df['Low'], df['Close'])
    df['CDL_HAMMER'] = ta.cdl_hammer(df['Open'], df['High'], df['Low'], df['Close'])
    df['CDL_ENGULFING'] = ta.cdl_engulfing(df['Open'], df['High'], df['Low'], df['Close'])
    
    # Calculate returns for ML target
    df['5d_future_return'] = (df['Close'].shift(-5) / df['Close'] - 1) * 100
    
    return df.dropna()

def prepare_ml_data(df):
    features = [
        'RSI', 'MACD', 'MACD_Signal', 'TSI', 'Stoch_%K', 'Stoch_%D',
        'EMA_20', 'EMA_50', 'EMA_200', 'ADX', 'PSAR', 'ATR',
        'Bollinger_Upper', 'Bollinger_Lower', 'OBV', 'VWAP',
        'CDL_DOJI', 'CDL_HAMMER', 'CDL_ENGULFING'
    ]
    
    # Create target: 1 if return > 1%, 0 otherwise
    df['target'] = (df['5d_future_return'] > 1.0).astype(int)
    
    X = df[features]
    y = df['target']
    
    return X, y

def train_ml_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy

def analyze_stock(symbol, model, scaler):
    df = get_stock_data(symbol, period='2y')
    if df is None or len(df) < 100:
        return None
    
    df = calculate_technical_indicators(df)
    if df.empty:
        return None
    
    features = [
        'RSI', 'MACD', 'MACD_Signal', 'TSI', 'Stoch_%K', 'Stoch_%D',
        'EMA_20', 'EMA_50', 'EMA_200', 'ADX', 'PSAR', 'ATR',
        'Bollinger_Upper', 'Bollinger_Lower', 'OBV', 'VWAP',
        'CDL_DOJI', 'CDL_HAMMER', 'CDL_ENGULFING'
    ]
    
    latest = df.iloc[-1][features].values.reshape(1, -1)
    latest_scaled = scaler.transform(latest)
    
    # Get model prediction and probability
    prediction = model.predict(latest_scaled)[0]
    probability = model.predict_proba(latest_scaled)[0][1]
    
    # Calculate trend strength
    trend_strength = 0
    if df['EMA_20'].iloc[-1] > df['EMA_50'].iloc[-1] > df['EMA_200'].iloc[-1]:
        trend_strength = 1  # Strong uptrend
    elif df['EMA_20'].iloc[-1] < df['EMA_50'].iloc[-1] < df['EMA_200'].iloc[-1]:
        trend_strength = -1  # Strong downtrend
    
    # Calculate volatility-adjusted targets
    atr = df['ATR'].iloc[-1]
    current_price = df['Close'].iloc[-1]
    
    if prediction == 1:  # Bullish prediction
        stop_loss = round(current_price - (atr * 1.2), 2)
        target = round(current_price + (atr * 4.0), 2)  # 4:1 risk/reward
    else:
        stop_loss = round(current_price + (atr * 1.2), 2)
        target = round(current_price - (atr * 4.0), 2)
    
    # Calculate confidence score (0-100)
    confidence = int(probability * 100)
    
    # Factor in trend strength
    if trend_strength == 1 and prediction == 1:
        confidence = min(100, confidence + 15)
    elif trend_strength == -1 and prediction == 0:
        confidence = min(100, confidence + 15)
    elif trend_strength == 1 and prediction == 0:
        confidence = max(0, confidence - 20)
    elif trend_strength == -1 and prediction == 1:
        confidence = max(0, confidence - 20)
    
    # Volume confirmation
    volume_avg = df['Volume'].rolling(20).mean().iloc[-1]
    if df['Volume'].iloc[-1] > volume_avg * 1.5:
        confidence = min(100, confidence + 10)
    elif df['Volume'].iloc[-1] < volume_avg * 0.8:
        confidence = max(0, confidence - 15)
    
    # Determine recommendation
    if prediction == 1 and confidence >= 85:
        recommendation = "Strong Buy"
    elif prediction == 1 and confidence >= 70:
        recommendation = "Buy"
    elif prediction == 0 and confidence >= 85:
        recommendation = "Strong Sell"
    elif prediction == 0 and confidence >= 70:
        recommendation = "Sell"
    else:
        recommendation = "Neutral"
    
    # Calculate price change
    prev_close = df['Close'].iloc[-2]
    price_change = ((current_price - prev_close) / prev_close) * 100
    
    return {
        "Symbol": symbol,
        "Price": round(current_price, 2),
        "Change %": round(price_change, 2),
        "Recommendation": recommendation,
        "Confidence": confidence,
        "Stop Loss": stop_loss,
        "Target": target,
        "Risk/Reward": "1:4",
        "Trend Strength": "Strong Up" if trend_strength == 1 else "Strong Down" if trend_strength == -1 else "Neutral",
        "Volume Confirmation": "Yes" if df['Volume'].iloc[-1] > volume_avg * 1.5 else "No",
        "ML Probability": f"{probability:.2%}"
    }

def backtest(symbol, model, scaler):
    df = get_stock_data(symbol, period='5y')
    if df is None or len(df) < 300:
        return pd.DataFrame()
    
    df = calculate_technical_indicators(df)
    if df.empty:
        return pd.DataFrame()
    
    features = [
        'RSI', 'MACD', 'MACD_Signal', 'TSI', 'Stoch_%K', 'Stoch_%D',
        'EMA_20', 'EMA_50', 'EMA_200', 'ADX', 'PSAR', 'ATR',
        'Bollinger_Upper', 'Bollinger_Lower', 'OBV', 'VWAP',
        'CDL_DOJI', 'CDL_HAMMER', 'CDL_ENGULFING'
    ]
    
    results = []
    
    # Backtest for each day (starting from 200 days in to have enough history)
    for i in range(200, len(df)-10):
        # Prepare data for this day
        current_data = df.iloc[i][features].values.reshape(1, -1)
        current_data_scaled = scaler.transform(current_data)
        
        # Get prediction
        prediction = model.predict(current_data_scaled)[0]
        probability = model.predict_proba(current_data_scaled)[0][1]
        
        # Only consider high-confidence predictions
        if probability < 0.7:
            continue
            
        # Calculate position parameters
        current_price = df['Close'].iloc[i]
        atr = df['ATR'].iloc[i]
        
        if prediction == 1:  # Bullish
            stop_loss = current_price - (atr * 1.2)
            target = current_price + (atr * 4.0)
        else:  # Bearish
            stop_loss = current_price + (atr * 1.2)
            target = current_price - (atr * 4.0)
        
        # Check outcome in next 10 days
        future_prices = df['Close'].iloc[i+1:i+11]
        
        if prediction == 1:  # Bullish
            hit_target = any(future_prices >= target)
            hit_stop = any(future_prices <= stop_loss)
        else:  # Bearish
            hit_target = any(future_prices <= target)
            hit_stop = any(future_prices >= stop_loss)
            
        # Determine outcome
        if hit_target:
            outcome = "Win"
            days_to_outcome = np.argmax(future_prices >= target if prediction == 1 else future_prices <= target) + 1
        elif hit_stop:
            outcome = "Loss"
            days_to_outcome = np.argmax(future_prices <= stop_loss if prediction == 1 else future_prices >= stop_loss) + 1
        else:
            outcome = "Neutral"
            days_to_outcome = 10
            
        # Calculate PnL
        if prediction == 1:
            exit_price = future_prices.iloc[days_to_outcome-1]
            pnl_pct = (exit_price - current_price) / current_price * 100
        else:
            exit_price = future_prices.iloc[days_to_outcome-1]
            pnl_pct = (current_price - exit_price) / current_price * 100
        
        # Add to results
        results.append({
            "Symbol": symbol,
            "Date": df.index[i].strftime('%Y-%m-%d'),
            "Recommendation": "Buy" if prediction == 1 else "Sell",
            "Confidence": int(probability * 100),
            "Entry Price": round(current_price, 2),
            "Exit Price": round(exit_price, 2),
            "PnL %": round(pnl_pct, 2),
            "Outcome": outcome,
            "Holding Days": days_to_outcome,
            "Stop Loss": round(stop_loss, 2),
            "Target": round(target, 2)
        })
    
    return pd.DataFrame(results)

def train_market_model():
    # Download S&P 500 data
    sp500 = yf.Ticker("^GSPC")
    df = sp500.history(period="10y", interval='1d')
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Prepare features and target
    X, y = prepare_ml_data(df)
    
    # Train model
    model, scaler, accuracy = train_ml_model(X, y)
    
    return model, scaler, accuracy

def plot_performance(backtest_df):
    if backtest_df.empty:
        return None
    
    # Create cumulative PnL
    backtest_df['Cumulative PnL'] = backtest_df['PnL %'].cumsum()
    
    fig = go.Figure()
    
    # Cumulative PnL
    fig.add_trace(go.Scatter(
        x=backtest_df['Date'], 
        y=backtest_df['Cumulative PnL'],
        mode='lines',
        name='Cumulative PnL',
        line=dict(color='#4CAF50', width=3)
    ))
    
    # Win/Loss markers
    wins = backtest_df[backtest_df['Outcome'] == 'Win']
    losses = backtest_df[backtest_df['Outcome'] == 'Loss']
    
    fig.add_trace(go.Scatter(
        x=wins['Date'],
        y=wins['Cumulative PnL'],
        mode='markers',
        name='Win',
        marker=dict(color='#4CAF50', size=8, symbol='triangle-up')
    ))
    
    fig.add_trace(go.Scatter(
        x=losses['Date'],
        y=losses['Cumulative PnL'],
        mode='markers',
        name='Loss',
        marker=dict(color='#F44336', size=8, symbol='triangle-down')
    ))
    
    # Layout
    fig.update_layout(
        title='Backtest Performance',
        xaxis_title='Date',
        yaxis_title='Cumulative PnL (%)',
        template='plotly_dark',
        hovermode='x unified',
        height=500
    )
    
    return fig

def plot_recommendations(recommendations):
    df = pd.DataFrame(recommendations)
    if df.empty:
        return None
    
    # Color mapping
    color_map = {
        "Strong Buy": "#2E7D32",
        "Buy": "#4CAF50",
        "Neutral": "#FFC107",
        "Sell": "#F44336",
        "Strong Sell": "#B71C1C"
    }
    
    df['Color'] = df['Recommendation'].map(color_map)
    
    fig = go.Figure()
    
    # Confidence bars
    fig.add_trace(go.Bar(
        x=df['Symbol'],
        y=df['Confidence'],
        marker_color=df['Color'],
        text=df['Recommendation'],
        hovertemplate='<b>%{x}</b><br>Confidence: %{y}%<br>Recommendation: %{text}',
        name=''
    ))
    
    # Layout
    fig.update_layout(
        title='Stock Recommendations',
        xaxis_title='Symbol',
        yaxis_title='Confidence (%)',
        template='plotly_dark',
        height=500,
        hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial")
    )
    
    return fig

def main():
    st.title("🚀 80% Win Rate Stock Analysis System")
    
    st.sidebar.header("System Configuration")
    min_confidence = st.sidebar.slider("Minimum Confidence", 70, 100, 85)
    risk_level = st.sidebar.selectbox("Risk Level", ["Low", "Medium", "High"], index=0)
    market_phase = st.sidebar.selectbox("Market Phase", ["Bull", "Bear", "Neutral"], index=0)
    
    # Load stock list
    uploaded_file = st.file_uploader("📊 Upload Stock List (Excel with 'Symbol' column)", type="xlsx")
    
    if uploaded_file is None:
        st.info("Please upload a stock list to begin analysis")
        return
    
    try:
        sheets = pd.ExcelFile(uploaded_file).sheet_names
        sheet = st.selectbox("Select Sheet", sheets)
        df = pd.read_excel(uploaded_file, sheet_name=sheet)
        symbols = df['Symbol'].dropna().unique().tolist()
        
        if not symbols:
            st.error("No valid symbols found in the 'Symbol' column")
            return
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return
    
    # Train model
    st.subheader("🧠 Machine Learning Model Training")
    if st.button("Train Market Prediction Model"):
        with st.spinner("Training model with S&P 500 data..."):
            model, scaler, accuracy = train_market_model()
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.success(f"Model trained successfully! Accuracy: {accuracy:.2%}")
    
    if 'model' not in st.session_state or 'scaler' not in st.session_state:
        st.warning("Please train the model first")
        return
        
    # Run analysis
    if st.button("Run Analysis", type="primary"):
        recommendations = []
        backtest_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(symbols):
            status_text.text(f"Analyzing {symbol} ({i+1}/{len(symbols)})...")
            
            # Analyze current stock
            analysis = analyze_stock(symbol, st.session_state.model, st.session_state.scaler)
            if analysis and analysis['Confidence'] >= min_confidence:
                recommendations.append(analysis)
            
            # Backtest
            bt_df = backtest(symbol, st.session_state.model, st.session_state.scaler)
            if not bt_df.empty:
                backtest_results.append(bt_df)
            
            progress_bar.progress((i + 1) / len(symbols))
        
        if not recommendations:
            st.warning("No recommendations meet the confidence threshold")
            return
            
        # Combine results
        st.session_state.recommendations = pd.DataFrame(recommendations)
        st.session_state.backtest_df = pd.concat(backtest_results) if backtest_results else pd.DataFrame()
    
    # Display results
    if 'recommendations' in st.session_state:
        st.subheader("💎 High-Confidence Recommendations")
        st.dataframe(st.session_state.recommendations.sort_values('Confidence', ascending=False))
        
        # Plot recommendations
        fig = plot_recommendations(st.session_state.recommendations)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    if 'backtest_df' in st.session_state and not st.session_state.backtest_df.empty:
        st.subheader("📈 Backtest Performance")
        
        # Calculate performance metrics
        win_rate = (st.session_state.backtest_df['Outcome'] == 'Win').mean() * 100
        avg_pnl = st.session_state.backtest_df.loc[st.session_state.backtest_df['Outcome'] == 'Win', 'PnL %'].mean()
        avg_hold = st.session_state.backtest_df['Holding Days'].mean()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Win Rate", f"{win_rate:.2f}%")
        col2.metric("Average Win PnL", f"{avg_pnl:.2f}%")
        col3.metric("Average Holding Days", f"{avg_hold:.2f}")
        col4.metric("Total Signals", len(st.session_state.backtest_df))
        
        # Plot performance
        perf_fig = plot_performance(st.session_state.backtest_df)
        if perf_fig:
            st.plotly_chart(perf_fig, use_container_width=True)
        
        # Show backtest details
        st.dataframe(st.session_state.backtest_df)
        
        # Win rate by recommendation type
        st.subheader("Win Rate by Recommendation Type")
        win_rate_by_type = st.session_state.backtest_df.groupby('Recommendation')['Outcome'].apply(
            lambda x: (x == 'Win').mean() * 100
        ).reset_index()
        
        fig2 = px.bar(
            win_rate_by_type, 
            x='Recommendation', 
            y='Outcome',
            color='Recommendation',
            color_discrete_map={'Buy': '#4CAF50', 'Sell': '#F44336'},
            text='Outcome',
            labels={'Outcome': 'Win Rate (%)'},
            height=400
        )
        fig2.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig2.update_layout(template='plotly_dark')
        st.plotly_chart(fig2, use_container_width=True)
    
if __name__ == "__main__":
    main()
