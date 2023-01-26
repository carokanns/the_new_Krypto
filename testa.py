import yfinance as yf
import pandas as pd
import streamlit as st

tickers = ['BTC-USD', 'ETH-USD', 'BCH-USD', 'ZRX-USD', 'XRP-USD']
ticker_names = ['Bitcoin', 'Ethereum', 'Bitcoin Cash', '0X', 'Ripple']


def import_ticker_data(tickers, ticker_names):
    data = yf.download(tickers, interval='1d',
                       group_by='ticker', auto_adjust=True)
    df = pd.DataFrame(data.xs('Close', level=1, axis=1))
    df.columns = ticker_names
    return df


def plot_ticker_data(df, ticker):
    data = df[ticker].pct_change()*100
    st.line_chart(data)

if 'data' not in st.session_state:
    st.session_state.data = import_ticker_data(tickers, ticker_names)
    st.write("Data imported")
    
data = st.session_state.data

st.title("Stock Ticker Information")
ticker = st.selectbox("Select Ticker", ticker_names)
plot_ticker_data(data, ticker)
