import matplotlib.pyplot as plt
from pytrends.request import TrendReq
import streamlit as st
from datetime import datetime as dt
import datetime
import pandas as pd
import numpy as np
import ta
# import pickle
from datetime import timedelta
import yfinance as yf
from matplotlib import pyplot as plt
from xgboost import XGBClassifier

tickers = ['BTC-USD', 'ETH-USD', 'BCH-USD', 'ZRX-USD', 'XRP-USD']
ticker_names = ['Bitcoin', 'Ethereum', 'Bitcoin Cash', '0X', 'Ripple']

def get_data(ticker, start="1900-01-01", end=dt.today()):
    data = yf.download(ticker, start=None, end=None)
    data.reset_index(inplace=True)
    data.index = pd.to_datetime(data.Date)
    data = data.drop("Date", axis=1)

    return data


def get_all(tickers):
    all_tickers = {}
    for enum, ticker in enumerate(tickers):
        all_tickers[ticker] = get_data(ticker)
    return all_tickers


def new_features(df_, ticker, target):
    df = df_.copy()
    # tidsintervall i dagar för rullande medelvärden
    # skulle helst ha med upp till 4 år men ETH har för få värden
    horizons = [2, 5, 60, 250]
    new_predictors = []
    df['stoch_k'] = ta.momentum.stochrsi_k(df[ticker], window=10)

    # Target
    # tomorrow's close price - alltså nästa dag
    df['Tomorrow'] = df[ticker].shift(-1)
    # after tomorrow's close price - alltså om två dagar
    df['After_tomorrow'] = df[ticker].shift(-2)
    df['y1'] = (df['Tomorrow'] > df[ticker]).astype(int)
    df['y2'] = (df['After_tomorrow'] > df[ticker]).astype(int)
    # st.dataframe(df.iloc[-3:][['Tomorrow', 'After_tomorrow', 'y1', 'y2']])
    # df.dropna(inplace=True)

    for horizon in horizons:
        rolling_averages = df.rolling(horizon, 1).mean()

        ratio_column = f"Close_Ratio_{horizon}"
        df[ratio_column] = df[ticker] / rolling_averages[ticker]

        trend_column = f"Trend_{horizon}"
        df[trend_column] = df.shift(1).rolling(horizon, 1).sum()[target]

        new_predictors += [ratio_column, trend_column]

    new_predictors.append('stoch_k')
    # df = df.dropna()
    # st.dataframe(df.iloc[-3:][new_predictors])
    return df, new_predictors


def latest_is_up(dict):
    yesterday = dict.iloc[-2].Close
    today = dict.iloc[-1].Close

    return today > yesterday


with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title('Crypto Currency Price App')

# fill up a dataframe with all dates from 2015 up to today


def get_all_dates():
    start_date = dt(2005, 1, 1)
    end_date = dt.today()
    dates = pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=dates)
    return df

# get google trends data

# @st.cache


def get_trends_data():
    # pytrends = TrendReq(hl='en-US', tz=360)

    df_trend = get_all_dates()
    for ticker_name in ticker_names:
        keyword = [ticker_name]
        pytrend = TrendReq()
        # pytrend.build_payload(kw_list=keyword)

        pytrend.build_payload(kw_list=keyword, cat=7, timeframe='all')
        df_temp = pytrend.interest_over_time()
        df_temp = df_temp.drop(columns=['isPartial'])
        # df_temp.reset_index(inplace=True)
        # st.info(keyword)
        # st.info(f"last date before merge {df_trend.index[-1]}")
        df_trend = df_trend.merge(
            df_temp, how='left', left_index=True, right_index=True)
        # st.info(f"last date after merge {df_trend.index[-1]}")
        df_trend[ticker_name] = df_trend[ticker_name].fillna(method='ffill')/30
        df_trend[ticker_name +
                 '_goog30'] = df_trend[ticker_name].rolling(30, 1).mean()
        df_trend[ticker_name +
                 '_goog90'] = df_trend[ticker_name].rolling(90, 1).mean()
        df_trend[ticker_name +
                 '_goog250'] = df_trend[ticker_name].rolling(250, 1).mean()

    df_trend = df_trend[ticker_names + [ticker_name+'_goog30' for ticker_name in ticker_names] + [ticker_name +
                                                                                                  '_goog90' for ticker_name in ticker_names] + [ticker_name+'_goog250' for ticker_name in ticker_names]]
    return df_trend


# get google trends data from keyword list
if st.button('Refresh'):
    if 'all_tickers' in st.session_state:
        del st.session_state.all_tickers

    if 'df_trend' in st.session_state:
        del st.session_state.df_trends

choice = 'Graph...'
choice = st.sidebar.radio('Vad vill du se', ('Graph...', 'Prognos'), index=0)

if 'all_tickers' not in st.session_state:
    st.session_state.all_tickers = get_all(tickers)

all_tickers = st.session_state.all_tickers

if choice == 'Graph...':
    if 'start_date' not in st.session_state:
        st.session_state.start_date = st.date_input(
            'Start date', datetime.date(2022, 1, 1))
    else:
        st.session_state.start_date = st.date_input(
            'Start date', st.session_state.start_date)

    start_date = st.session_state.start_date

    BTC = all_tickers['BTC-USD'].query('index >= @start_date')
    ETH = all_tickers['ETH-USD'].query('index >= @start_date')
    BCH = all_tickers['BCH-USD'].query('index >= @start_date')
    XRP = all_tickers['XRP-USD'].query('index >= @start_date')
    ZRX = all_tickers['ZRX-USD'].query('index >= @start_date')

    BTC = BTC.rolling(60).mean()
    ETH = ETH.rolling(60).mean()
    BCH = BCH.rolling(60).mean()
    XRP = XRP.rolling(60).mean()
    ZRX = ZRX.rolling(60).mean()

    # compute relative development
    BTC['rel_dev'] = BTC.Close / BTC.Close.shift(1) - 1
    BTC.dropna(inplace=True)
    just = BTC.rel_dev.head(1).values[0]
    BTC.rel_dev -= just

    ETH['rel_dev'] = ETH.Close / ETH.Close.shift(1) - 1
    ETH.dropna(inplace=True)
    just = ETH.rel_dev.head(1).values[0]
    ETH.rel_dev -= just

    BCH['rel_dev'] = BCH.Close / BCH.Close.shift(1) - 1
    BCH.dropna(inplace=True)
    just = BCH.rel_dev.head(1).values[0]
    BCH.rel_dev -= just

    XRP['rel_dev'] = XRP.Close / XRP.Close.shift(1) - 1
    XRP.dropna(inplace=True)
    just = XRP.rel_dev.head(1).values[0]
    XRP.rel_dev -= just

    ZRX['rel_dev'] = ZRX.Close / ZRX.Close.shift(1) - 1
    ZRX.dropna(inplace=True)
    just = ZRX.rel_dev.head(1).values[0]
    ZRX.rel_dev -= just

    fig, ax = plt.subplots()
    ax.plot(BTC.index, BTC.rel_dev, label='BTC')
    ax.plot(ETH.index, ETH['rel_dev'], label='ETH')
    ax.plot(BCH.index, BCH['rel_dev'], label='BCH')
    ax.plot(XRP.index, XRP['rel_dev'], label='XRP')
    ax.plot(ZRX.index, ZRX['rel_dev'], label='ZRX')
    ax.legend()
    ax.set(ylabel="Price US$",
           title='Crypto relativ utveckling med alla startvärden satt till 0')
    # bigger graph sizes
    fig.set_size_inches(14, 12)
    # set theme
    plt.style.use('fivethirtyeight')
    st.pyplot(fig)

# %%
    if 'df_trends' not in st.session_state:
        st.session_state.df_trends = get_trends_data()

    df_trends = st.session_state.df_trends

    fig2, ax2 = plt.subplots()
    df_trends = st.session_state.df_trends.query('index >= @start_date')[
        ['Bitcoin_goog90', 'Ethereum_goog90', 'Ripple_goog90', 'Bitcoin Cash_goog90', '0X_goog90']]
    df_trends.index = pd.to_datetime(df_trends.index)
    ax2.plot(df_trends.index, df_trends['Bitcoin_goog90'], label='BTC')
    ax2.plot(df_trends.index, df_trends['Ethereum_goog90'], label='ETH')
    ax2.plot(df_trends.index, df_trends['Bitcoin Cash_goog90'], label='BCH')
    ax2.plot(df_trends.index, df_trends['Ripple_goog90'], label='XRP')
    ax2.plot(df_trends.index, df_trends['0X_goog90'], label='ZRX')

    ax2.legend()
    ax2.set(ylabel="Antal sök", title='Crypto Google Trends')
    # bigger graph sizes
    fig2.set_size_inches(14, 12)
    # set theme
    plt.style.use('fivethirtyeight')
    st.pyplot(fig2)
    # st.info(df_trends.iloc[-1, :].name)
    exp = st.expander('Crypto Förkortningar')
    exp.write("""BTC = Bitcoin   
              ETH = Ethereum   
              BCH = Bitcoin Cash  
              XRP = Ripple  
              ZRX = 0x   
              """
              )


def load_and_predict(file, data_, predictors):

    data = data_.copy()
    import xgboost 
    xgb = xgboost.XGBClassifier()
    xgb.load_model(file)
    data = data[predictors].dropna()
    # st.write(data.iloc[-5:][predictors])
    # st.dataframe(data.iloc[-1:, :][predictors])
    # st.info(xgb)
    return xgb.predict(data.iloc[-1:, :][predictors])


def add_google_trends(df_, df_trend, ticker, new_predictors):
    df = df_.copy()

    lookup = {'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum',
              'BCH-USD': 'Bitcoin Cash', 'XRP-USD': 'Ripple', 'ZRX-USD': '0X'}
    ticker_namn = lookup[ticker]

    df[ticker_namn + '_goog30'] = df_trend[ticker_namn + '_goog30']
    new_predictors.append(ticker_namn + '_goog30')
    df[ticker_namn + '_goog90'] = df_trend[ticker_namn + '_goog90']
    new_predictors.append(ticker_namn + '_goog90')
    df[ticker_namn + '_goog250'] = df_trend[ticker_namn + '_goog250']
    new_predictors.append(ticker_namn + '_goog250')
    # st.dataframe(df)
    return df, new_predictors


if choice == 'Prognos':
    if 'df_trends' not in st.session_state:
        st.session_state.df_trends = get_trends_data()

    # day name today
    today = datetime.datetime.today().strftime("%A")
    tomorrow = (datetime.date.today() +
                datetime.timedelta(days=1)).strftime("%A")
    day_after = (datetime.date.today() +
                 datetime.timedelta(days=2)).strftime("%A")
    """Priser i US$
    Prognos för i morgon och övermorgon"""

    col1, col2, col3 = st.columns(3)

    col1.markdown('## Bitcoin')
    BTC = all_tickers['BTC-USD']
    dagens = round(BTC.iloc[-1].Close, 1)

    latest = "+ " if latest_is_up(BTC) else "- "
    col1.metric("Dagens pris $", str(dagens), latest)

    BTC_data1, new_predictors = new_features(BTC, 'Close', 'y1')
    BTC_data1, new_predictors = add_google_trends(
        BTC_data1, st.session_state.df_trends, 'BTC-USD', new_predictors)
    # st.info(f"BTC1 gör predict på {BTC_data1.iloc[-1].name}")
    # st.dataframe(BTC_data1.iloc[-3:][new_predictors])
    tomorrow_up = load_and_predict('BTC_y1.json', BTC_data1, new_predictors)
    # st.info(tomorrow_up[0])
    # st.info(f"BTC {BTC_data1.iloc[-1].name}, {BTC_data1.iloc[-1][['Tomorrow','y1']]}")
    BTC_data2, new_predictors = new_features(BTC, 'Close', 'y2')
    BTC_data2, new_predictors = add_google_trends(
        BTC_data2, st.session_state.df_trends, 'BTC-USD', new_predictors)
    # st.info(f"BTC2 gör predict på {BTC_data2.iloc[-1].name}")
    # st.dataframe(BTC_data2.iloc[-3:][new_predictors])
    two_days_upp = load_and_predict('BTC_y2.json', BTC_data2, new_predictors)
    # st.info(two_days_upp[0])
    col1.metric(tomorrow, "", "+ " if tomorrow_up[0] else "- ")
    col1.metric(day_after, "", "+ " if two_days_upp[0] else "- ")
    # st.info(f'{tomorrow_up}  {two_days_upp}')

    col2.markdown('## Ether')
    ETH = all_tickers['ETH-USD']
    dagens = round(ETH.iloc[-1].Close, 1)
    latest = "+ " if latest_is_up(ETH) else "- "
    col2.metric("Dagens pris $", str(dagens), latest)

    ETH_data1, new_predictors = new_features(ETH, 'Close', 'y1')
    ETH_data1, new_predictors = add_google_trends(
        ETH_data1, st.session_state.df_trends, 'ETH-USD', new_predictors)

    tomorrow_up = load_and_predict('ETH_y1.json', ETH_data1, new_predictors)
    ETH_data2, new_predictors = new_features(ETH, 'Close', 'y2')
    ETH_data2, new_predictors = add_google_trends(
        ETH_data2, st.session_state.df_trends, 'ETH-USD', new_predictors)

    two_days_upp = load_and_predict('ETH_y2.json', ETH_data2, new_predictors)
    col2.metric(tomorrow, "", "+ " if tomorrow_up else "- ")
    col2.metric(day_after, "", "+ " if two_days_upp else "- ")
    # st.info(f'{tomorrow_up}  {two_days_upp}')

    col3.markdown('## BCH')
    BCH = all_tickers['BCH-USD']
    dagens = round(BCH.iloc[-1].Close, 2)
    latest = "+ " if latest_is_up(BCH) else "- "
    col3.metric("Dagens pris $", str(dagens), latest)

    BCH_data1, new_predictors = new_features(BCH, 'Close', 'y1')
    BCH_data1, new_predictors = add_google_trends(
        BCH_data1, st.session_state.df_trends, 'BCH-USD', new_predictors)

    tomorrow_up = load_and_predict('BCH_y1.json', BCH_data1, new_predictors)
    BCH_data2, new_predictors = new_features(BCH, 'Close', 'y2')
    BCH_data2, new_predictors = add_google_trends(
        BCH_data2, st.session_state.df_trends, 'BCH-USD', new_predictors)

    two_days_upp = load_and_predict('BCH_y2.json', BCH_data2, new_predictors)
    col3.metric(tomorrow, "", "+ " if tomorrow_up else "- ")
    col3.metric(day_after, "", "+ " if two_days_upp else "- ")
    # st.info(f'{tomorrow_up}  {two_days_upp}')

    col4, col5, col6 = st.columns(3)
    col4.markdown('## 0x')
    ZRX = all_tickers['ZRX-USD']
    dagens = round(ZRX.iloc[-1].Close, 3)
    latest = "+ " if latest_is_up(ZRX) else "- "
    col4.metric("Dagens pris $", str(dagens), latest)

    ZRX_data1, new_predictors = new_features(ZRX, 'Close', 'y1')
    ZRX_data1, new_predictors = add_google_trends(
        ZRX_data1, st.session_state.df_trends, 'ZRX-USD', new_predictors)

    tomorrow_up = load_and_predict('ZRX_y1.json', ZRX_data1, new_predictors)
    ZRX_data2, new_predictors = new_features(ZRX, 'Close', 'y2')
    ZRX_data2, new_predictors = add_google_trends(
        ZRX_data2, st.session_state.df_trends, 'ZRX-USD', new_predictors)

    two_days_upp = load_and_predict('ZRX_y2.json', ZRX_data2, new_predictors)
    col4.metric(tomorrow, "", "+ " if tomorrow_up else "- ")
    col4.metric(day_after, "", "+ " if two_days_upp else "- ")
    # st.info(f'{tomorrow_up}  {two_days_upp}')

    col5.markdown('## Ripple')
    XRP = all_tickers['XRP-USD']
    dagens = round(XRP.iloc[-1].Close, 3)
    latest = "+ " if latest_is_up(XRP) else "- "
    col5.metric("Dagens pris $", str(dagens), latest)

    XRP_data1, new_predictors = new_features(XRP, 'Close', 'y1')
    XRP_data1, new_predictors = add_google_trends(
        XRP_data1, st.session_state.df_trends, 'XRP-USD', new_predictors)

    tomorrow_up = load_and_predict('XRP_y1.json', XRP_data1, new_predictors)
    XRP_data2, new_predictors = new_features(XRP, 'Close', 'y2')
    XRP_data2, new_predictors = add_google_trends(
        XRP_data2, st.session_state.df_trends, 'XRP-USD', new_predictors)

    two_days_upp = load_and_predict('XRP_y2.json', XRP_data2, new_predictors)

    col5.metric(tomorrow, "", "+ " if tomorrow_up else "- ")
    col5.metric(day_after, "", "+ " if two_days_upp else "- ")
    # st.info(f'{tomorrow_up}  {two_days_upp}')

    col6.markdown(""" """)

    exp = st.expander('Crypto Förkortningar')
    exp.write("""BTC = Bitcoin   
              ETH = Ethereum   
              BCH = Bitcoin Cash  
              XRP = Ripple  
              ZRX = 0x   
              """
              )
