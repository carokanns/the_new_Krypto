import matplotlib.pyplot as plt
from pytrends.request import TrendReq
import streamlit as st
from datetime import datetime as dt
import datetime
import pandas as pd
import ta
import yfinance as yf
import xgboost as xgb

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


def new_features(df_, ticker, target, horizons=[2, 5, 60, 250], extra=[]):
    df = df_.copy()
    # tidsintervall i dagar för rullande medelvärden
    # skulle helst ha med upp till 4 år men ETH har för få värden

    new_predictors = []
    if 'stoch_k' in extra:
        df['stoch_k'] = ta.momentum.stochrsi_k(df[ticker], window=10)
        new_predictors += ['stoch_k']
        
    # Target
    # tomorrow's close price - alltså nästa dag
    df['Tomorrow'] = df[ticker].shift(-1)
    # after tomorrow's close price - alltså om två dagar
    df['After_tomorrow'] = df[ticker].shift(-2)
    df['y1'] = (df['Tomorrow'] > df[ticker]).astype(int)
    df['y2'] = (df['After_tomorrow'] > df[ticker]).astype(int)
    # df.dropna(inplace=True)

    for horizon in horizons:
        rolling_averages = df.rolling(horizon, min_periods=1).mean()

        ratio_column = f"Close_Ratio_{horizon}"
        df[ratio_column] = df[ticker] / rolling_averages[ticker]

        # edit_column = f"edit_{horizon}"
        # df[edit_column] = rolling_averages['edit_count']
        ## Den ovan ändras till min Google Trends och görs på annan plats

        rolling = df.rolling(horizon, closed='left', min_periods=1).mean()

        trend_column = f"Trend_{horizon}"
        target_name = 'Tomorrow' if target == 'y1' else 'After_tomorrow'
        # OBS! Skilj trend_column från Google Trends
        df[trend_column] = rolling[target_name]

        new_predictors += [ratio_column, trend_column]
        
    if 'month' in extra:
        df['month'] = df.index.month
        new_predictors += ['month']
    if 'day_of_month' in extra:
        df['day_of_month'] = df.index.day
        new_predictors += ['day_of_month']
    if 'day_of_week' in extra:
        df['day_of_week'] = df.index.dayofweek
        new_predictors += ['day_of_week']  
    
    # df = df.dropna()
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



choice = 'Graph...'
choice = st.sidebar.radio('what do you want to see',
                          ('Graph...', 'Price forecasts'), index=0)

if st.sidebar.button(f'Refresh Data'):
    if 'all_tickers' in st.session_state:
        del st.session_state.all_tickers

    if 'df_trend' in st.session_state:
        del st.session_state.df_trends



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

    BTC = BTC.rolling(30).mean()
    ETH = ETH.rolling(30).mean()
    BCH = BCH.rolling(30).mean()
    XRP = XRP.rolling(30).mean()
    ZRX = ZRX.rolling(30).mean()

    # compute relative development
    def rel_dev(df_ticker_):
        df_ticker = df_ticker_.copy()
        df_ticker = df_ticker/df_ticker.shift(1)-1
        df_ticker = df_ticker.dropna()
        just = df_ticker.head(1).values[0]
        df_ticker -= just
        return df_ticker
           
    BTC['rel_dev'] = rel_dev(BTC.Close)
    ETH['rel_dev'] = rel_dev(ETH.Close)
    BCH['rel_dev'] = rel_dev(BCH.Close)
    XRP['rel_dev'] = rel_dev(XRP.Close)
    ZRX['rel_dev'] = rel_dev(ZRX.Close)
    

    # ETH['rel_dev'] = ETH.Close / ETH.Close.shift(1) - 1
    # ETH.dropna(inplace=True)
    # just = ETH.rel_dev.head(1).values[0]
    # ETH.rel_dev -= just

    # BCH['rel_dev'] = BCH.Close / BCH.Close.shift(1) - 1
    # BCH.dropna(inplace=True)
    # just = BCH.rel_dev.head(1).values[0]
    # BCH.rel_dev -= just

    # XRP['rel_dev'] = XRP.Close / XRP.Close.shift(1) - 1
    # XRP.dropna(inplace=True)
    # just = XRP.rel_dev.head(1).values[0]
    # XRP.rel_dev -= just

    # ZRX['rel_dev'] = ZRX.Close / ZRX.Close.shift(1) - 1
    # ZRX.dropna(inplace=True)
    # just = ZRX.rel_dev.head(1).values[0]
    # ZRX.rel_dev -= just

    last_date = BTC.index[-1].date()
    st.info(f'Until {last_date}')

    # fig, ax = plt.subplots()
    # ax.plot(BTC.index, BTC.rel_dev, label='BTC')
    # ax.plot(ETH.index, ETH['rel_dev'], label='ETH')
    # ax.plot(BCH.index, BCH['rel_dev'], label='BCH')
    # ax.plot(XRP.index, XRP['rel_dev'], label='XRP')
    # ax.plot(ZRX.index, ZRX['rel_dev'], label='ZRX')
    # ax.legend()

    # ax.set(ylabel="Price US$",
    #        title=f'Crypto relative deveolpment starting from 0')
    # ax.title.set_fontsize(28)
    # ax.tick_params(axis='both', which='major', labelsize=18)
    
    # fig.set_size_inches(14, 12)
    # # set theme
    # plt.style.use('fivethirtyeight')
    # st.pyplot(fig)
    
    import plotly.express as px
    # comb = pd.concat([BTC, ETH, BCH, ZRX, XRP], axis=1,)
    # BTC = BTC.rel_dev.dropna()
    ETH.rel_dev.dropna(inplace=True)
    BCH.rel_dev.dropna(inplace=True)
    XRP.rel_dev.dropna(inplace=True)
    ZRX.rel_dev.dropna(inplace=True)

    fig = px.line(BTC, x=BTC.index, y=BTC.rel_dev, title=f'Crypto relative deveolpment starting from 0')
        
    fig.add_scatter(x=BTC.index, y=BTC.rel_dev, name='BTC')
    fig.add_scatter(x=ETH.index, y=ETH.rel_dev, name='ETH')
    fig.add_scatter(x=BCH.index, y=BCH.rel_dev, name='BCH')
    fig.add_scatter(x=XRP.index, y=XRP.rel_dev, name='XRP')
    fig.add_scatter(x=ZRX.index, y=ZRX.rel_dev, name='ZRX')
    
    fig.update_xaxes(title_text='Date', title_font_size=20)
    graph_range = [BTC.dropna().index[0], BTC.index[-1]]
    fig.update_xaxes(range=graph_range)
    fig.update_yaxes(title_text='Relative development', title_font_size=20)
    
    fig.update_layout(
        autosize=False,
        font_color="black",
        title_text=f"Relative development",
        width=900,
        height=400,
        font=dict(size=18),
        legend=dict(font=dict(color="black")),
        margin=dict(
            l=0,
            r=50,
            b=10,
            t=40,
            pad=10
        ),
        paper_bgcolor="LightSteelBlue",
        plot_bgcolor='DarkBlue',
    )
    st.plotly_chart(fig, use_container_width=True)
    
# %%
    if 'df_trends' not in st.session_state:
        st.session_state.df_trends = get_trends_data()

    df_trends = st.session_state.df_trends

    
    df_trends = st.session_state.df_trends.query('index >= @start_date')[
        ['Bitcoin_goog90', 'Ethereum_goog90', 'Ripple_goog90', 'Bitcoin Cash_goog90', '0X_goog90']]
    df_trends.index = pd.to_datetime(df_trends.index)
    
    fig2 = px.line(df_trends, x=df_trends.index, y=df_trends['Bitcoin_goog90'], title=f'Crypto Google Trends until {last_date}')

    fig2.add_scatter(x=df_trends.index, y=df_trends['Bitcoin_goog90'], name='BTC')
    fig2.add_scatter(x=df_trends.index, y=df_trends['Ethereum_goog90'], name='ETH')
    fig2.add_scatter(x=df_trends.index, y=df_trends['Bitcoin Cash_goog90'], name='BCH')
    fig2.add_scatter(x=df_trends.index, y=df_trends['Ripple_goog90'], name='XRP')
    fig2.add_scatter(x=df_trends.index, y=df_trends['0X_goog90'], name='ZRX')
    
    fig2.update_xaxes(title_text='Date', title_font_size=20)
    fig2.update_xaxes(range=graph_range)
    fig2.update_yaxes(title_text='Traffic', title_font_size=20)

    fig2.update_layout(
        autosize=False,
        font_color="black",
        title_text=f"Google Trends",
        width=900,
        height=400,
        font=dict(size=18),
        legend=dict(font=dict(color="black")),
        margin=dict(
            l=0,
            r=50,
            b=10,
            t=40,
            pad=10
        ),
        paper_bgcolor="LightSteelBlue",
        plot_bgcolor='DarkBlue',
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # fig2, ax2 = plt.subplots()
    # ax2.plot(df_trends.index, df_trends['Bitcoin_goog90'], label='BTC')
    # ax2.plot(df_trends.index, df_trends['Ethereum_goog90'], label='ETH')
    # ax2.plot(df_trends.index, df_trends['Bitcoin Cash_goog90'], label='BCH')
    # ax2.plot(df_trends.index, df_trends['Ripple_goog90'], label='XRP')
    # ax2.plot(df_trends.index, df_trends['0X_goog90'], label='ZRX')

    # background color
    # ax2.set_facecolor('darkblue')
    # ax2.legend()
    # last_date = df_trends.index[-1].date()
    # ax2.set(ylabel="Trafic",
    #         title=f'Crypto Google Trends until {last_date}')
    # ax2.title.set_fontsize(28)
    # ax2.tick_params(axis='both', which='major', labelsize=18)
    
    # fig2.set_size_inches(14, 12)
    # # set theme
    # plt.style.use('fivethirtyeight')
    # st.pyplot(fig2)
    # # st.info(df_trends.iloc[-1, :].name)

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
    model = xgb.XGBClassifier()
    model.load_model(file)
    data = data[predictors].dropna()
    return model.predict(data.iloc[-1:, :][predictors])


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


if choice == 'Price forecasts':
    if 'df_trends' not in st.session_state:
        st.session_state.df_trends = get_trends_data()

    horizons = [2,5,15,30,60,90,250]
    extra = ['day_of_week', 'day_of_month']  # skippar 'month och stoch_k
    
    
    # day names
    today = datetime.datetime.today().strftime("%A")
    tomorrow = (datetime.date.today() +
                datetime.timedelta(days=1)).strftime("%A")
    day_after = (datetime.date.today() +
                 datetime.timedelta(days=2)).strftime("%A")
    last_date = all_tickers['BTC-USD'].index[-1].date()
    
    st.info(
        f'{last_date}\n- Todays prices in US$ and if it went up or down compared to yesterday\n - Predictions for {tomorrow} and {day_after}')
    
    def make_predictions(ticker, col, extra, r=1):
        ticker_df = all_tickers[ticker]
        dagens = round(ticker_df.iloc[-1].Close ,r)
        latest = latest = "+ " if latest_is_up(ticker_df) else "- "
        col.metric("Aktuellt pris $", str(dagens), latest)
        ticker_data1, new_predictors = new_features(ticker_df, 'Close', 'y1', horizons=horizons,extra=extra)
        ticker_data1, new_predictors = add_google_trends(ticker_data1, st.session_state.df_trends, ticker, new_predictors)
        ticker_short= ticker[:3]
        tomorrow_up = load_and_predict(ticker_short+'_y1.json', ticker_data1, new_predictors)
        
        ticker_data2, new_predictors = new_features(ticker_df, 'Close', 'y2', horizons=horizons,extra=extra)
        ticker_data2, new_predictors = add_google_trends(ticker_data2, st.session_state.df_trends, ticker, new_predictors)
        
        two_days_upp = load_and_predict(ticker_short+'_y2.json', ticker_data2, new_predictors)
        col.metric(tomorrow, "", "+ " if tomorrow_up[0] else "- ")
        col.metric(day_after, "", "+ " if two_days_upp[0] else "- ")
        
        
    col1, col2, col3 = st.columns(3)

    col1.markdown('## Bitcoin')
    make_predictions('BTC-USD', col1, extra)
    
    col2.markdown('## Ether')
    make_predictions('ETH-USD', col2, extra)
    
    col3.markdown('## BCH')
    make_predictions('BCH-USD', col3, extra, r=2)
    
    col4, col5, col6 = st.columns(3)
    
    col4.markdown('## 0x')
    make_predictions('ZRX-USD', col4, extra, r=3)
    
    col5.markdown('## Ripple')
    make_predictions('XRP-USD', col5, extra, r=3)

    col6.markdown(""" """)

    exp = st.expander('Crypto Förkortningar')
    exp.write("""BTC = Bitcoin   
              ETH = Ethereum   
              BCH = Bitcoin Cash  
              XRP = Ripple  
              ZRX = 0x   
              """
              )
