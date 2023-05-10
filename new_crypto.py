import streamlit as st
from datetime import datetime as dt
import datetime
from datetime import timedelta
import numpy as np 
import pandas as pd
import ta
import yfinance as yf
import xgboost as xgb
from catboost import CatBoostClassifier
import plotly.express as px

tickers = ['BTC-USD', 'ETH-USD', 'BCH-USD', 'ZRX-USD', 'XRP-USD']
ticker_names = ['Bitcoin', 'Ethereum', 'Bitcoin Cash', '0X', 'Ripple']


def add_horizon_columns(inflation, horizons):
    # print(horizons)
    for horizon in horizons:
        # print(horizon)
        inflation['US_inflation_' +
                  str(horizon)] = inflation['US_inflation'].rolling(horizon, 1).mean()
        inflation['SE_inflation_' +
                  str(horizon)] = inflation['SE_inflation'].rolling(horizon, 1).mean()

        # print(inflation.columns)
    return inflation


def initiate_data(inflation, df_dates, lang_dict, value_name):
    # display(inflation)
    inflation = inflation.melt(
        id_vars=['Year'], var_name='month', value_name=value_name)

    # use lang_dict to translate month names to numbers
    inflation['month'] = inflation['month'].map(lang_dict)

    inflation['date'] = pd.to_datetime(inflation['Year'].astype(
        str) + '-' + inflation['month'].astype(str))
    inflation.set_index('date', inplace=True)
    inflation.drop(['Year', 'month'], axis=1, inplace=True)
    inflation = df_dates.merge(
        inflation, how='left', left_on='date', right_index=True)
    inflation.set_index('date', inplace=True)
    inflation[value_name] = inflation[value_name].astype(str)
    inflation[value_name] = inflation[value_name].str.replace(',', '.')
    inflation[value_name] = inflation[value_name].str.replace(
        chr(8209), chr(45))
    inflation[value_name] = inflation[value_name].astype(float)
    inflation[value_name].interpolate(method='linear', inplace=True)
    return inflation


def get_inflation_data():
    # Explain this function here

    df_dates = pd.DataFrame(pd.date_range(
        '1988-12-01', pd.to_datetime('today').date()), columns=['date'])

    US_inflation = pd.read_html(
        'https://www.usinflationcalculator.com/inflation/current-inflation-rates/')
    US_inflation = US_inflation[0]
    # replace the cell including string starting with "Avail" with the NaN
    US_inflation.replace(to_replace=r'^Avail.*$',
                         value=np.nan, regex=True, inplace=True)
    # set the first row as the header and drop the first row
    US_inflation.columns = US_inflation.iloc[0]
    US_inflation.drop(US_inflation.index[0], inplace=True)
    US_inflation.drop('Ave', axis=1, inplace=True)

    SE_inflation = pd.read_html(
        'https://www.scb.se/hitta-statistik/statistik-efter-amne/priser-och-konsumtion/konsumentprisindex/konsumentprisindex-kpi/pong/tabell-och-diagram/konsumentprisindex-med-fast-ranta-kpif-och-kpif-xe/kpif-12-manadersforandring/')
    SE_inflation = SE_inflation[0]
    SE_inflation.rename(columns={'År': 'Year'}, inplace=True)

    se_dict = dict(Jan='1', Feb='2', Mar='3', Apr='4', Maj='5', Jun='6',
                   Jul='7', Aug='8', Sep='9', Okt='10', Nov='11', Dec='12')
    us_dict = dict(Jan='1', Feb='2', Mar='3', Apr='4', May='5', Jun='6',
                   Jul='7', Aug='8', Sep='9', Oct='10', Nov='11', Dec='12')
    #
    SE_inflation = initiate_data(
        SE_inflation, df_dates, se_dict, value_name='SE_inflation')
    # SE_inflation is in percent, divide by 10 to get decimal
    SE_inflation['SE_inflation'] = SE_inflation['SE_inflation'] / 10
    US_inflation = initiate_data(
        US_inflation, df_dates, us_dict,  value_name='US_inflation')

    # concat and set one column to US_index and the other to SE_index
    inflations = pd.concat([US_inflation, SE_inflation], axis=1)
    inflations = inflations.dropna()
    inflations = add_horizon_columns(inflations, [75, 90, 250])
    return inflations

def get_ticker_data(ticker, start="1900-01-01", end=dt.today()):
    data = yf.download(ticker, start=None, end=None)
    # data.set_index(inplace=True)
    # data.index = pd.to_datetime(data.Date)
    # data = data.drop("Date", axis=1)

    return data


def get_all_data(tickers):
    all_tickers = {}
    for enum, ticker in enumerate(tickers):
        all_tickers[ticker] = get_ticker_data(ticker)
    df_inflations = get_inflation_data()

    return all_tickers, df_inflations


def add_predictors(df_, ticker, target, horizons=[2, 5, 60, 250], extra=[]):
    
    df = df_.copy()
    ticker_name = ticker.split('-')[0]
    # skulle helst ha med rolling upp till 4 år men de har har för få värden

    predictors = []
    if 'stoch_k' in extra:
        df['stoch_k'] = ta.momentum.stochrsi_k(df[ticker], window=10) # type: ignore
        predictors += ['stoch_k']
    
    if 'ETH_BTC' in extra:
        df['ETH_BTC_ratio'] = df['ETH-USD']/df['BTC-USD']
        predictors += ['ETH_BTC_ratio']

        df['ETH_BTC_lag1'] = df['ETH_BTC_ratio'].shift(1)
        predictors += ['ETH_BTC_lag1']

        df['ETH_BTC_lag2'] = df['ETH_BTC_ratio'].shift(2)
        predictors += ['ETH_BTC_lag2']

        if ticker not in ['BTC-USD', 'ETH-USD']:
            df[ticker_name+'_BTC_ratio'] = df[ticker]/df['BTC-USD']
            predictors += [ticker_name+'_BTC_ratio']

            df[ticker_name+'_BTC_lag1'] = df[ticker_name+'_BTC_ratio'].shift(1)
            predictors += [ticker_name+'_BTC_lag1']

            df[ticker_name+'_BTC_lag2'] = df[ticker_name+'_BTC_ratio'].shift(2)
            predictors += [ticker_name+'_BTC_lag2']

            df[ticker_name+'_ETH_ratio'] = df[ticker]/df['ETH-USD']
            predictors += [ticker_name+'_ETH_ratio']

            df[ticker_name+'_ETH_lag1'] = df[ticker_name+'_ETH_ratio'].shift(1)
            predictors += [ticker_name+'_ETH_lag1']

            df[ticker_name+'_ETH_lag2'] = df[ticker_name+'_ETH_ratio'].shift(2)
            predictors += [ticker_name+'_ETH_lag2']

    
    #### Target ####
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

        rolling = df.rolling(horizon, closed='left', min_periods=1).mean()

        trend_column = f"Trend_{horizon}"
        target_name = 'Tomorrow' if target == 'y1' else 'After_tomorrow'
        # OBS! Skilj trend_column från Google Trends
        df[trend_column] = rolling[target_name]

        predictors += [ratio_column, trend_column]
        
    if 'month' in extra:
        df['month'] = df.index.month
        predictors += ['month']
    if 'day_of_month' in extra:
        df['day_of_month'] = df.index.day
        predictors += ['day_of_month']
    if 'day_of_week' in extra:
        df['day_of_week'] = df.index.dayofweek
        predictors += ['day_of_week']  
    
    # df = df.dropna()
    return df, predictors


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

def get_inflations_data():
   return None

# choice = 'Price forecasts'
choice = st.sidebar.radio('what do you want to see',
                          ('Graph...', 'Price forecasts'), index=1)

if st.sidebar.button(f'Refresh Data'):
    if 'all_data' in st.session_state:
        del st.session_state.all_data

    if 'df_inflations' in st.session_state:
        del st.session_state.df_inflations


if 'all_data' not in st.session_state:
    st.session_state.all_data, st.session_state.df_inflations = get_all_data(tickers)

all_data = st.session_state.all_data
df = pd.concat([all_data, df_inflations], axis=1) # type: ignore

if choice == 'Graph...':
     
    # make start date at least 31 days before todays date
    max_date = (dt.today() - timedelta(days=31)).date()
    
    if 'start_date' not in st.session_state:
        st.session_state.start_date = st.date_input(
            f'Start date', datetime.date(2022, 1, 1), min_value=datetime.date(2014, 9, 17), max_value=max_date)
    else:
        st.session_state.start_date = st.date_input(
            f'Start date', st.session_state.start_date, min_value=datetime.date(2014, 9, 17), max_value=max_date)

    start_date = st.session_state.start_date
    days = (dt.today().date() - start_date).days # type: ignore
    
    BTC = all_data['BTC-USD'].query('index >= @start_date')
    ETH = all_data['ETH-USD'].query('index >= @start_date')
    BCH = all_data['BCH-USD'].query('index >= @start_date')
    XRP = all_data['XRP-USD'].query('index >= @start_date')
    ZRX = all_data['ZRX-USD'].query('index >= @start_date')

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


    last_date = BTC.index[-1].date()
    st.info(f'Until {last_date}')
    
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
    

def load_xgb_and_predict(file, data_, predictors):
    print('predictors:\n', predictors, len(predictors))
    data = data_.copy()
    model = xgb.XGBClassifier()
    model.load_model(file)
    predictors_real =model.get_booster().feature_names
    print('predictors_real:\n', predictors_real, len(predictors_real)) # type: ignore
    data = data[predictors].dropna()
    return model.predict(data.iloc[-1:, :][predictors])

def load_cat_and_predict(file, data_, predictors):
    print('predictors:\n', predictors, len(predictors))
    data = data_.copy()
    model = CatBoostClassifier()

    model.load_model(file)
    predictors_real = model.feature_names_

    print('predictors_real:\n', predictors_real, len(predictors_real)) # type: ignore
    data = data[predictors].dropna()
    return model.predict(data.iloc[-1:, :][predictors])


if choice == 'Price forecasts':
    if 'df_inflations' not in st.session_state:
        st.session_state.df_inflations = get_inflations_data()

    horizons = [2,5,15,30,60,90,250]
    extra = [] #['day_of_week', 'day_of_month']  # skippar 'month och stoch_k och alla ETH-BTC-grejor
    
    
    # day names
    today = datetime.datetime.today().strftime("%A")
    tomorrow = (datetime.date.today() +
                datetime.timedelta(days=1)).strftime("%A")
    day_after = (datetime.date.today() +
                 datetime.timedelta(days=2)).strftime("%A")
    last_date = all_data['BTC-USD'].index[-1].date()
    
    st.info(
        f'{last_date}\n- Todays prices in US$ and if it went up or down compared to yesterday\n - Predictions for {tomorrow} and {day_after}')
    
    def make_predictions(ticker, col, extra, modeltype='xgb',r=1):
        ticker_df = all_data[ticker]
        dagens = round(ticker_df.iloc[-1].Close ,r)
        latest = latest = "+ " if latest_is_up(ticker_df) else "- "
        col.metric("Aktuellt pris $", str(dagens), latest)
        
        ticker_data1, predictors = add_predictors(ticker_df, 'Close', 'y1', horizons=horizons,extra=extra)
        print('predictors:\n', predictors, len(predictors))
 
        # read the file 'predictors.txt' and store the content as a list of strings
        with open('predictors.txt', 'r') as file:
            org_predictors = file.readline().split()
        print('org_predictors:\n', org_predictors, len(org_predictors))
        
        ticker_short= ticker[:3]
        
        if modeltype == 'xgb':
            tomorrow_up = load_xgb_and_predict('xgb_'+ticker_short+'_y1.json', ticker_data1, predictors)
        elif modeltype == 'cat':
            tomorrow_up = load_cat_and_predict('cat_'+ticker_short+'_y1.json', ticker_data1, predictors)
        else:
            raise ValueError('modeltype must be xgb or lgbm')
        ticker_data2, predictors = add_predictors(
            ticker_df, 'Close', 'y2', horizons=horizons, extra=extra)
        two_days_upp = load_xgb_and_predict('xgb_'+ticker_short+'_y2.json', ticker_data2, predictors)
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
