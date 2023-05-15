#!/usr/bin/env python
# coding: utf-8


#%% 
# Komplett omtag av new_crypto.py
# använd ML-modell från my_test_modeller.ipynb, scalers från scalers-foldern

# TODO  Publicera en första version 

# TODO: Flera sidor? På andra sidan enbart prognoser för 6 valda

# TODO: Slutligen: Byt namn till new_crypto.py igen innan publicering
# TODO: skapa requrements.txt ta hjälp av ChatGPT
# TODO: merge gridSearch branchen into master
#
# TODO: Skapa ett helt nytt program men med Autogluon och streamlit som gör precis samma sak som detta program
#%%

import pandas as pd
import numpy as np
import streamlit as st
from pandas.tseries.offsets import DateOffset
from binance.client import Client
import yfinance as yf
import preprocess as pp
import pickle
from datetime import datetime as dt

@st.cache_data
def get_gold_data():
    df_dates = pd.DataFrame(pd.date_range(
        '1988-12-01', pd.to_datetime('today').date()), columns=['Date'])
    df_dates.set_index('Date', inplace=True)
    # Hämta historiska guldprisdata (GLD är ticker-symbolen för SPDR Gold Shares ETF)
    gld_data = yf.download('GLD', end=dt.today().date(), progress=False)
    # gld_data.set_index('Date', inplace=True)

    # Behåll endast 'Close' priser och döp om kolumnen till 'GLDUSDT'
    gld_data = gld_data[['Close']].rename(columns={'Close': 'GLD-USD'})

    df_dates = pd.DataFrame(pd.date_range(start=gld_data.index[0], end=pd.to_datetime(  # type: ignore
        'today').date(), freq='D'), columns=['Date'])  # type: ignore

    df_dates.set_index('Date', inplace=True)
    gld_data = df_dates.merge(gld_data, how='left',
                              left_on='Date', right_index=True)
    # interpolating missing values
    gld_data.interpolate(method='linear', inplace=True)
    return gld_data
df_gold = get_gold_data()

# ## Inflation
def add_horizon_columns(inflation, horizons):
    for horizon in horizons:
        inflation['US_inflation_'+str(horizon)] = inflation['US_inflation'].rolling(horizon, 1).mean()
        inflation['SE_inflation_'+str(horizon)] = inflation['SE_inflation'].rolling(horizon, 1).mean()
    return inflation


def initiate_data(inflation, df_dates, lang_dict, value_name):
    inflation = inflation.melt(id_vars=['Year'], var_name='month', value_name=value_name)
    inflation['month'] = inflation['month'].map(lang_dict)
    inflation['date'] = pd.to_datetime(inflation['Year'].astype(str) + '-' + inflation['month'].astype(str))
    inflation.set_index('date', inplace=True)
    inflation.drop(['Year', 'month'], axis=1, inplace=True)
    inflation = df_dates.merge(inflation, how='left', left_on='date', right_index=True)
    inflation.set_index('date', inplace=True)
    inflation[value_name] = inflation[value_name].astype(str)
    inflation[value_name] = inflation[value_name].str.replace(',', '.')
    inflation[value_name] = inflation[value_name].str.replace(chr(8209), chr(45))
    inflation[value_name] = inflation[value_name].astype(float)
    inflation[value_name].interpolate(method='linear', inplace=True)
    return inflation


#%%

@st.cache_data
def get_inflation_data():
    df_dates = pd.DataFrame(pd.date_range(
        '1988-12-01', pd.to_datetime('today').date()), columns=['date'])

    US_inflation = pd.read_html(
        'https://www.usinflationcalculator.com/inflation/current-inflation-rates/')
    US_inflation = US_inflation[0]
    US_inflation.replace(to_replace=r'^Avail.*$',
                         value=np.nan, regex=True, inplace=True)
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

    SE_inflation = initiate_data(SE_inflation, df_dates, se_dict, value_name='SE_inflation')
    SE_inflation['SE_inflation'] = SE_inflation['SE_inflation'] / 10

    US_inflation = initiate_data(US_inflation, df_dates, us_dict, value_name='US_inflation')

    inflations = pd.concat([US_inflation, SE_inflation], axis=1)
    inflations = inflations.dropna()
    # inflations = add_horizon_columns(inflations, [75, 90, 250])
    return inflations

inflation = get_inflation_data()

st.title('Performance of Crypto currencies')
MAX_MONTHS = 24

@st.cache_data
def get_data():  # från Binance

    api_key = '2jxiCQ8OIWmU4PZH4xfwKEY9KYerDkSWzwNCqoaMzj41eJgWBsSqA3VYqkt2wmdX'
    api_secret = 'YY1Qj1t0JZrE4tQdaBBxT8iwl2tbFalWp1FHjyZ9selBb6OnQ0Oj8aVdiXO7YLMz'

    client = Client(api_key, api_secret)

    # Hämta handelspar
    symbols = client.get_all_tickers()
    symbols = [symbol for symbol in symbols if symbol['symbol'].endswith('USDT')]

    # Sätt upp en tom lista för att lagra close-priser
    close_prices = {}

    # Ange den tidsram du vill ha för historiska data
    interval = Client.KLINE_INTERVAL_1DAY
    start_time = f"{MAX_MONTHS} month ago UTC"

    # Hämta close-priser för alla kryptovalutor
    dates = None

    progress_bar = st.progress(0)  # Create a progress bar

    for idx, symbol in enumerate(symbols):
        try:
            klines = client.get_historical_klines(
                symbol['symbol'], interval, start_time)
            if dates is None:
                # Extrahera och konvertera tidsstämplar till datum
                dates = [dt.fromtimestamp(
                    int(kline[0]) / 1000).strftime('%Y-%m-%d') for kline in klines]
            close_prices[symbol['symbol']] = [
                float(kline[4]) for kline in klines]
        except Exception as e:
            print(f"Kunde inte hämta data för {symbol['symbol']}: {e}")

        # Update the progress bar
        if idx+1 == len(symbols):
            progress_text = f"Done! last symbol . . . . . . . . {symbol['symbol']} fetched."
        else:
            progress_text = f"This will take several minutes. Symbol number {idx+1} of {len(symbols)}: . . . . . . . . {symbol['symbol']}"
        progress_bar.progress((idx + 1) / len(symbols), progress_text)
        
    # Konvertera close_prices-dikten till en pandas DataFrame
    df = pd.DataFrame.from_dict(close_prices, orient='index').transpose()
    # Lägg till datum som index för DataFrame
    # type: ignore
    df.index = pd.to_datetime(dates)  # type: ignore
    print(df.head())

    return df

# hämta yf_tickers från yfinance
@st.cache_data
def get_yf_data(tickers, time_period='2y'):
    # Hämta historiska data från yfinance
    # yf_data = yf.download(tickers, start='2019-01-01', end=dt.today().date(), progress=False)
    yf_data = yf.download(tickers, interval='1d',
                       period=time_period, group_by='ticker', auto_adjust=True, progress=True)

    df_cur = pd.DataFrame(yf_data.xs('Close', axis=1, level=1)) 
    df_vol = pd.DataFrame(yf_data.xs('Volume', axis=1, level=1)) 
   
    return df_cur, df_vol


@st.cache_data
def read_ticker_names(filenam):
    with open(filenam, 'r') as f:
        ticker_names = f.read().splitlines()
    print(f'{len(ticker_names)} yFinance ticker_names')
    return ticker_names

@st.cache_data
def get_predictions(df_curr_, df_vol_, df_gold, df_infl):
    predictors = ['Close','Ratio_2', 'Trend_2', 'Ratio_5', 'Trend_5', 'Ratio_30', 'Trend_30', 'Ratio_60', 'Trend_60', 'Ratio_90', 'Trend_90',
                  'Ratio_250', 'Trend_250', 'GLD-USD', 'GLD_Ratio_2', 'GLD_Ratio_5', 'GLD_Ratio_30', 'GLD_Ratio_60', 'GLD_Ratio_90', 'GLD_Ratio_250',
                  'Volume', 'vol_Ratio_2', 'vol_Ratio_5', 'vol_Ratio_30', 'vol_Ratio_60', 'vol_Ratio_90', 'vol_Ratio_250',
                  'US_inflation', 'infl_Ratio_75', 'infl_Ratio_90', 'infl_Ratio_250', 'diff']
    
    df_curr = pp.preprocessing_currency(df_curr_)
    df_vol = pp.preprocessing_currency(df_vol_)
    my_model = pickle.load(open("my_model.pkl", "rb"))
    
    predictions = pd.DataFrame(columns=['prediction'])
    predictions.index.name = 'Ticker'

    if df_curr is not None and df_vol is not None:
        assert df_curr.shape[1] == df_vol.shape[1], 'The number of columns in df_curr and df_vol are not the same'
        assert df_curr.columns.values.tolist() == df_vol.columns.values.tolist(), 'The columns in df_curr and df_vol are not the same'
        assert df_curr.isna().any().sum() == 0, 'There are NaN values in df_curr'
        assert df_vol.isna().any().sum() == 0, 'There are NaN values in df_vol'
        
        progress_placeholder = st.empty()  # Create an empty placeholder
        progress_bar = progress_placeholder.progress(0)  # Create a progress bar in the placeholder

        for cnt, column_name in enumerate(df_curr.columns):
            df = pp.preprocess(df_curr[[column_name]], df_vol[[column_name]], df_gold, df_infl)
            # set fist column-name to 'Close'
            df.columns = ['Close'] + df.columns.tolist()[1:]
            df = df[predictors]
            assert len(df.columns) == len(
                predictors), f'Number of columns in df ({len(df.columns)}) is not the same as in predictors ({len(predictors)})'
            pred_value = my_model.predict_proba(df[predictors].iloc[-1:])[:,1]

            predictions.loc[column_name] = np.round(pred_value,5)
            progress_bar.progress((cnt + 1) / len(df_curr.columns), 'Please wait...') 
                 
        progress_placeholder.empty()
        
    return predictions  # df with one row per kryptovaluta 

#%%
months = int(st.sidebar.number_input('Return period in months',
             min_value=1, max_value=MAX_MONTHS-1, value=12))

@st.cache_data
def get_returns(df, months):
    target_date = df.index[-1] - DateOffset(months=months)
  
    closest_index = df.index[df.index.get_loc(target_date, method='pad')]
    # st.write('Closest date to target date is: ', closest_index)
    old_prices = df.loc[closest_index].squeeze()

    recent_prices = df.loc[df.index[-1]]
    returns_df = (recent_prices - old_prices) / old_prices

    return closest_index, returns_df


# get data
filnamn = 'yf_tickers.txt'
yf_ticker_names = read_ticker_names(filnamn)

df_curr, df_vol = get_yf_data(yf_ticker_names)
date, df_returns = get_returns(df_curr, months)

n_examine = int(st.sidebar.number_input(
    'Number of currencies to show', min_value=1, max_value=100, value=10))
winners, losers = pd.DataFrame(df_returns.nlargest(n_examine)), pd.DataFrame( df_returns.nsmallest(n_examine))
winners.columns,losers.columns = [f'Return {months}mo'],[f'Return {months}mo']

predictions = get_predictions(df_curr, df_vol, df_gold, inflation[['US_inflation']])

# Ersätt 'up Tomorrow' kolumnen i winners och losers med förutsägelserna
winners['up Tomorrow'] = predictions.loc[winners.index]
losers['up Tomorrow'] = predictions.loc[losers.index]
st.title(f'Returns since {date.date()}')

# make two columns
col1, col2 = st.columns(2)
col1.title('Best')
winners.index.name = 'Ticker'     
losers.index.name = 'Ticker'

col1.dataframe(winners)
bestPick = col1.selectbox('Select one for the graph', winners.index, index=0)
col2.title('Worst')
col2.dataframe(losers)
worstPick = col2.selectbox('Select one for the graph', losers.index, index=0)

st.info(f'Graph: {bestPick}')
st.line_chart(df_curr[bestPick]) # type: ignore

st.info(f'Graph: {worstPick}')
st.line_chart(df_curr[worstPick])  # type: ignore

if st.sidebar.checkbox('Show inflation data', True):
    # make a line graph of the inflations
    inflations = get_inflation_data()
    st.title("Inflation in US and Sweden")
    st.line_chart(inflations[['US_inflation', 'SE_inflation']])
    
if st.sidebar.checkbox('Show Gold data', True):
    # make a line graph of the inflations
    df_gold = get_gold_data()
    st.title("Gold price")

    st.line_chart(df_gold, width=800, height=500) 
    # st.line_chart(df_gold, use_container_width=True)


if st.sidebar.checkbox('Show my own cryptocurrencies', False):
    # my_model = pickle.load(open("my_model.pkl", "rb"))
    # print(my_model.classes_)

    st.title('My own cryptocurrencies')
    my_own_list = ['ETH-USD', 'BTC-USD', 'BCH-USD', 'XRP-USD', 'ZRX-USD', ]
    my_own = pd.DataFrame(df_returns[my_own_list].nlargest(len(my_own_list)))
    my_own.columns = ['Return']
    
    my_own['Prob up tomorrow'] = predictions.loc[my_own_list]
    my_own['Prob up tomorrow'] = my_own['Prob up tomorrow'].round(7)

    st.dataframe(my_own)
    myPick = st.selectbox(
        'Select one of my own cryptocurrencies for graph', my_own_list, index=0)
    st.info(f'Graph: {myPick}')
    st.line_chart(df_curr[myPick])  # type: ignore



