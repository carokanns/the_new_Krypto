#!/usr/bin/env python
# coding: utf-8


#%% 
# Komplett omtag av new_crypto.py

# TODO: Skapa 1 (en) modell för prediction av valfri krypto i my_testa_modeller.ipynb.
# TODO: Testa olika modelltyper och bestäm vilken som skall användas i my_skapa_modeller.ipynb
# TODO: Skapa en StandardScaler för alla kryptovalutor i Binance
# TODO: Varians (alt max - min) inom varje rad mellan olika horisontvärden
# TODO: Ta med guld i modellen
# TODO: Ta bort svensk inflation

# TODO: Flera sidor?
# TODO: Slutligen: Byt namn till new_crypto.py igen innan publicering
# TODO: merge into master
#%%

import pandas as pd
import numpy as np
import streamlit as st
from pandas.tseries.offsets import DateOffset
from binance.client import Client
import datetime


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
    inflations = add_horizon_columns(inflations, [75, 90, 250])
    return inflations


st.title('Performance of Crypto currencies')
MAX_MONTHS = 24

@st.cache_data
def get_data():

    api_key = '2jxiCQ8OIWmU4PZH4xfwKEY9KYerDkSWzwNCqoaMzj41eJgWBsSqA3VYqkt2wmdX'
    api_secret = 'YY1Qj1t0JZrE4tQdaBBxT8iwl2tbFalWp1FHjyZ9selBb6OnQ0Oj8aVdiXO7YLMz'

    client = Client(api_key, api_secret)

    # Hämta handelspar
    symbols = client.get_all_tickers()
    symbols = [
        symbol for symbol in symbols if symbol['symbol'].endswith('USDT')]

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
                dates = [datetime.datetime.fromtimestamp(
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

df = get_data()

months = int(st.number_input('Months back to compute returns from', min_value=1, max_value=MAX_MONTHS-1, value=12))

#%%

@st.cache_data
def get_returns(df, months):
    target_date = df.index[-1] - DateOffset(months=months)
    # st.write('Target date is: ', target_date.date())
    # st.write('Most recent date is: ', df.index[-1])
    # st.write('oldest date is: ', df.index[0])
    # st.write('Number of rows in dataframe: ', len(df))
    # Find the index that is closest to the target_date and does not exceed it
    closest_index = df.index[df.index.get_loc(target_date, method='pad')]
    # st.write('Closest date to target date is: ', closest_index)
    old_prices = df.loc[closest_index].squeeze()

    recent_prices = df.loc[df.index[-1]]
    returns_df = (recent_prices - old_prices) / old_prices

    return closest_index, returns_df


date, returns_df = get_returns(df, months)

n_examine = int(st.sidebar.number_input(
    'Number of currencies to examine', min_value=1, max_value=25, value=10))
winners, losers = returns_df.nlargest(n_examine), returns_df.nsmallest(n_examine)
winners.name,losers.name = 'Best','Worst'

st.title(f'Returns since {date.date()}')
# make two columns
col1, col2 = st.columns(2)
col1.title('Best')
col1.table(winners)
bestPick = col1.selectbox('Select one for the graph', winners.index, index=0)
col2.title('Worst')
col2.table(losers)
worstPick = col2.selectbox('Select one for the graph', losers.index, index=0)

st.info(f'Graph: {bestPick}')
st.line_chart(df[bestPick]) # type: ignore
# st.dataframe(returns_df)

st.info(f'Graph: {worstPick}')
st.line_chart(df[worstPick])  # type: ignore

if st.sidebar.checkbox('Show my own cryptocurrencies', True):
    st.title('My own cryptocurrencies')
    my_own_list = [ 'ETHUSDT', 'BTCUSDT', 'BCHUSDT', 'XRPUSDT','ZRXUSDT',]
    my_own = returns_df[my_own_list].nlargest(len(my_own_list))
    my_own.name='My own'
    st.table(my_own)
    myPick = st.selectbox('Select one of my own cryptocurrencies for graph', my_own_list, index=0)
    st.info(f'Graph: {myPick}')
    st.line_chart(df[myPick])  # type: ignore

if st.sidebar.checkbox('Show inflation data', True):
    # make a line graph of the inflations
    inflations = get_inflation_data()
    st.title("Inflation in US and Sweden")
    st.line_chart(inflations[['US_inflation', 'SE_inflation']])


