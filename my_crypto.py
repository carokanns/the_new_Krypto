#!/usr/bin/env python
# coding: utf-8

# In[31]:


#%% 
# Komplett omtag av new_crypto.py
# HACK: Påbörjad dummy - fixa resten
# TODO: Inkludera inflationsgraf med grafer ovan

# TODO: Lägg till valet av my own crypto och hantera dem precis som ovan
# TODO: Kopier new_skapa_modeller.ipynb till my_testa_modeller.ipynb
# TODO: Skapa 1 (en) modell för prediction av valfri krypto
# TODO: Testa olika modelltyper i my_skapa_modeller.ipynb

# TODO: Flera sidor?
# TODO: Slutligen: Byt namn till new_crypto.py igen innan publicering
#%%


# In[32]:


import pandas as pd
import numpy as np
import streamlit as st
from pandas.tseries.offsets import DateOffset
from binance.client import Client
import datetime


# In[33]:


@st.cache_data
def get_data():

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
    start_time = "2 month ago UTC"

    # Hämta close-priser för alla kryptovalutor
    dates = None
     #add a counter to print a message every 100 symbols
    counter = 0
    for symbol in symbols:   
        try:
            klines = client.get_historical_klines(symbol['symbol'], interval, start_time)
            if dates is None:
                # Extrahera och konvertera tidsstämplar till datum
                dates = [datetime.datetime.fromtimestamp(int(kline[0]) / 1000).strftime('%Y-%m-%d') for kline in klines]
            close_prices[symbol['symbol']] = [float(kline[4]) for kline in klines]
            # every 100 symbols print a message
            if counter % 100 == 0:
                print(f"Hämtade {counter} symboler")
        except Exception as e:
            print(f"Kunde inte hämta data för {symbol['symbol']}: {e}")
        counter += 1
    # Konvertera close_prices-dikten till en pandas DataFrame
    df = pd.DataFrame.from_dict(close_prices, orient='index').transpose()  
    # Lägg till datum som index för DataFrame
    df.index = pd.to_datetime(dates)                                   # type: ignore
    print(df.head())
    return df
df=get_data()


# ## Inflation

# In[34]:


def add_horizon_columns(inflation, horizons):
    for horizon in horizons:
        inflation['US_inflation_'+str(horizon)] = inflation['US_inflation'].rolling(horizon, 1).mean()
        inflation['SE_inflation_'+str(horizon)] = inflation['SE_inflation'].rolling(horizon, 1).mean()
    return inflation


# In[35]:


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


# In[36]:


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



# In[37]:


st.title('Performance av Kryptovalutor')


months = int(st.number_input(
    'Please enter the number of previous months', min_value=1, max_value=48, value=12))
n_examine = int(st.number_input(
    'Please enter the number of tickers to examine', min_value=1, max_value=25, value=10))

#%%


def get_returns(df, months):
    target_date = df.index[-1] - DateOffset(months=months)

    # Find the index that is closest to the target_date and does not exceed it
    closest_index = df.index[df.index.get_loc(target_date, method='pad')]
    st.write('Closest date to target date is: ', closest_index)
    old_prices = df.loc[closest_index].squeeze()

    recent_prices = df.loc[df.index[-1]]
    returns_df = (recent_prices - old_prices) / old_prices

    return closest_index, returns_df


date, returns_df = get_returns(df, months)


winners, losers = returns_df.nlargest(n_examine), returns_df.nsmallest(n_examine)
winners.name,losers.name = 'Best','Worst'

st.table(winners)
st.table(losers)


# In[40]:


bestPick = st.selectbox('Pick one for graph', winners.index, index=0)
st.line_chart(df[bestPick]) # type: ignore

worstPick = st.selectbox('Pick one for graph', losers.index, index=0)
st.line_chart(df[worstPick]) # type: ignore

# make a line graph of the inflations
inflations = get_inflation_data()
st.line_chart(inflations[['US_inflation','SE_inflation']]) # type: ignore
    

