'''
Detta är en gemensam preprocessing för att skapa stdScaler, testa olika modeller, samt my_crypto.py
'''
from binance import AsyncClient, BinanceSocketManager
import asyncio
import sys

import aiohttp
import numpy as np
import pandas as pd


def create_new_columns(df_, ticker, target, trend:bool=True, horizons=[2,5,60,250]):
    
    '''
    Creates predictors for a given ticker.
    
        df_ is the dataframe with tickers.
        ticker is the ticker to create predictors for.
        target is the target column to create predictors for. : 'y1' or 'y2'
        trend is a boolean to create Trend columns for the target or not.
        horizons is the list of horizons to create predictors for.
        Returns a new dataframe
    '''
    df = df_.copy()
    
    # Move these lines outside the loop
    df['Tomorrow'] = df[ticker].shift(-1)
    df['After_tomorrow'] = df[ticker].shift(-2)
    df['y1'] = (df['Tomorrow'] > df[ticker]).astype(int)
    df['y2'] = (df['After_tomorrow'] > df[ticker]).astype(int)
    
    hpref = 'GLD_' if 'GLD' in ticker else 'infl_' if 'inflation' in ticker else ''
    for horizon in horizons:
        rolling_averages = df.rolling(horizon, min_periods=1).mean()

        ratio_column = f"{hpref}Ratio_{horizon}"
        df[ratio_column] = df[ticker] / rolling_averages[ticker]
        
        rolling = df.rolling(horizon,closed='left', min_periods=1).mean()
        print('innan trend',df.shape)
        if trend:
            trend_column = f"{hpref}Trend_{horizon}"
            target_name = 'Tomorrow' if target=='y1' else 'After_tomorrow'
            print('target_name =' , target_name)
            print('trend_column =' , trend_column)
            print("len df.columns", len(df.columns))
            print('innan rolling',df.shape)
            df[trend_column] = rolling[target_name]  
            print('efter rolling',df.shape)

    return df


def generate_new_columns(df_, horizons=[2, 5, 60, 250], trend=True):
    df = df_.copy()
    ticker = df.columns[0]

    target = 'y1'
    hpref = 'GLD_' if 'GLD' in ticker else 'infl_' if 'inflation' in ticker else ''
    for horizon in horizons:
        rolling_averages = df.rolling(horizon, min_periods=1).mean()

        ratio_column = f"{hpref}Ratio_{horizon}"
        df[ratio_column] = df[ticker] / rolling_averages[ticker]
        df['Tomorrow'] = df[ticker].shift(-1)
        rolling = df.rolling(horizon, closed='left', min_periods=1).mean()
        if trend:
            # tomorrow's close price - alltså nästa dag
            df['Tomorrow'] = df[ticker].shift(-1)
            trend_column = f"{hpref}Trend_{horizon}"
            target_name = 'Tomorrow' if target == 'y1' else 'After_tomorrow'
            df[trend_column] = rolling[target_name]

    df = df.drop(['Tomorrow'], axis=1)
    return df

def general_preprocessing(df_,horizons=[2,5,60,250], trend=True):
    df = df_.copy()
    # df = df.reset_index()
    tickers = df.columns.tolist()
    target = 'y1'

    for tix, ticker in enumerate(tickers):

        df_temp = create_new_columns(df[[ticker]], ticker, target, trend=trend, horizons=horizons)
        df_temp = df_temp.reset_index()
        df_temp['Ticker'] = ticker
        print('innan concat',df.shape, df_temp.shape)
        # print('df.index', df.index)
        # df_temp.set_index(['Date'], inplace=True)
        # print('df_temp.index', df_temp.index)

        df = pd.concat([df, df_temp], axis=0)
       
    # Special preprocessing for crypto data
    print(f'{df.shape} innan dropping nan columns')
    # drop the columns where all values are nan
    df = df.dropna(axis=1,how='all')
    print(f'{df.shape} efter dropping nan columns')

    s=int(0.9*len(df))
    # remove all columns where there are more than s nan values
    df = df.dropna(axis=1, thresh=s)
    
    print(f'{df.shape} efter dropping rader med för många NaNs')      
    
    return df

def preprocessing_currency(df_):
    df = df_.copy()
    df = df.drop_duplicates()
    s=int(0.9*len(df))
    # remove all columns where there are more than s nan values
    df = df.dropna(axis=1, thresh=s)
    print(f'{df.isna().any().sum()} rader med någon nan \n{len(df)-df.isna().any().sum()} rader utan nan')
    # interpolate missing values
    df.interpolate(method='linear', inplace=True)
    print(f'{df.isna().any().sum()} rader med någon nan \n{len(df)-df.isna().any().sum()} rader utan nan')
    df.shape
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date
    df = df.set_index('Date')
    return df

def preprocess(df_, df_gold_, df_infl_):
    df = df_.copy()
    df_gold = df_gold_.copy()
    df_infl = df_infl_.copy()
    
    horizons = [2, 5, 15, 30, 60, 90, 250]
    horizons_infl = [75, 90, 250]
    
    df = preprocessing_currency(df)
    # spcial preprocessing för crypto currencies
    df = general_preprocessing(df,horizons=horizons)
    
    # TODO:inflation preprocessing
    # df_infl = general_preprocessing(df_infl, trend=False, horizons=horizons_infl)
    
    # TODO: gold preprocessing        
    # df_gold = general_preprocessing(df_gold, trend=False, horizons=horizons) 
           
    # TODO: merge inflation and gold data into df
    # df=df.set_index('Date')
    
    # df = df.merge(df_infl, left_index=True,cright_index=True, how='left')
    # df = df.merge(df_gold, left_index=True,cright_index=True, how='left')
    
    # TODO: add variance/diff column för varje rad  
    def add_diff(df_, new_col, col_list):
        df = df_.copy()
        df[new_col] = df.apply(lambda row: max(
            [row[col] for col in col_list]) - min([row[col] for col in col_list]), axis=1)
        return df

    ticker_cols = [col for col in df.columns if 'Trend' in col and 'GLD' not in col]
    print(ticker_cols)
    df = add_diff(df, 'diff', ticker_cols)
    
    gold_cols = [col for col in df.columns if 'GLD' in col]
    print(gold_cols)
    df = add_diff(df, 'gold_diff', gold_cols)
    return df


#%%
# asyncio lösning

# Ange valutorna du vill ladda ner
cryptos = ['BTCUSDT', 'ETHUSDT', 'BCHUSDT', 'XRPUSDT', 'ZRXUSDT',
           'BNBUSDT', 'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'AVAXUSDT']

# start timer
import time
start_time = time.time()
async def get_historical_klines(client, symbol):
    klines = await client.get_historical_klines(symbol, AsyncClient.KLINE_INTERVAL_1DAY, "24 months ago UTC")
    return klines


async def main():
    client = await AsyncClient.create()

    # Hämta historiska klinedata för varje valuta
    klines = {}
    for symbol in cryptos:
        klines[symbol] = await get_historical_klines(client, symbol)

    close_prices_list = []
    timestamps_list = []
    for crypto_data in klines.values():
        close_prices = [float(kline[4]) for kline in crypto_data]
        timestamps = [int(kline[0]) for kline in crypto_data]
        close_prices_list.append(close_prices)
        timestamps_list.append(timestamps)

    await client.close_connection()

    # Konvertera close_prices_list till en pandas DataFrame
    data = {symbol: prices for symbol,
            prices in zip(cryptos, close_prices_list)}
    df = pd.DataFrame(data)

    # Hämta tidpunkterna för varje datapunkt och konvertera dem till datetime-objekt
    dates = pd.to_datetime(timestamps_list[0], unit='ms')

    # Sätt Date som index för DataFrame
    df.index = dates

    # Visa DataFrame
    print('med Binance async')
    print(df)
    
    print(f'Total time: {round(time.time() - start_time, 1)}s')
    

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())


# %%
