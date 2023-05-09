'''
Detta är en gemensam preprocessing för att skapa stdScaler, testa olika modeller, samt my_crypto.py
'''
import asyncio
import sys

import aiohttp
import numpy as np
import pandas as pd
from binance import AsyncClient, BinanceSocketManager

# def create_new_columns(df_, ticker, target, trend:bool=True, horizons=[2,5,60,250]):
    
#     '''
#     Creates predictors for a given ticker.
    
#         df_ is the dataframe with tickers.
#         ticker is the ticker to create predictors for.
#         target is the target column to create predictors for. : 'y1' or 'y2'
#         trend is a boolean to create Trend columns for the target or not.
#         horizons is the list of horizons to create predictors for.
#         Returns a new dataframe
#     '''
#     df = df_.copy()
    
#     # Move these lines outside the loop
#     df['Tomorrow'] = df[ticker].shift(-1)
#     df['After_tomorrow'] = df[ticker].shift(-2)
#     df['y1'] = (df['Tomorrow'] > df[ticker]).astype(int)
#     df['y2'] = (df['After_tomorrow'] > df[ticker]).astype(int)
    
#     hpref = 'GLD_' if 'GLD' in ticker else 'infl_' if 'inflation' in ticker else ''
#     for horizon in horizons:
#         rolling_averages = df.rolling(horizon, min_periods=1).mean()

#         ratio_column = f"{hpref}Ratio_{horizon}"
#         df[ratio_column] = df[ticker] / rolling_averages[ticker]
        
#         rolling = df.rolling(horizon,closed='left', min_periods=1).mean()
#         print('innan trend',df.shape)
#         if trend:
#             trend_column = f"{hpref}Trend_{horizon}"
#             target_name = 'Tomorrow' if target=='y1' else 'After_tomorrow'
#             print('target_name =' , target_name)
#             print('trend_column =' , trend_column)
#             print("len df.columns", len(df.columns))
#             print('innan rolling',df.shape)
#             df[trend_column] = rolling[target_name]  
#             print('efter rolling',df.shape)

#     return df


def generate_new_columns(df_, horizons=[2, 5, 60, 250], trend=True):
    df = df_.copy()
    ticker = df.columns[0]

    target = 'y1'
    hpref = 'GLD_' if ticker.startswith('GLD') else 'infl_' if 'inflation' in ticker else ''

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

def general_preprocessing(df_,horizons=[2,5,15,30,60,90,250], trend=True):

    df = generate_new_columns(df_, horizons=horizons, trend=trend)
    # df = df.reset_index()   
    
    return df


def preprocessing_currency(df_, quiet=False):
    # kontrollera att index är dateindex
    if type(df_.index[0]) != pd.Timestamp:
        # break with error
        print('index är inte dateindex'*10)
        return df_

    df = df_.copy()
    df = df.drop_duplicates()
    if not quiet:
        print(len(df.columns), 'kolumner totalt')
    s = int(0.9*len(df))
    # remove all columns where there are more than s nan values
    df = df.dropna(axis=1, thresh=s)
    if not quiet:
        print(len(df.columns),
            f'kolumner med minst {s} rader utan nan efter dropna')
        print(f'{df.isna().sum().sum()} rader med någon nan \n{len(df)-df.isna().any().sum()} rader utan nan')
    
    # interpolate missing values
    df = df.interpolate(method='linear',limit_direction='both', axis=0)
    
    if not quiet:
        print(f'Efter interpollate: {df.isna().any().sum()} rader med någon nan \n{len(df)-df.isna().any().sum()} rader utan nan')
        print(df.shape)

    return df

#%%
def add_diff(df_, new_col, col_list):
    df = df_.copy()
    try:
        df[new_col] = df.apply(lambda row: max(
            [row[col] for col in col_list]) - min([row[col] for col in col_list]), axis=1)
    except:
        print('error', col_list)
        print(df.shape)
        print('NaN i add_diff',df.isna().sum().sum())
        print(df.columns)
        print(df.head())
    
    return df
    

def preprocess(df_, df_gold_, df_infl_):
    df = df_.copy()
    df_gold = df_gold_.copy()
    df_infl = df_infl_.copy()
    
    horizons = [2, 5, 30, 60, 90, 250]
    horizons_infl = [75, 90, 250]
    
    df = preprocessing_currency(df,quiet=True)
    # special preprocessing för crypto currencies
    df = general_preprocessing(df,horizons=horizons)
    
    # df = df.set_index('Date')
    
    if df_infl is not None:
      df_infl = general_preprocessing(df_infl, trend=False, horizons=horizons_infl)
      df = df.merge(df_infl, left_index=True, right_index=True, how='left')
    
    if df_gold is not None:     
        df_gold = general_preprocessing(df_gold, trend=False, horizons=horizons) 
        df = df.merge(df_gold, left_index=True, right_index=True, how='left')
       
    ticker_cols = [col for col in df.columns if 'Trend' in col and 'GLD' not in col]
    
    if df.isna().sum().sum() > 0:
        df.dropna(inplace=True,axis=0)
    df = add_diff(df, 'diff', ticker_cols)
    
    # skit i detta för tillfället:
    # gold_cols = [col for col in df.columns if 'GLD' in col]
    # if len(gold_cols) > 0:
    #     print(gold_cols)
    #     df = add_diff(df, 'gold_diff', gold_cols)
    
    return df

def main():
    # read in df_yf.csv
    df_yf = pd.read_csv('df_yf.csv', index_col=0)
    df_yf.index = pd.to_datetime(df_yf.index)
    
    df_gold = pd.read_csv('gold.csv', index_col=0)
    df_gold.index = pd.to_datetime(df_gold.index)
    
    df_infl = pd.read_csv('inflation.csv', index_col=0)    
    df_infl.index = pd.to_datetime(df_infl.index)
    
    df = preprocessing_currency(df_yf)
    print('Antal NaN',df.isna().sum().sum())
    print(df.columns)
    for col in df.columns:
        print('testa med', col )
        preprocess(df_yf[[col]], df_gold, df_infl)
    
    
if __name__ == '__main__':
    main()      
