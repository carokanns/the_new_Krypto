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
    
    hpref = 'GLD_' if 'GLD' in ticker else 'infl_' if 'inflation' in ticker else ''
    for horizon in horizons:
        rolling_averages = df.rolling(horizon, min_periods=1).mean()

        ratio_column = f"{hpref}Ratio_{horizon}"
        df[ratio_column] = df[ticker] / rolling_averages[ticker]
        
        rolling = df.rolling(horizon,closed='left', min_periods=1).mean()
        
        if trend:
            trend_column = f"{hpref}Trend_{horizon}"
            target_name = 'Tomorrow' if target=='y1' else 'After_tomorrow'
            df[trend_column] = rolling[target_name]  

    return df

def general_preprocessing(df_,horizons=[2,5,60,250], trend=True):
    df = df_.copy()
    tickers = df.columns.tolist()
    target = 'y1'

    for ticker in tickers:
        print(ticker)
        df_temp = create_new_columns(df[[ticker]], ticker, target, trend=trend, horizons=horizons)
        df_temp = df_temp.reset_index()
        df_temp['Ticker'] = ticker
        df = pd.concat([df, df_temp], axis=0)

    # Special preprocessing for crypto data
    # df = pd.read_csv('all_tickers.csv', index_col=0)
    print(f'{df.shape} innan dropping nan columns')
    # drop the columns where all values are nan
    df = df.dropna(axis=1,how='all')
    print(f'{df.shape} efter dropping nan columns')

    s=int(0.9*len(df))
    # remove all columns where there are more than s nan values
    df = df.dropna(axis=1, thresh=s)
    
    print(f'{df.shape} efter dropping rader med för många NaNs')      
    
    return df


def preprocess(df_, df_gold_, df_infl_):
    horizons = [2, 5, 15, 30, 60, 90, 250]
    horizons_infl = [75, 90, 250]
    df = df_.copy()
    df_gold = df_gold_.copy()
    df_infl = df_infl_.copy()
    
    df = df.drop_duplicates()
    # spcial preprocessing för crypto currencies
    df = general_preprocessing(df,horizons=horizons)

    # TODO:inflation preprocessing
    # df_infl = general_preprocessing(df_infl, trend=False, horizons=horizons_infl)
    
    # TODO: gold preprocessing        
    # df_gold = general_preprocessing(df_gold, trend=False, horizons=horizons) 
           
    # TODO: merge inflation and gold data into df
    df=df.set_index('Date')
    
    # df = df.merge(df_infl, left_index=True,cright_index=True, how='left')
    # df = df.merge(df_gold, left_index=True,cright_index=True, how='left')
    
    # add variance/diff column   
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