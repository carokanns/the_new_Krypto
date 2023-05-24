'''
Detta är en gemensam preprocessing för att 
  - skapa stdScaler 
  - testa olika modeller och träna en av dem 
  - köra my_crypto.py som visar/selekter krypto för valfri tidshorisont samt prognos
'''
import numpy as np
import pandas as pd


def generate_new_columns(df_, horizons=[2, 5, 30, 60, 250], trend=True):
    df = df_.copy()
    ticker = df.columns[0]

    target = 'y1'
    
    if ticker == 'Volume':
        hpref = 'vol_'
    elif ticker == 'US_inflation':
        hpref = 'infl_'
    elif ticker.startswith('GLD'):
        hpref = 'GLD_'
    else:
        hpref = ''
        
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


def preprocessing_currency(df_, quiet=False):
    # kontrollera att index är dateindex

    assert type(df_.index[0]) == pd.Timestamp, f'index {type(df_.index[0])} är inte dateindex'

    df = df_.copy()
    df = df.drop_duplicates()
    
    s = int(0.9*len(df))
    # remove all columns where there are more than s nan values
    df = df.dropna(axis=1, thresh=s)
    
    # interpolate missing values
    df = df.interpolate(method='linear',limit_direction='both', axis=0)

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
    
def add_last_day(df_):
    df = df_.copy()
    df['before_kvot'] = np.where(df.iloc[:,0].shift(1) == 0, 0, df.iloc[:,0]/df.iloc[:,0].shift(1))
    
    df['before_up'] = df['before_kvot'] > 1
    return df
        

def preprocess(df_curr_, df_vol_,df_gold_, df_infl_):
    df_curr = df_curr_.copy()
    df_vol = df_vol_.copy()
    df_gold = df_gold_.copy()
    df_infl = df_infl_.copy()
    
    horizons = [2, 5, 30, 60, 90, 250]
    horizons_infl = [75, 90, 250]
    # rename column 1 to Volume
    df_vol.columns = ['Volume']
    
    # print('FÖRST')
    # print(df_curr.head(3))
    # print(df_vol.head(3))
    
    # special preprocessing för valutor
    df_curr = preprocessing_currency(df_curr,quiet=True)
    # special preprocessing för crypto currencies
    df_vol = preprocessing_currency(df_vol,quiet=True)
    
    # print('EFTER PREPROCESSING - df_curr och df_vol')
    # print(df_curr.head(3))
    # print(df_vol.head(3))
   
    df_curr = generate_new_columns(df_curr,horizons=horizons)
    # print('EFTER generate_new_columns - df_curr')
    # print(df_curr.head(3))
    df_vol = generate_new_columns(df_vol, trend=False, horizons=horizons)
    # print('EFTER generate_new_columns - df_vol')
    # print(df_vol.head(3))
    
    df = df_curr.merge(df_vol, left_index=True, right_index=True, how='left')
    # print('EFTER MERGE -> df')
    # print(df.head(3))
    # df = df.set_index('Date')
    
    if df_infl is not None:
      df_infl = generate_new_columns(df_infl, trend=False, horizons=horizons_infl)
      df = df.merge(df_infl, left_index=True, right_index=True, how='left')
    
    if df_gold is not None:     
        df_gold = generate_new_columns(df_gold, trend=False, horizons=horizons) 
        df = df.merge(df_gold, left_index=True, right_index=True, how='left')
       
    ticker_cols = [col for col in df.columns if 'Trend' in col and 'GLD' not in col]
    
    if df.isna().sum().sum() > 0:
        df.dropna(inplace=True,axis=0)
    df = add_diff(df, 'diff', ticker_cols)
    df = add_last_day(df)
    
    
    # skit i detta för tillfället:
    # gold_cols = [col for col in df.columns if 'GLD' in col]
    # if len(gold_cols) > 0:
    #     print(gold_cols)
    #     df = add_diff(df, 'gold_diff', gold_cols)
    
    return df

def main():
    # read in df_yf.csv
    df_curr = pd.read_csv('df_curr.csv', index_col=0)
    df_curr.index = pd.to_datetime(df_curr.index)
    df_vol = pd.read_csv('df_vol.csv', index_col=0)
    df_vol.index = pd.to_datetime(df_vol.index)
    
    df_gold = pd.read_csv('gold.csv', index_col=0)
    df_gold.index = pd.to_datetime(df_gold.index)
    
    df_infl = pd.read_csv('inflation.csv', index_col=0)    
    df_infl.index = pd.to_datetime(df_infl.index)
    
    df_curr = preprocessing_currency(df_curr)
    df_vol = preprocessing_currency(df_vol)
    # my_cols = None
    assert df_curr is not None, 'df_curr is None'
    assert df_vol is not None, 'df_vol is None'

    # print('Antal NaN',df_curr.isna().sum().sum())
    # print('curr',df_curr.columns)
    # print('vol',df_vol.columns)
    for idx,col in enumerate(df_curr.columns):
        print('preprocess', col )
        df = preprocess(df_curr[[col]], df_vol[[col]],df_gold, df_infl)

        assert not np.isinf(df).any().any(), f'hittade inf i {col}'
        
        
    df.to_csv(f'preprocessed.csv')
if __name__ == '__main__':
    main()      
