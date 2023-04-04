#%% 
# Komplett omtag av new_crypto.py
# TODO: kopiera algovitz YT-kod med grafer
# TODO: Gör om till Binance enligt TDD_krypto
# TODO: Läs in inflationsdata och gör gemensam df med krypto
# TODO: Inkludera inflationsgraf med grafer ovan

# TODO: Lägg till valet av my own crypto och hantera dem precis som ovan
# TODO: Kopier new_skapa_modeller.ipynb till my_testa_modeller.ipynb
# TODO: Skapa 1 (en) modell för prediction av valfri krypto
# TODO: Testa olika modelltyper i my_skapa_modeller.ipynb

# TODO: Flera sidor?
# TODO: Slutligen: Byt namn till new_crypto.py igen innan publicering
#%%
#TODO: importera moduler
#%%
import pandas as pd
import numpy as np
import streamlit as st
from pandas.tseries.offsets import DateOffset
#%%
@st.cache_data
def get_data():
    print("Kolla i Binance")
    df=pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]], columns=['a','b','c'])
    return df

st.title('Performance av Kryptovalutor')

months = int(st.number_input('Please enter the number of previous months', min_value=1, max_value=48, value=12))
n_examine = int(st.number_input('Please enter the number of tickers to examine', min_value=1, max_value=25, value=1))

#%%
def get_returns(df, months):
    # This formula gives the day n months back in time
    #           df.index[-1] - DateOffset(months=months)
    # but we can not be sure it is a trading day so instead:
    old_prices = df[:df.index[-1] - DateOffset(months = months)].tail(1).squeeze()   # n months back in time as a Series
    recent_prices = df.loc[df.index[-1]]
    returns_df = (recent_prices - old_prices) / old_prices
    
    return old_prices.name, returns_df  # date of the old prices and the returns

#%%
df = get_data()
date, returns_df = get_returns(df, months)

winners, losers = returns_df.nlargest(n_examine), returns_df.nsmallest(n_examine)
winners.name,losers.name = 'Best','Worst'

st.table(winners)
st.table(losers)

bestPick = st.selectbox('Pick one for graph', winners.index, index=0)
st.line_chart(df[bestPick])

worstPick = st.selectbox('Pick one for graph', losers.index, index=0)
st.line_chart(df[worstPick])

