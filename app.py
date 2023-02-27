# Kenny Style
import time
start = time.time()

import streamlit as st

# Initializing all libraries
import pandas as pd
import numpy as np

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px               
from plotly.subplots import make_subplots 
import plotly.graph_objects as go
st.set_option('deprecation.showPyplotGlobalUse', False)

page_bg_img = '''
<style>
body {
background-image: url("https://github.com/gitsim02/FoundationProject-1/blob/75a5a682399b79813a60141c05562544914e0abf/StreamlitBackground.png");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

logo = st.container()
with logo:
   col1, col2 = st.columns([4, 1])
   col1.title("Oil Production forecast using Time Series Analysis")
   col2.image("https://github.com/gitsim02/FoundationProject-1/blob/377e2fb72eb68f2d7559fb8c0c617ca89d8db845/transparent_Black.jpg?raw=true",  width=150)
   st.markdown("""------""")
  
    
dataset = st.container()
with dataset:
    st.markdown('''### Introduction''')
    
    st.markdown(" ###### Oil Production data for 20 years")
    #col2.write(df.head(3))
    
    st.markdown("""Oil Production has great impact on the economy of each country.
    Oil is considered liquid gold, as it is that valuable a commodity.
    We extract data from econdb.  
    Click on the link to view the Oil Production data:   
    https://github.com/Kenrich005/ISB_FP2_Oil_Production_Forecast/blob/main/oil_prod_data.csv    
    """)
  

plots = st.container()
with plots:
    st.markdown(""" ##### Oil Production Trend Chart over 20 Years""")

    
    df = pd.read_csv('https://raw.githubusercontent.com/Kenrich005/ISB_FP2_Oil_Production_Forecast/main/oil_prod_data.csv')
    df.plot(kind = "line",x="Date",y='Oilproduction',figsize=(20,5) )
    plt.ylabel('Oil Production', fontsize=10)
    plt.xlabel('Time Period', fontsize=10)
    plt.title('Oil Production in US over 5 Years', fontsize = 12)
    plt.show()
    st.pyplot()
    st.write("\n\n\n")
    st.markdown("""        """)


st.markdown(""" ### Oil Price Production using PROPHET  """)

#from joblib import load
import joblib

#prophet_model = load("prophet.pkl")
#sarimax_model = joblib.load("sarimax.pkl")



from prophet import Prophet

prediction = st.container()
with prediction:
    n_months = st.text_input('Enter the number of months for forecast',12)
    n_months = int(n_months)
    
    st.write("Warning: As is with any forecast, the farther the duration, the lesser is the confidence of the estimate")
    
    
    df_pr = df[['Date','Oilproduction']].copy()
    df_pr.columns = ['ds','y']
    
    m = Prophet()
    m.fit(df_pr)
    
    future = m.make_future_dataframe(periods=n_months+9,freq='MS')    
    prophet_pred = m.predict(future)
    st.write("Oil Production forecast - we can see numerous features that can benefit in decision making with respect to the Oil Purchasing")
    st.write(prophet_pred.tail())
    st.write("")
    
    prophet_pred_df = pd.DataFrame(prophet_pred)[['ds','yhat','yhat_lower','yhat_upper']]
    prophet_pred_df.columns = ['Date','oil_prod_forecast','forecast_lower','forecast_upper']
    prophet_pred_df = prophet_pred_df.set_index("Date")
    prophet_pred.index.freq = "MS"
    
    st.write("Here we see the Oil Production forecast for the next ",n_months," months. Along with the conservative(lower) and high risk(upper) estimates.")
    st.write(prophet_pred_df.tail(n_months))
    
    prophet_df_temp = prophet_pred_df.tail(n_months)
    
    plt.figure(figsize=(12,7))
    sns.lineplot(x= prophet_df_temp.index, y=prophet_df_temp["oil_prod_forecast"],label='oil_prod_forecast',linestyle="-")
    sns.lineplot(x= prophet_df_temp.index, y=prophet_df_temp["forecast_lower"],label='forecast_lower',linestyle="-.")
    sns.lineplot(x= prophet_df_temp.index, y=prophet_df_temp["forecast_upper"],label='forecast_upper',linestyle="--")
    plt.legend()
    plt.ylabel('Oil Production', fontsize=10)
    plt.xlabel('Time Period', fontsize=10)
    plt.title('Oil Production in US over time period', fontsize = 12)
    plt.show()
    st.pyplot()
    st.write("\n\n\n")
    st.markdown("""        """)
    
    
    
end = time.time()

st.write("Performance - Time taken to execute the entire program:",round(end-start,4),"seconds")
