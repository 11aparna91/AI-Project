import statsmodels.api as sm
import streamlit as st
import pandas as pd
import matplotlib
from matplotlib import pyplot

#Loading up the Regression model we created
model = sm.tsa.arima.ARIMA()
model.load('model_arima.json')
food_series_df = pd.read_cav("food_series_df.csv")
movingAverage = food_series_df.rolling(window=30).mean()
movingAverage.fillna(0)

shfited = pd.DataFrame({'predicShfited2':pd.Series(model.fittedvalues,copy=True),'day':food_series_df.index[:1969]})
shfited = shfited.set_index('day')

predictVsActual = pd.DataFrame({'actual':food_series_df,
                                'predictDiff':shfited['predicShfited2'],
                                'base':movingAverage})

predictVsActual['predict'] = predictVsActual.loc[:,['predictDiff','base']].sum(axis=1)

#Caching the model for faster loading
@st.cache



def display_graph(s,e):
    pyplot.plot(predictVsActual['predict'].iloc[s:e],label='Predicted')
    pyplot.legend()
    pyplot.title("Prediction using ARIMA")
    st.pyplot()



st.title('Food Delivery Forecasting')
st.image("""https://builtin.com/sites/www.builtin.com/files/styles/og/public/food-delivery-companies.jpg""")
st.header('Enter forecasting days:')


s = st.number_input('Enter start day for prediction', min_value=0, max_value=1960, value=1)
e = st.number_input('Enter end day for prediction', min_value=s, max_value=1969, value=1)

if st.button('Predict'):
    display_graph(s,e)
