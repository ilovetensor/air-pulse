import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from constants import state_dict, state_dict_reverse, month_map
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
@st.cache_data()
def load_model():
    models = joblib.load("nb-playground/weighted_models.pkl")
    weights = joblib.load("nb-playground/model_weights.pkl")
    return models, weights


@st.cache_data()
def load_data():
    df = pd.read_csv("nb-playground/historical_data_cleaned.csv")
    return df


# Define a function for weighted averaging
def weighted_average_prediction(models, weights, input_data):
    predictions = [model.predict(input_data) for model in models]
    weighted_prediction = sum(weight * prediction for weight, prediction in zip(weights, predictions))
    return weighted_prediction


st.title("AQI Levels Forecasting ")
st.subheader("Take actions based of upcoming trends")
st.markdown("""The `weighted average` model is trained over the historical dataset from 1990 to 2015 for all states,
capturing the monthly and yearly trends to make better predictions. 
The data has training accuracy of `98%` and test accuracy for 1 year of data as `80.1%`. 
**Input Parameters** : state name, year, month
**Output** : AQI value
""")
st.subheader("AQI Predictions for 2024")
st.text("Can also compare predictions of previous years with actuals.")

c1, c2 = st.columns(2)
with c1:
    state_selected = st.selectbox('Select State', state_dict_reverse,
                                  format_func=lambda x: state_dict_reverse[x],
                                  index=state_dict['Delhi']
                                  )

with c2:
    year_selected = st.selectbox('Select The Year',
                                 [2024, *range(2015, 1990, -1)],
                                 index=0)


cols = ['state', 'year', 'month']
input_data = pd.DataFrame([[state_selected, year_selected, i] for i in range(1, 13)], columns=cols)

models, weights = load_model()
weights = [0.3,0.6,0.1]
print(models, weights)

prediction = weighted_average_prediction(models, weights, input_data)
df = load_data()

if year_selected == 2024:
    pred_df = pd.DataFrame({'AQI Value': prediction})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(month_map.values()), y=prediction, name="AQI Values", mode='lines'))
    # fig.add_trace(go.Bar(x=list(month_map.values()), y=prediction, name='AQI Values'))
    fig.update_layout(title='AQI Prediction 2024 for every month', xaxis_title='Month',
                     yaxis_title='Air Quality Index Prediction')
    st.plotly_chart(fig, use_container_width=True)

else:
    try:
        year_s = df[df.year == year_selected]
        state_s = year_s[year_s.state == state_selected]['aqi']
        pred_df = pd.DataFrame(data={'predicted':prediction,
                                   'actual': state_s.values})
        fig = px.line(pred_df, x=month_map.values(), y=['predicted','actual'])
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning('Insufficient Past Data Available ')
        pred_df = pd.DataFrame({'predicted': prediction})
        fig = px.line(pred_df, x=month_map.values(), y=['predicted'])
        st.plotly_chart(fig, use_container_width=True)


