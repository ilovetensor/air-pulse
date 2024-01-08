import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from constants import state_dict, state_dict_reverse, month_map
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from streamlit_card import card


# st.set_page_config(layout='wide', initial_sidebar_state='expanded')


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
# def weighted_average_prediction(models, weights, input_data):
#     predictions = [model.predict(input_data) for model in models]
#     weighted_prediction = sum(weight * prediction for weight, prediction in zip(weights, predictions))
#     return weighted_prediction


def weighted_average_prediction(models: list, weights: list, predict_df: pd.DataFrame):
    preds = pd.DataFrame()
    for i, m in enumerate(models):
        # m.fit(x_train, y_train)
        preds[i] = m.predict(predict_df)
    preds['weighted_pred'] = (preds * weights).sum(axis=1) / sum(weights)

    return preds['weighted_pred']


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
print(models, weights)

prediction = weighted_average_prediction(models, weights, input_data)
df = load_data()

if year_selected == 2024:
    pred_df = pd.DataFrame({'AQI Value': prediction})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(month_map.values()), y=prediction, name="AQI Values", mode='lines'))
    fig.update_layout(title='AQI Prediction 2024 for every month', xaxis_title='Month',
                      yaxis_title='Air Quality Index Prediction')
    st.plotly_chart(fig, use_container_width=True)

else:

    year_s = df[df.year == year_selected]
    state_s = year_s[year_s.state == state_selected]['aqi']
    if len(state_s) == len(prediction):
        pred_df = pd.DataFrame(data={'predicted': prediction,
                                     'actual': state_s.values})
        fig = px.line(pred_df, x=month_map.values(), y=['predicted', 'actual'])
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info('Insufficient Past Data Available ')
        pred_df = pd.DataFrame({'predicted': prediction})
        fig = px.line(pred_df, x=month_map.values(), y=['predicted'])
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Some more Insights based on Historical Dataset :")

year_state = st.selectbox('Select the State', state_dict_reverse,
                          format_func=lambda x: state_dict_reverse[x],
                          index=state_dict['Delhi']
                          )
fig = go.Figure()
year_wise = df[df['state'] == year_state].groupby('year')['aqi'].mean().reset_index()
fig.add_trace(go.Bar(x=year_wise['year'], y=year_wise['aqi'], name="AQI Values", ))
fig.update_layout(title=f"{state_dict_reverse[year_state]} - AQI Levels change over 24 years", xaxis_title='Year',
                  yaxis_title='Air Quality Levels')
st.plotly_chart(fig, use_container_width=True)

st.subheader("Some Facts over the Years  : ")

st.markdown("---")
col1, col2, col3, = st.columns(3)

lowest = df[['month', 'aqi']].groupby('month').mean().values
low, low_val = np.argmin(lowest), min(lowest)[0]
col1.metric("CLEANEST MONTH IN THE YEAR ", month_map[low]+" 🍃", int(low_val), )

hi, hi_val = np.argmax(lowest), max(lowest)[0]
col2.metric("WORST MONTH OF THE YEAR ", month_map[hi]+" 😷", int(-hi_val))

hi_year = df[df.year>1990][['year','aqi']].groupby('year').mean().values
hi_y, hi_y_val = np.argmax(hi_year), max(hi_year)[0]
col3.metric("MOST POLLUTED YEAR IN HISTORY ", str(int(hi_y)+1990)+" ⌛", int(-hi_y_val))
st.markdown('---')


col1, col2, _ = st.columns(3)

lowest_c = df[['state', 'aqi']].groupby('state').mean().values
low_c, low_c_val = np.argmin(lowest_c), min(lowest_c)[0]
col1.metric("CLEANEST CITY ", state_dict_reverse[low_c] + " 🪷", int(low_c_val), )

hi_c, hi_c_val = np.argmax(lowest_c), max(lowest_c)[0]
col2.metric("WORST CITY ", state_dict_reverse[hi_c] + " 🚦", int(-hi_c_val))


