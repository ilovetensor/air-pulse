import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from constants import state_dict, state_dict_reverse, month_map

# conn = st.connection("snowflake")
@st.cache_data()
def load_data():
    # session = conn.session()
    # respiratory = session.table('AIRPULSE.HISTORICAL_DATA.RESPIRATORY_PROBLEMS').to_pandas()
    # respiratory.columns = ['state', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
    #    'Oct', 'Nov', 'Dec', 'Total', 'year']
    # df = session.table('AIRPULSE.HISTORICAL_DATA.AQI_PAST').to_pandas()
    respiratory = pd.read_csv('nb-playground/dataset/respiratory_problems_08_to_15.csv')
    df = pd.read_csv("nb-playground/historical_data_cleaned.csv")

    df.columns = ['state', 'year', 'month', 'aqi']
    df.state = df.state.map(state_dict_reverse)
    df.month = (df.month - 1).map(month_map)
    merged_df = pd.merge(df, respiratory)

    resp = []
    for i in merged_df.index:
        resp.append(merged_df[merged_df.iloc[i].month].iloc[i])
    st.text(resp)
    merged_df['resp'] = resp  #[float(i)*0.1 for i in resp]
    merged_df = merged_df[['state', 'month', 'year', 'aqi', 'resp']].drop_duplicates()
    return df, merged_df


st.title('Air Quality Impacts and other Demographics')
st.subheader('An analysis of the impact of air quality on respiratory problems')

df, respiratory = load_data()


# Add a section for selecting major cities
st.subheader("Compare Cities")
c1, c2 = st.columns([1,1])
with c1:
    selected_state = st.selectbox('Select the state', state_dict.keys(), )
with c2:
    plot_style = st.selectbox('Select filter', ['By Month', 'By Year'])

if plot_style == 'By Month':
    x = list(month_map.values())
    selected_cities_data = respiratory[respiratory.state == selected_state][['month','aqi','resp']].groupby('month').mean()
else:
    x = list(range(2008, 2016))
    selected_cities_data = respiratory[respiratory.state == selected_state][['year','aqi','resp']].groupby('year').mean()


# Plot AQI trends for selected cities
fig_cities = go.Figure()
#
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=np.log10(selected_cities_data['aqi']), name="AQI Values", mode='lines'))
fig.add_trace(go.Scatter(x=x, y=np.log10(selected_cities_data['resp']*0.2), name="Respiratory Problems", mode='lines'))
#
fig.update_layout(title='AQI Levels and ', xaxis_title='Month',
                  yaxis_title='Air Quality Index Prediction')
st.plotly_chart(fig, use_container_width=True)


print(df.head())
print(respiratory.head())

st.subheader("Compare Cities")
# selected_state = st.selectbox('Select the state', state_dict.keys())
# selected_cities_data = respiratory[respiratory.state == selected_state]

