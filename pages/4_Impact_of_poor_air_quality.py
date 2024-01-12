import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.constants import state_dict, state_dict_reverse, month_map
from scipy.stats import pearsonr

np.seterr(divide='ignore', invalid='ignore',all='ignore')

# conn = st.connection("snowflake")
st.set_page_config(layout="wide")


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
    merged_df['resp'] = resp  # [float(i)*0.1 for i in resp]
    merged_df = merged_df[['state', 'month', 'year', 'aqi', 'resp']].drop_duplicates()
    return df, merged_df


def load_vehicles():
    df = pd.read_csv("nb-playground/historical_data_cleaned.csv")

    df.columns = ['state', 'year', 'month', 'aqi']
    df.state = df.state.map(state_dict_reverse)
    df = df.groupby(['year', 'state', 'month'])['aqi'].mean().reset_index()
    df = df.pivot(index=['year', 'month'], columns='state', values='aqi')
    df.drop(df.columns[df.count() == 1], axis=1, inplace=True)
    df.fillna(df.mean(), inplace=True)
    df.columns = ["".join(i.split()).upper() for i in df.columns]

    vehicles = pd.read_csv('nb-playground/dataset/vehicles.csv', parse_dates=['DATE'])
    vehicles.drop(['CATEGORY', 'SUBCATEGORY', 'SOURCE', 'UNIT', 'FREQUENCY', 'CURRENCY',
                   'IDENTIFIER', 'TITLE', ], inplace=True, axis=1)

    vehicles['year'] = vehicles.DATE.dt.year
    vehicles['month'] = vehicles.DATE.dt.month
    vehicles.drop(vehicles.columns[vehicles.count() == 1], axis=1, inplace=True)
    vehicles.fillna(vehicles.mean(), inplace=True)
    vehicles = pd.merge(df, vehicles, on=['year', 'month'], suffixes=('_veh', '_aqi'))
    vehicles.drop(['DATE', 'INDIA'], inplace=True, axis=1)
    return vehicles, df



def load_electricity():
    df = pd.read_csv("nb-playground/historical_data_cleaned.csv")

    df.columns = ['state', 'year', 'month', 'aqi']
    df.state = df.state.map(state_dict_reverse)
    df = df.groupby(['year', 'state', 'month'])['aqi'].mean().reset_index()
    df = df.pivot(index=['year', 'month'], columns='state', values='aqi')
    df.drop(df.columns[df.count() == 1], axis=1, inplace=True)
    df.fillna(df.mean(), inplace=True)
    df.columns = ["".join(i.split()).upper() for i in df.columns]

    electricity = pd.read_csv('nb-playground/dataset/electricity_demand.csv', parse_dates=['DATE'])
    electricity.drop(['CATEGORY', 'SUBCATEGORY', 'SOURCE', 'UNIT', 'FREQUENCY', 'CURRENCY',
                      'IDENTIFIER', 'TITLE', ], inplace=True, axis=1)
    electricity['year'] = electricity.DATE.dt.year
    electricity['month'] = electricity.DATE.dt.month
    electricity.fillna(electricity.mean(), inplace=True)
    electricity = pd.merge(df, electricity, on=['year', 'month'], suffixes=('_elec', '_aqi'))
    electricity.drop(['DATE', 'INDIA'], inplace=True, axis=1)

    return electricity, df


st.title('Air Quality Impacts and External Factors')
st.markdown('---')
st.subheader("1. Respiratory Health and AQI Correlation")
st.markdown(
    "Analyzing the correlation between Respiratory Problems and Air Quality Index (AQI) provides valuable insights "
    "into how"
    "air pollution affects health across different states. The correlation scores highlight which regions are more "
    "significantly"
    "impacted by AQI in terms of respiratory issues.")

df, respiratory = load_data()

resp_cols = st.columns([1, 3])
with resp_cols[0]:
    # Add a section for selecting major cities
    st.subheader("Compare Cities")
    selected_state = st.selectbox('Select the state', sorted(respiratory.state.unique()))
    plot_style = st.selectbox('Select filter', ['By Month', 'By Year'])
    scaler = st.selectbox('Select value Scaler', ['Standard Scaler', 'MinMax Scaler'])

    if plot_style == 'By Month':
        x = list(month_map.values())
        respiratory = respiratory[['state', 'month', 'aqi', 'resp']].groupby(['state', 'month']).mean().reset_index()
        selected_cities_data = respiratory[respiratory.state == selected_state][['month', 'aqi', 'resp']].groupby(
            'month').mean()
    else:
        x = list(range(2008, 2016))
        respiratory = respiratory[['state', 'year', 'aqi', 'resp']].groupby(['state', 'year']).mean().reset_index()
        selected_cities_data = respiratory[respiratory.state == selected_state][['year', 'aqi', 'resp']].groupby(
            'year').mean()

    corr_resp = respiratory[['state', 'aqi', 'resp']].groupby('state').corr().xs('aqi', level=1).sort_values(by='resp',
                                                                                                             ascending=False)
    st.metric("CORRELATION SCORE ", corr_resp.loc[selected_state]['resp'].round(1), selected_state)

    scaler_obj = MinMaxScaler() if scaler == 'MinMax Scaler' else StandardScaler()
    selected_cities_data[['aqi', 'resp']] = scaler_obj.fit_transform(selected_cities_data[['aqi', 'resp']])

# Plot AQI trends for selected cities
with resp_cols[1]:
    fig_cities = go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=selected_cities_data['aqi'],
                             name="AQI Values", mode='lines', line_shape='spline'))
    fig.add_trace(go.Scatter(x=x, y=selected_cities_data['resp'],
                             name="Respiratory Problems", mode='lines', line_shape='spline'))
    #
    fig.update_layout(title='AQI Levels and ', xaxis_title='Month',
                      yaxis_title='Air Quality Index Prediction')

    st.plotly_chart(fig, use_container_width=True)

fig_resp = go.Figure()
fig_resp.add_trace(go.Bar(x=corr_resp.index, y=corr_resp['resp'].values))
st.plotly_chart(fig_resp, use_container_width=True)

st.markdown('---')
st.subheader("2. Discovering the Link Between New Vehicles and Air Quality")
st.markdown(
    "Hold on, we're about to unravel a bit of a mystery here! Ever wondered how new vehicle registrations play a role in the air "
    "we breathe? Let's compare registrations and see which states are secretly impacted by changes in Air Quality Index (AQI). Ready "
    "to find out the hidden connection between our vehicles and the air we share?")

# VEHICLES COMPARISON

vehicles, df2 = load_vehicles()

veh_cols = st.columns([1, 3])
with veh_cols[0]:
    veh_state = st.selectbox("Select the state", df2.columns)

    veh_filter = st.selectbox("Select the type of filter", ['By Year', 'By Month'])
    if veh_filter == 'By Year':
        vehicles = vehicles.drop('month', axis=1).groupby('year').mean()
    else:
        vehicles = vehicles.drop('year', axis=1).groupby('month').mean()

    scaler_veh = st.selectbox('Select value Scaler', ['Standard Scaler', 'MinMax Scaler'], key='s')

    scaler_veh_obj = MinMaxScaler() if scaler_veh == 'MinMax Scaler' else StandardScaler()

    vehicles[vehicles.columns] = scaler_veh_obj.fit_transform(vehicles[vehicles.columns])

    # Calculate correlations and display the metric
    corr_veh = pd.Series()
    for i in df2.columns:
        corr_veh[i] = vehicles[i + '_veh'].corr(vehicles[i + '_aqi'])

    corr_veh.sort_values(ascending=False,inplace=True)
    corr_veh.fillna(0, inplace=True)
    st.metric("CORRELATION SCORE", corr_veh[veh_state].round(2), veh_state)

with veh_cols[1]:
    # Plot AQI trends for selected cities and correlation bar chart
    fig_veh = go.Figure()
    fig_veh.add_trace(
        go.Scatter(x=vehicles.index, y=vehicles[veh_state + '_aqi'], name="AQI Values", mode='lines',
                   line_shape='spline'))
    fig_veh.add_trace(
        go.Scatter(x=vehicles.index, y=vehicles[veh_state + '_veh'],
                   name="Vehicles Registrations", mode='lines', line_shape='spline'))
    fig_veh.update_layout(title=f'AQI Levels and Vehicle Registrations ({veh_state})',
                          xaxis_title='Month', yaxis_title='Air Quality Index Prediction')

    st.plotly_chart(fig_veh, use_container_width=True)

fig_veh_c = go.Figure()
fig_veh_c.add_trace(go.Bar(x=corr_veh.index, y=corr_veh.values))
fig_veh_c.update_layout(title=f'Correlation of states AQI with Vehicle Registrations ({veh_state})',
                        xaxis_title='Month', yaxis_title='Air Quality Index Prediction')

st.plotly_chart(fig_veh_c, use_container_width=True)

# Electricity


electricity, df_electricity = load_electricity()

elec_cols = st.columns([1, 3])
with elec_cols[0]:
    elec_state = st.selectbox("Select the state", df_electricity.columns, key='ss')

    elec_filter = st.selectbox("Select the type of filter", ['By Year', 'By Month'], key='ssa')
    if elec_filter == 'By Year':
        electricity = electricity.drop('month', axis=1).groupby('year').mean()
    else:
        electricity = electricity.drop('year', axis=1).groupby('month').mean()

    scaler_elec = st.selectbox('Select value Scaler', ['Standard Scaler', 'MinMax Scaler'], key='us')

    scaler_elec_obj = MinMaxScaler() if scaler_elec == 'MinMax Scaler' else StandardScaler()
    electricity[electricity.columns] = scaler_elec_obj.fit_transform(electricity[electricity.columns])

    # Calculate correlations and display the metric
    corr_elec = pd.Series()
    for i in df_electricity.columns:
        corr_elec[i] = electricity[i + '_elec'].corr(electricity[i + '_aqi'])

    corr_elec = pd.Series(corr_elec).sort_values(ascending=False)
    st.metric("CORRELATION SCORE", corr_elec[elec_state].round(2), elec_state)

with elec_cols[1]:
    # Plot AQI trends for selected cities and correlation bar chart
    fig_elec = go.Figure()
    fig_elec.add_trace(
        go.Scatter(x=electricity.index, y=electricity[elec_state + '_aqi'], name="AQI Values", mode='lines',
                   line_shape='spline'))
    fig_elec.add_trace(
        go.Scatter(x=electricity.index, y=electricity[elec_state + '_elec'],
                   name="Electricity Consumption", mode='lines', line_shape='spline'))
    fig_elec.update_layout(title=f'AQI Levels and Electricity Consumption ({elec_state})',
                           xaxis_title='Month', yaxis_title='Air Quality Index Prediction')

    st.plotly_chart(fig_elec, use_container_width=True)

fig_elec_c = go.Figure()
fig_elec_c.add_trace(go.Bar(x=corr_elec.index, y=corr_elec.values))
fig_elec_c.update_layout(title=f'Correlation of states AQI with Electricity Consumption ({elec_state})',
                         xaxis_title='Month', yaxis_title='Air Quality Index Prediction')

st.plotly_chart(fig_elec_c, use_container_width=True)
