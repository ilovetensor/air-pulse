import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import base64

from utils.constants import state_dict, state_dict_reverse, month_map


# conn = st.connection("snowflake")


st.set_page_config(layout='wide', initial_sidebar_state='expanded',
                   page_icon='🪨', page_title='AQI Future Predictions', )
)


@st.cache_data()
def load_model():
    models = joblib.load("nb-playground/weighted_models.pkl")
    weights = joblib.load("nb-playground/model_weights.pkl")
    return models, weights


@st.cache_data()
def load_data():
    df = pd.read_csv("nb-playground/historical_data_cleaned.csv")
    return df
    # session = conn.session()
    # table = session.table('AIRPULSE.HISTORICAL_DATA.AQI_PAST').to_pandas()
    # table.columns = ['state', 'year', 'month', 'aqi']
    # return table


def weighted_average_prediction(models: list, weights: list, predict_df: pd.DataFrame):
    preds = pd.DataFrame()
    for i, m in enumerate(models):
        preds[i] = m.predict(predict_df)
    preds['weighted_pred'] = (preds * weights).sum(axis=1) / sum(weights)
    return preds['weighted_pred']


def get_image_as_base64(file):
    with open(file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def get_image_as_base64(file):
    with open(file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

with open("style_page1.css") as file:
    st.markdown(f'<style>{file.read()}</style>', unsafe_allow_html=True)

img = get_image_as_base64('templates/1567736.png')
st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"]{{
    background-image: url("data:image/png;base64,{img}");
    background-color: grey;
    background-size: cover;
    opacity: blur;
    }}
    </style>
""", unsafe_allow_html=True)

st.title("AQI Levels Forecasting ")
st.subheader("Take actions based of upcoming trends")
st.markdown("""The `weighted average` model is trained over the historical dataset from 1990 to 2015 for all states,
capturing the monthly and yearly trends to make better predictions. 
The data has training accuracy of `98%` and test accuracy for 1 year of data as `80.1%`. 

- **Input Parameters** : `state name`, `year`, `month`

- **Output** : `AQI value`
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
# print(models, weights)

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

kpi_head = st.columns([1, 1])
with kpi_head[0]:
    st.subheader("Some Facts over the Years  : ")
with kpi_head[1]:
    kpi_yr = st.slider("Select year range", 1990, 2015, (2000, 2015), 1)

st.markdown("---")
col1, col2, col3, = st.columns(3)
lb = df[df.year <= kpi_yr[1]]
kpi_df = lb[lb.year >= kpi_yr[0]]
lowest = kpi_df[['month', 'aqi']].groupby('month').mean().values
low, low_val = np.argmin(lowest), min(lowest)[0]
col1.metric("CLEANEST MONTH IN THE YEAR ", month_map[low] + " 🍃", int(low_val), 'normal',
            help='The month which has least levels of AQI in an year')

hi, hi_val = np.argmax(lowest), max(lowest)[0]
col2.metric("WORST MONTH OF THE YEAR ", month_map[hi] + " 😷", int(hi_val), 'inverse',
            help='The month which has highest levels of AQI in a year')

hi_year = kpi_df[['year', 'aqi']].groupby('year').mean().values
hi_y, hi_y_val = np.argmax(hi_year), max(hi_year)[0]
col3.metric("MOST POLLUTED YEAR IN HISTORY ", str(int(hi_y) + kpi_yr[0]) + " ⌛",
            int(hi_y_val), 'inverse', help='This year had the highest levels of AQI on average', )
st.markdown('---')

col1, col2, _ = st.columns(3)

lowest_c = kpi_df[['state', 'aqi']].groupby('state').mean().values
low_c, low_c_val = np.argmin(lowest_c), min(lowest_c)[0]
col1.metric("CLEANEST CITY ", state_dict_reverse[low_c] + " 🪷", int(low_c_val),
            help='The city with lowest levels of AQI in past years on avg.')

hi_c, hi_c_val = np.argmax(lowest_c), max(lowest_c)[0]
col2.metric("WORST CITY ", state_dict_reverse[hi_c] + " 🚦", int(hi_c_val), 'inverse',
            help='The city with highest levels of AQI in past years on avg.')

# Load additional data or perform preprocessing if needed
# For example, you can load a dataset with information about major cities for better city selection dropdown options.

# Add a section for selecting major cities
st.subheader("Compare Cities")
selected_cities = st.multiselect("Select Cities", list(state_dict_reverse.values()), default=["Delhi", "Rajasthan"])

# Filter data for selected cities
selected_cities_data = df[df['state'].isin([state_dict[c] for c in selected_cities])]
# Plot AQI trends for selected cities
fig_cities = go.Figure()

for city in selected_cities:
    city_data = selected_cities_data[selected_cities_data['state'] == state_dict[city]]
    city_data = city_data[['month', 'aqi']].groupby('month').mean().reset_index()
    fig_cities.add_trace(go.Line(x=city_data['month'], y=city_data['aqi'], mode='lines', name=city))

fig_cities.update_layout(title=f"AQI Trends Comparison for Selected Cities", xaxis_title='Month',
                         yaxis_title='Air Quality Index')
st.plotly_chart(fig_cities, use_container_width=True)

st.subheader("Heatmap of AQI Levels")
year_lapse = df[df['year'] >= 2000]
heatmap_data = year_lapse.pivot_table(values='aqi', index='state', columns='year', aggfunc='mean')

fig_heatmap = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=list(heatmap_data.columns),
    y=list(heatmap_data.index.map(state_dict_reverse)),
    colorscale='Magma',  # Choose your preferred colorscale
    hoverongaps=False,
    hoverinfo='z',
    colorbar=dict(title='AQI Levels')
))

fig_heatmap.update_layout(title="Heatmap of AQI Levels Across States (2000 and later)",
                          xaxis_title="Year",
                          yaxis_title="State",
                          xaxis_nticks=len(heatmap_data.columns),
                          yaxis_nticks=len(heatmap_data.index),
                          height=600
                          )

st.plotly_chart(fig_heatmap, use_container_width=True)

# Provide summary statistics for the selected year
# st.subheader("Summary Statistics for the Selected Year")
# selected_year_data = df[df['year'] == year_selected]
# st.dataframe(selected_year_data.describe())

# # Add a download button for the predictions dataframe
# st.subheader("Download Predictions Data")
# if st.button("Download Predictions CSV"):
#     st.download_button("Download Predictions CSV", pred_df.to_csv(), key="predictions_csv")
#
