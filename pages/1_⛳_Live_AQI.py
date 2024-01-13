from io import StringIO
import base64
import folium
import pandas as pd
import requests
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from utils.drawtools import create_icon, create_datacard
from utils.calculations import calculate_aqi
import geopandas
from branca.colormap import linear
import numpy as np
import plotly.graph_objects as go

api_key = st.secrets["api_key"]
def get_image_as_base64(file):
    with open(file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache_data()
def load_data():
    # df = pd.read_csv("./nb-playground/dataset/live_data.csv")
    response_API = requests.get(
        f"https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69?api-key={api_key}&format=csv&limit=9000")
    data = response_API.text
    df = pd.read_csv(StringIO(data))
    df.fillna(0, inplace=True)

    return df


@st.cache_data()
def clean_data(df):
    df.dropna(inplace=True)
    df['state'] = df.apply(lambda row: " ".join(" ".join(row.state.split("_")).split(" ")), axis=1)
    df = df.pivot(columns='pollutant_id', index=['state', 'city', 'station', 'latitude', 'longitude'],
                  values='pollutant_avg')
    df.fillna(df.mean(), inplace=True)
    aqi = df.apply(lambda row: calculate_overall_aqi(row), axis=1)
    df['AQI'] = aqi
    df.drop_duplicates(inplace=True)
    pollutants = df.columns
    country = df.mean()
    df = df.reset_index()
    state = df.groupby('state')[pollutants].mean()
    return df, country, state


def calculate_overall_aqi(row):
    parameters = row.index
    parameters = ['PM2.5', 'PM10']
    aqi_values = [calculate_aqi(parameter, value) for value, parameter in zip(row[parameters].values, parameters)]
    return int(max(aqi_values))


with open('pages/style.css', 'r') as f:
    card_css = f.read()

st.set_page_config(layout="wide",
                   initial_sidebar_state="collapsed",
                   page_icon='⛳', page_title='Live AQI',)

with open("style_page1.css") as file:
    st.markdown(f'<style>{file.read()}</style>', unsafe_allow_html=True)

img = get_image_as_base64('templates/58328.jpg')


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
img = get_image_as_base64('templates/1761746.jpg')


st.title("Real-Time AQI & Hotspots 💨")
st.subheader("Map Your Way to Cleaner Air")

st.markdown("""

 **Hey there, air-conscious citizen!** 

We know you deserve to breathe easy, but deciphering those government reports can leave you gasping for answers. That's where **Live AQI Mapping** comes in – your real-time window into India's air quality, state by state. ️

**Ditch the stuffy reports and dive into an interactive playground of air data.**  Craving pollution heatmaps that 
pulse with life? Charts that tell tales of the wind? Graphs that waltz with seasonal trends? We've got them all, 
ready to transform confusing stats into crystal-clear insights.

**And the best part?** This vibrant air portrait is painted with **live data**, streaming straight from data.gov.in. 
⚡️ No stale stats, no murky sources, just pure, unfiltered air truth.

**Ready to take a deep breath of confidence?**  Go ahead, explore your air, your way. Welcome to the world of clear skies and informed breaths. ✨

**Welcome to Live AQI Mapping.** 

""")

# Load Dataset
df = load_data()
# Clean Dataset
df, country_avg, state_avg = clean_data(df)

# Controllers
maps_cols = st.columns([3, 1])
with maps_cols[1]:
    st.markdown(
        """
        # Live AQI
        Select the `state filter`, `pollutant` to analyse and make changes in `heatmap controlls` 
        to get the best results from the map 🍕
        """)
    all_states_option = ['All States'] + list(df['state'].unique())
    selected_state = st.selectbox('Select State:', all_states_option)
    selected_pollutant = st.selectbox('Select Pollutant:', df.columns[-8:])
    checks = st.columns(2)
    with checks[0]:
        is_heatmap = st.checkbox('Heatmap', True)
    with checks[1]:
        is_marker = st.checkbox('Markers', True)
    st.markdown("**Heatmap Controlls**")
    radius = st.slider("Heatmap Radius", 1, 40, 25)
    blur = st.slider("Heatmap Blur", 1, 40, 25)
    st.info('Data Last Updated : 12 minutes ago')

# Filter DataFrame based on selected options
if selected_state == 'All States':
    filtered_df = df
else:
    filtered_df = df[df['state'] == selected_state]

# Create Map 1 : HEAT MAP & MARKERS

m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, width=1000)

# FeatureGroup for markers
fg_markers = folium.FeatureGroup(name="Markers", overlay=True)

l: int = filtered_df.shape[0]
# Add CircleMarkers with Popup and Tooltip to the FeatureGroup
for idx, i in enumerate(filtered_df.index):
    # bar.progress(0.22 + ((idx + 1) / l) / 2, "Adding Markers")
    pollutant_avg = filtered_df.loc[i][selected_pollutant]
    pollutant_id = selected_pollutant

    folium.Marker(
        location=filtered_df.loc[i][['latitude', 'longitude']].values,
        icon=create_icon(pollutant_avg, pollutant_id),
        popup=create_datacard(filtered_df.loc[i]['station'],
                              country_avg,
                              state_avg,
                              pollutant_avg,
                              pollutant_id,
                              filtered_df.loc[i]['state'],
                              card_css
                              ),
        tooltip=f"{filtered_df.loc[i]['city']} - {pollutant_avg}"
    ).add_to(fg_markers)

# Heatmap - FeatureGroup
fg_heatmap = folium.FeatureGroup(name="Heatmap", overlay=True)
heatmap_data = filtered_df[['latitude', 'longitude', selected_pollutant]].values.tolist()
HeatMap(heatmap_data, radius=radius, blur=blur).add_to(fg_heatmap)

# Add LayerControl to manage visibility
layers = folium.LayerControl(collapsed=False)

# Checkbox functionality
if is_marker and is_heatmap:
    fgs = [fg_heatmap, fg_markers]
if is_marker and not is_heatmap:
    fgs = [fg_markers]
if is_heatmap and not is_marker:
    fgs = [fg_heatmap]
if not is_heatmap and not is_marker:
    fgs = []

# Displaying map with st_folium
with maps_cols[0]:
    st_folium(m, width=1000, feature_group_to_add=fgs, returned_objects=[])

#


# KPI METRICS : VALUES AND FACTS

# st.markdown('---')
st.markdown('### Some AQI live facts ')

aqi_states = state_avg['AQI']
kpis = st.columns(4)
with kpis[0]:
    max_s = np.argmax(aqi_states)
    st.metric("STATE WITH WORST AQI", str(aqi_states.index[max_s]) + "  ⚠", aqi_states.values[max_s],
              delta_color="inverse")
with kpis[1]:
    min_s = np.argmin(aqi_states)
    st.metric("STATE WITH LEAST AQI", str(aqi_states.index[min_s]) + "  🐦", aqi_states.values[min_s].round(),
              delta_color="normal")
with kpis[2]:
    min_c = aqi_states[aqi_states < 150]
    st.metric("TOTAL GOOD STATES ", str(len(min_c)) + "🌷", "Below 150", delta_color="normal",
              help=", ".join(min_c.index))
with kpis[3]:
    max_c = aqi_states[aqi_states > 300]
    st.metric("HAZARDOUS STATES ", str(len(max_c)) + "☢", "Above 300", delta_color="inverse",
              help=", ".join(max_c.index))
st.markdown('---')

# CREATE MAP2 : STATES AVERAGE

st.markdown("## Explore Your State's Air Quality")
st.markdown("""Dive deep into the air quality of your state with our dedicated filters, providing you with the most useful insights about your local regions.

**Click on your state** to reveal a visual representation of cities within that state, accompanied by key visuals that highlight crucial air quality information.

Lets interact ....""")

state_cols = st.columns(2)

with state_cols[1]:
    select_cols = st.columns(2)
    with select_cols[0]:
        s_pollutant = st.selectbox("Select pollutant for map", df.columns[-8:])

    with select_cols[1]:
        s_state = st.selectbox('Selected State:', list(df['state'].unique()), )

    heatmap_data = df[df.state == s_state].groupby('city')[df.columns[-8:]].mean().astype('int')

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=list(heatmap_data.columns),
        y=list(heatmap_data.index),
        colorscale='Aggrnyl',  # Choose your preferred colorscale
        hoverongaps=False,
        hoverinfo='z',
        colorbar=dict(title='AQI Levels')
    ))
    #
    fig_heatmap.update_layout(title="Live City vs Pollutant Values Heatmap",
                              xaxis_title="Year",
                              yaxis_title="State",
                              xaxis_nticks=len(heatmap_data.columns),
                              yaxis_nticks=len(heatmap_data.index),
                              height=600
                              )

    st.plotly_chart(fig_heatmap, use_container_width=True)


@st.cache_data()
def load_state_data():
    json_file = geopandas.read_file('nb-playground/dataset/india.json')
    json_file = json_file[json_file['NAME_1'].isin(filtered_state.index)]
    for col in filtered_state.columns:
        json_file[col] = json_file['NAME_1'].map(filtered_state[col].to_dict())
    return json_file


filtered_state = state_avg

m2 = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
ind_geojson = load_state_data()

colormap = linear.RdPu_04.scale(
    min(filtered_state[s_pollutant]), max(filtered_state[s_pollutant])
)

popup = folium.GeoJsonPopup(fields=["NAME_1"], )
tooltip = folium.GeoJsonTooltip(
    fields=["NAME_1", s_pollutant],
    aliases=["State:", f"{s_pollutant} :"],
    localize=True,
    sticky=True,
    labels=True,
    style="""
        background-color: #F0EFEF;
        border-radius: 6px;
        box-shadow: 3px;
        font-size: 13px;
    """,
    max_width=800,
)

fg_geojson = folium.FeatureGroup(name='geojson', overlay=True)

folium.GeoJson(
    ind_geojson,
    name="pollutant avg",
    style_function=lambda feature: {
        "fillColor": colormap(filtered_state[s_pollutant][feature["properties"]["NAME_1"]]),
        "color": "black",
        "weight": 1,
        "dashArray": "5, 5",
        "fillOpacity": 1,
    },
    highlight_function=lambda feature: {
        "fillColor": (
            colormap(filtered_state[s_pollutant][feature["properties"]["NAME_1"]] - 2)
        ),
    },
    # popup=popup,
    tooltip=tooltip,
    popup_keep_highlighted=True,

).add_to(fg_geojson)

fg_colormap = folium.FeatureGroup(name="colormap")
colormap.caption = " color scale"

with state_cols[0]:
    st_folium(m2, 'k', width=700, height=660, feature_group_to_add=[fg_geojson, fg_colormap], returned_objects=[])

# Data Query : Bar Graph

st.subheader("Query the Data ")

query_c = st.columns(2)
with query_c[0]:
    q_method = st.selectbox("Select the query method: ", ['State Level', 'City Level'], )
    if q_method == "State Level":
        query_filter = state_avg

with query_c[1]:
    if q_method == 'City Level':
        q_state = st.selectbox("Select State", df.state.unique(), disabled=False)
        query_filter = df[df.state == q_state].groupby('city')[df.columns[-8:]].mean()
st.dataframe(query_filter, width=1400)
filename = "_".join(q_method.split()) + "Air_Quality_Data.csv"
st.download_button('Download CSV', query_filter.to_csv(), filename, )
