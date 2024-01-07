import base64

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
import requests
import pandas as pd
from io import StringIO
from constants import api_key
import streamlit.components.v1 as components
import vincent
import json
from io import BytesIO




@st.cache_data()
def load_data():
    response_API = requests.get(
        f"https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69?api-key={api_key}&format=csv&limit=9000")
    data = response_API.text
    df = pd.read_csv(StringIO(data))
    df.fillna(0, inplace=True)
    return df


def create_map(id, lst):
    fig = plt.figure(figsize=(2,2))
    ax = fig.add_subplot(1,1,1)
    ax.bar(id, lst)
    tmp = BytesIO()
    fig.savefig(tmp, format='png')
    encode = base64.b64encode(tmp.getvalue()).decode('utf-8')
    plt.close()
    return encode


def create_datacard(city, pollutant_min, pollutant_max, pollutant_avg, pollutant_id, state):
    map_encode = create_map(['country', 'state', 'city'],
                            [country_avg[pollutant_id], state_avg[state, pollutant_id], pollutant_avg],
                            )
    datacard_html = f"""
    <div style="background-color: white; padding: 5px; padding-bottom: 0px;width: 200px; ">
        <h4>{city}</h6>
    </div>
    """ + '<img src=\'data:image/png;base64,{}\'>'.format(map_encode)

    # Customize your color palette

    # Customize your color palette
    color_palette = ["#1f78b4", "#33a02c", "#e31a1c"]

    popup = folium.Popup(folium.Html(datacard_html, script=True), max_width=300)

    # Create a bar chart with Vincent
    bar = vincent.Bar(
        [country_avg[pollutant_id], state_avg[state, pollutant_id], pollutant_avg],
        width=180,
        height=150,
    )
    data = json.loads(bar.to_json())

    # Add labels and titles
    data['axes'][0]['title'] = 'country avg    state avg    city avg'
    data['axes'][1]['title'] = pollutant_id + " levels"
    data['title'] = 'Pollutant Comparison'

    # Adjust color palette
    data['scales'][2]['range'] = color_palette

    # Display data values on top of the bars
    for mark in data['marks'][0]['marks']:
        mark['properties']['enter']['text'] = {"field": "data.val", "scale": "y", "format": ".2f"}

    custom_labels = ["Country", "State", "Location"]
    data['axes'][0]['type'] = "ordinal"
    data['axes'][0]['scale'] = "x"
    data['axes'][0]['values'] = list(range(len(custom_labels)))
    data['data'][0]['values'][0]['idx'] = "a"
    data['data'][0]['values'][1]['idx'] = "d"
    data['data'][0]['values'][2]['idx'] = "aa"


    # folium.Vega(data, width="20%", height="20%").add_to(popup)
    return popup


import bisect

def create_icon(pollutant_avg, pollutant_id):
    if pollutant_id == "SO2":
        breakpoints = [40, 80, 180, 380]
        colors = ["green", "orange", "red", "black"]
    elif pollutant_id == "OZONE":
        breakpoints = [100, 160, 200, 240]
        colors = ["green", "orange", "red", "black"]
    elif pollutant_id == "CO":
        breakpoints = [0, 40, 80, 150]
        colors = ["green", "orange", "red", "black"]
    elif pollutant_id == "PM2.5":
        breakpoints = [15, 35, 55, 150]
        colors = ["green", "orange", "red", "black"]
    elif pollutant_id == "NO2":
        breakpoints = [40, 80, 180, 380]
        colors = ["green", "orange", "red", "black"]
    elif pollutant_id == "NH3":
        breakpoints = [200, 400, 800, 1500]
        colors = ["green", "orange", "red", "black"]
    elif pollutant_id == "PM10":
        breakpoints = [50, 100, 150, 350]
        colors = ["green", "orange", "red", "black"]
    elif pollutant_id == "AQI":
        breakpoints = [0, 50, 150, 200, 300]
        colors = ["green", "yellow", "orange","red", "black"]
    else:
        # Handle any other pollutant IDs as needed
        breakpoints = []
        colors = []

    color = colors[max(0, bisect.bisect(breakpoints, pollutant_avg) - 1)]
    text='#FFFFFF'
    if color == "yellow": text='#000000'

    icon_plane = folium.plugins.BeautifyIcon(
        icon="plane",
        border_color=color,
        background_color=color,
        text_color=text,
        number=int(pollutant_avg),
        icon_shape="marker",
        inner_icon_style="margin-top:1;padding:0px 1px;border-radius:3px;"
    )
    return icon_plane

def calculate_aqi(parameter, concentration):
    # Define specific AQI breakpoints and corresponding AQI values for each parameter based on Indian standards
    parameter_breakpoints = {
        'SO2': [0, 40, 80, 380, 800, 1600],
        'OZONE': [0, 50, 100, 168, 209, 748],
        'CO': [0, 30, 60, 90, 180, 280],
        'PM2.5': [0, 30, 60, 90, 120, 250],
        'NO2': [0, 40, 80, 180, 280, 400],
        'NH3': [0, 200, 400, 800, 1200, 1800],
        'PM10': [0, 50, 100, 250, 350, 430],
        'AQI': [0, 50, 100, 250, 350, 430],
    }

    aqi_values = [0, 50, 100, 200, 300, 400, 500]

    for i in range(len(parameter_breakpoints[parameter]) - 1):
        if parameter_breakpoints[parameter][i] <= concentration <= parameter_breakpoints[parameter][i + 1]:
            break

    return int(((aqi_values[i + 1] - aqi_values[i]) / (
                parameter_breakpoints[parameter][i + 1] - parameter_breakpoints[parameter][i])) * (
                concentration - parameter_breakpoints[parameter][i]) + aqi_values[i])

def calculate_overall_aqi(row):
    parameters = ['SO2', 'OZONE', 'CO', 'PM2.5', 'NO2', 'NH3', 'PM10']
    id = row['station']
    query = df[df['station'] == id]
    values = [query[query['pollutant_id'] == parameter]['pollutant_avg'].values[0] if not
    query[query['pollutant_id'] == parameter]['pollutant_avg'].empty else 0 for parameter in parameters]

    aqi_values = [calculate_aqi(parameter, value) for value,parameter in zip(values, parameters)]
    return max(aqi_values)

st.title('Live AQI over the map 🌍')



# df = load_data()
df = pd.read_csv("./nb-playground/dataset/live_data.csv")
df.dropna(inplace=True)

aqi = df.apply(lambda row: calculate_overall_aqi(row), axis=1)
df2 = df.copy()
df2['pollutant_avg'] = aqi
df2['pollutant_min'] = aqi
df2['pollutant_max'] = aqi
df2['pollutant_id'] = 'AQI'

df = pd.concat([df,df2], ignore_index=True).reset_index()
df.drop_duplicates(inplace=True)

country_avg = df.groupby('pollutant_id')['pollutant_avg'].mean()
state_avg = df.groupby(['state','pollutant_id'])['pollutant_avg'].mean()



# Sidebar Filters
all_states_option = ['All States'] + list(df['state'].unique())
selected_state = st.sidebar.selectbox('Select State:', all_states_option)
selected_pollutant = st.sidebar.selectbox('Select Pollutant:', df['pollutant_id'].unique())

# Filter DataFrame based on selected options
if selected_state == 'All States':
    filtered_df = df[df['pollutant_id'] == selected_pollutant]
else:
    filtered_df = df[(df['state'] == selected_state) & (df['pollutant_id'] == selected_pollutant)]

# Create Map
m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# FeatureGroup for markers
fg_markers = folium.FeatureGroup(name="Markers")
m.add_child(fg_markers)

# Add CircleMarkers with Popup and Tooltip to the FeatureGroup
for i in filtered_df.index:
    pollutant_avg = filtered_df.loc[i]['pollutant_avg']
    pollutant_id = filtered_df.loc[i]['pollutant_id']


    folium.Marker(
        location=filtered_df.loc[i][['latitude', 'longitude']].values,
        icon=create_icon(pollutant_avg, pollutant_id),
        popup=create_datacard(filtered_df.loc[i]['station'],
                              filtered_df.loc[i]['pollutant_min'],
                              filtered_df.loc[i]['pollutant_max'],
                              filtered_df.loc[i]['pollutant_avg'],
                              filtered_df.loc[i]['pollutant_id'],
                              filtered_df.loc[i]['state']
                              ),
        tooltip=f"{filtered_df.loc[i]['city']} - {filtered_df.loc[i]['pollutant_avg']}"

    ).add_to(fg_markers)


# FeatureGroup for heatmap
fg_heatmap = folium.FeatureGroup(name="Heatmap", overlay=True)
m.add_child(fg_heatmap)


# Update the map based on checkbox status
heatmap_data = filtered_df[['latitude', 'longitude', 'pollutant_avg']].values.tolist()
HeatMap(heatmap_data, radius=25).add_to(fg_heatmap)


# Add LayerControl to manage visibility
folium.LayerControl(collapsed=True).add_to(m)


# Display Map using folium_static
folium_static(m)

# Display DataFrame
st.dataframe(df.head())
st.dataframe(df['pollutant_id'].unique())
