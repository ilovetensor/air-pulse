import streamlit as st
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
import requests
import pandas as pd
from io import StringIO
from constants import api_key


@st.cache_data()
def load_data():
    response_API = requests.get(
        f"https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69?api-key={api_key}&format=csv&limit=1000")
    data = response_API.text
    df = pd.read_csv(StringIO(data))
    df.fillna(0, inplace=True)
    return df


def create_datacard(city, pollutant_min, pollutant_max, pollutant_avg):
    datacard_html = f"""
    <div style="width: 200px; height: 150px; padding: 10px; background-color: white; ">
        <h4>{city}</h4>
        <p><b>Min:</b> {pollutant_min}</p>
        <p><b>Max:</b> {pollutant_max}</p>
        <p><b>Avg:</b> {pollutant_avg}</p>
    </div>
    """
    return folium.Popup(folium.Html(datacard_html, script=True), max_width=300)

def create_icon(number, color):
    icon_plane = folium.plugins.BeautifyIcon(
        icon="plane",
        icon_shape='rectangle',
        border_color=color,
        background_color=color,
        text_color='#FFFFFF',
        number=number,
        iconStyle="border-radius:5px;",
        inner_icon_style="margin-top:1;padding:0px 1px;border-radius:3px;"
    )
    return icon_plane

st.title('Live AQI over the map 🌍')

# df = load_data()
df = pd.read_csv("./nb-playground/dataset/live_data.csv")
df.dropna(inplace=True)


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
st.session_state['fg_markers'] = folium.FeatureGroup(name="Markers")
m.add_child(st.session_state['fg_markers'])

# Add CircleMarkers with Popup and Tooltip to the FeatureGroup
for i in filtered_df.index:
    pollutant_avg = filtered_df.loc[i]['pollutant_avg']
    color = 'red' if pollutant_avg > 50 else 'orange' if pollutant_avg > 25 else 'green'



    folium.Marker(location=filtered_df.loc[i][['latitude', 'longitude']].values,
                        icon=create_icon(pollutant_avg, color),
                        popup=create_datacard(filtered_df.loc[i]['station'],
                                              filtered_df.loc[i]['pollutant_min'],
                                              filtered_df.loc[i]['pollutant_max'],
                                              filtered_df.loc[i]['pollutant_avg']),
                        tooltip=f"{filtered_df.loc[i]['city']} - {filtered_df.loc[i]['pollutant_avg']}").add_to(
        st.session_state['fg_markers'])

# FeatureGroup for heatmap
st.session_state['fg_heatmap'] = folium.FeatureGroup(name="Heatmap", overlay=True)
m.add_child(st.session_state['fg_heatmap'])


# Update the map based on checkbox status
heatmap_data = filtered_df[['latitude', 'longitude', 'pollutant_avg']].values.tolist()
HeatMap(heatmap_data).add_to(st.session_state['fg_heatmap'])

# Add LayerControl to manage visibility
folium.LayerControl(collapsed=True).add_to(m)


# Display Map using folium_static
folium_static(m)


# Display DataFrame
st.dataframe(df.head())