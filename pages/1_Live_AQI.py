import streamlit as st
import folium
from streamlit_folium import folium_static, st_folium
from folium.plugins import HeatMap
import requests
import pandas as pd
from io import StringIO
from io import BytesIO
from constants import api_key
from drawtools import create_map, create_icon, create_datacard
from calculations import calculate_aqi
import geopandas
from branca.colormap import linear
from streamlit_card import card

with open('pages/style.css', 'r') as f:
    card_css = f.read()


@st.cache_data()
def load_data():
    df = pd.read_csv("./nb-playground/dataset/live_data.csv")
    # response_API = requests.get(
    #     f"https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69?api-key={api_key}&format=csv&limit=9000")
    # data = response_API.text
    # df = pd.read_csv(StringIO(data))
    df.fillna(0, inplace=True)
    return df


@st.cache_data()
def clean_data(df):
    df.dropna(inplace=True)
    df['state'] = df.apply(lambda row: " ".join(" ".join(row.state.split("_")).split(" ")), axis=1)
    aqi = df.apply(lambda row: calculate_overall_aqi(row), axis=1)
    df2 = df.copy()
    df2['pollutant_avg'] = aqi
    df2['pollutant_min'] = aqi
    df2['pollutant_max'] = aqi
    df2['pollutant_id'] = 'AQI'

    df = pd.concat([df, df2], ignore_index=True).reset_index()
    df.drop_duplicates(inplace=True)
    country = df.groupby('pollutant_id')['pollutant_avg'].mean()
    state = df.groupby(['state', 'pollutant_id'])['pollutant_avg'].mean()
    return df, country, state


def calculate_overall_aqi(row):
    parameters = ['SO2', 'OZONE', 'CO', 'PM2.5', 'NO2', 'NH3', 'PM10']
    parameters = ['PM2.5', 'PM10']

    id = row['station']
    query = df[df['station'] == id]
    values = [query[query['pollutant_id'] == parameter]['pollutant_avg'].values[0] if not
    query[query['pollutant_id'] == parameter]['pollutant_avg'].empty else 0 for parameter in parameters]

    aqi_values = [calculate_aqi(parameter, value) for value, parameter in zip(values, parameters)]
    return max(aqi_values)


st.set_page_config(layout="wide")
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
    selected_pollutant = st.selectbox('Select Pollutant:', df['pollutant_id'].unique())
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
    filtered_df = df[df['pollutant_id'] == selected_pollutant]
else:
    filtered_df = df[(df['state'] == selected_state) & (df['pollutant_id'] == selected_pollutant)]

# Create Map 1 : HEAT MAP & MARKERS

m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, width=1000)

# FeatureGroup for markers
fg_markers = folium.FeatureGroup(name="Markers", lay=True)

l: int = filtered_df.shape[0]
print(l)

# Add CircleMarkers with Popup and Tooltip to the FeatureGroup
for idx, i in enumerate(filtered_df.index):
    # bar.progress(0.22 + ((idx + 1) / l) / 2, "Adding Markers")
    pollutant_avg = filtered_df.loc[i]['pollutant_avg']
    pollutant_id = filtered_df.loc[i]['pollutant_id']

    folium.Marker(
        location=filtered_df.loc[i][['latitude', 'longitude']].values,
        icon=create_icon(pollutant_avg, pollutant_id),
        popup=create_datacard(filtered_df.loc[i]['station'],
                              country_avg,
                              state_avg,
                              filtered_df.loc[i]['pollutant_avg'],
                              filtered_df.loc[i]['pollutant_id'],
                              filtered_df.loc[i]['state'],
                              card_css
                              ),
        tooltip=f"{filtered_df.loc[i]['city']} - {filtered_df.loc[i]['pollutant_avg']}"
    ).add_to(fg_markers)

# Heatmap - FeatureGroup
fg_heatmap = folium.FeatureGroup(name="Heatmap", overlay=True)
heatmap_data = filtered_df[['latitude', 'longitude', 'pollutant_avg']].values.tolist()
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
    st_folium(m, width=1000, feature_group_to_add=fgs)

st.markdown('---')

# KPI METRICS : VALUES AND FACTS
kpis = st.columns(4)
with kpis[0]:
    st.metric('sdf','sdf','sdf')

st.markdown('---')

# CREATE MAP2 : STATES AVERAGE
@st.cache_data()
def load_state_data():
    json_file = geopandas.read_file('nb-playground/dataset/india.json')
    json_file = json_file[json_file['NAME_1'].isin(filtered_state.index)]
    json_file['avg'] = json_file['NAME_1'].map(filtered_state.to_dict())
    return json_file


filtered_state = state_avg.xs(selected_pollutant, level='pollutant_id')

m2 = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
ind_geojson = load_state_data()

# folium.GeoJson(ind_geojson).add_to(m2)


st.dataframe(filtered_state)

st.text(filtered_state.to_dict())

colormap = linear.YlGn_09.scale(
    min(filtered_state.values), max(filtered_state.values)
)
st.write(max(filtered_state))

popup = folium.GeoJsonPopup(fields=["NAME_1"], )
tooltip = folium.GeoJsonTooltip(
    fields=["NAME_1", "avg"],
    aliases=["State:", f"{selected_pollutant} :"],
    localize=True,
    sticky=False,
    labels=True,
    style="""
        background-color: #F0EFEF;
        border: 2px solid black;
        border-radius: 3px;
        box-shadow: 3px;
    """,
    max_width=800,
)
folium.GeoJson(
    ind_geojson,
    name="pollutant avg",
    style_function=lambda feature: {
        "fillColor": colormap(filtered_state[feature["properties"]["NAME_1"]]),
        "color": "black",
        "weight": 1,
        "dashArray": "5, 5",
        "fillOpacity": 0.9,
    },
    highlight_function=lambda feature: {
        "fillColor": (
            colormap(filtered_state[feature["properties"]["NAME_1"]] - 2)
        ),
    },
    # popup=popup,
    tooltip=tooltip,
    popup_keep_highlighted=True
).add_to(m2)

colormap.caption = " color scale"
colormap.add_to(m2)

folium.LayerControl().add_to(m2)
folium_static(m2)

