import streamlit as st
import folium
from streamlit_folium import folium_static
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
    id = row['station']
    query = df[df['station'] == id]
    values = [query[query['pollutant_id'] == parameter]['pollutant_avg'].values[0] if not
    query[query['pollutant_id'] == parameter]['pollutant_avg'].empty else 0 for parameter in parameters]

    aqi_values = [calculate_aqi(parameter, value) for value, parameter in zip(values, parameters)]
    return max(aqi_values)


st.title("Real-Time AQI & Hotspots 💨")
st.subheader("Map Your Way to Cleaner Air")

st.markdown("""
**Filters:**
- Select a pollutant to focus on.
- Choose a state to view specific data.

**Visualizations:**
- Markers indicate AQI levels at individual locations.
- Heatmap visualizes AQI intensity across the map.
- Popupw shows AQI comparision by state and country.
""")

# Load Dataset
bar = st.progress(0.0, "Loading Data")
df = load_data()
bar.progress(0.08, "Cleaning Data")

df, country_avg, state_avg = clean_data(df)


bar.progress(0.2, "Applying Filters")

# Sidebar Filters

all_states_option = ['All States'] + list(df['state'].unique())
selected_state = st.sidebar.selectbox('Select State:', all_states_option)
selected_pollutant = st.sidebar.selectbox('Select Pollutant:', df['pollutant_id'].unique())

# Filter DataFrame based on selected options
if selected_state == 'All States':
    filtered_df = df[df['pollutant_id'] == selected_pollutant]

else:
    filtered_df = df[(df['state'] == selected_state) & (df['pollutant_id'] == selected_pollutant)]


# Create Map 1 : HEAT MAP & MARKERS
bar.progress(0.22, "Setting up Maps")
m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# FeatureGroup for markers
fg_markers = folium.FeatureGroup(name="Markers")
m.add_child(fg_markers)

# Html, Css for datacard :
# with open('popup.html', 'r') as f:
#     card_html = f.read()
with open('pages/style.css', 'r') as f:
    card_css = f.read()

l = filtered_df.shape[0]
print(l)
# Add CircleMarkers with Popup and Tooltip to the FeatureGroup
for idx, i in enumerate(filtered_df.index):
    bar.progress(0.22 + ((idx + 1) / l) / 2, "Adding Markers")
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

# FeatureGroup for heatmap
fg_heatmap = folium.FeatureGroup(name="Heatmap", overlay=True)
m.add_child(fg_heatmap)

# Update the map based on checkbox status
heatmap_data = filtered_df[['latitude', 'longitude', 'pollutant_avg']].values.tolist()
HeatMap(heatmap_data, radius=25).add_to(fg_heatmap)

# Add LayerControl to manage visibility
folium.LayerControl(collapsed=False).add_to(m)
bar.progress(0.8, "you are just there... few seconds more")
# Display Map using folium_static
folium_static(m)

# CREATE MAP2 : STATE AVERGE
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
bar.progress(1.0, "hurray!")
bar.empty()