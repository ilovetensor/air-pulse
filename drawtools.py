import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import BytesIO
import base64
import folium
import bisect


def create_map(id, lst):
    id2 = [str(int(s)) for s in lst]
    fig, ax = plt.subplots(figsize=(2.3,1.5))
    mybars = ax.barh(id2, lst, color="#FFC600")

    for i, bar in enumerate(mybars):
        yval = id[i]
        ax.text(1, bar.get_y() + bar.get_height() / 2, f"{yval}", va='center', fontsize=9)

    # get rid of the frame
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # get rid of ticks
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelright=False, labelbottom=False)
    plt.rc('ytick', labelsize=6)

    tmp = BytesIO()
    fig.savefig(tmp, format='png')
    encode = base64.b64encode(tmp.getvalue()).decode('utf-8')
    plt.close()
    return encode




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

def create_datacard(city, country_avg, state_avg, pollutant_avg, pollutant_id, state):
    map_encode = create_map(['country avg.', 'state avg.', 'city avg.'],
                            [country_avg[pollutant_id], state_avg[state, pollutant_id], pollutant_avg],
                            )
    datacard_html = f"""
    <div style="background-color: white; padding: 5px; padding-bottom: 0px; margin: 0px;">
        <h5>{city}</h5>{pollutant_id}  |  {state}
    </div>
    """ + '<img src=\'data:image/png;base64,{}\'>'.format(map_encode)
    popup = folium.Popup(folium.Html(datacard_html, script=True),
                         max_width=220,
                         )
    return popup