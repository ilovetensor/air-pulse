import streamlit as st
import base64
from templates.breathe import html_code

st.set_page_config(layout="wide", page_icon='🎈', page_title='AirPulse',)


@st.cache_data
def get_image_as_base64(file):
    with open(file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_image_as_base64('templates/1761746.jpg')
st.text('');
st.text('');
st.text('');
st.text('');

st.markdown("""<h1 id="title-h"><span data-id="s1" class="spans">AIR</span><span data-id="s2">- PULSE 🎈</span></h1>""",
            unsafe_allow_html=True)

with open("templates/title.css") as file:
    st.markdown(f'<style>{file.read()}</style>', unsafe_allow_html=True)
with open("style.css") as file:
    st.markdown(f'<style>{file.read()}</style>', unsafe_allow_html=True)

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


st.markdown(
    """
    <style>
    .transparent-iframe iframe {
        background-color: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True)

st.components.v1.html(html_code, height=150)

st.text('');
st.text('');
st.text('');
st.text('');
st.link_button('Explore now !', '/home')
st.text('');
st.text('');
st.text('');
st.text('');
st.text('');
st.text('');
st.text('');
st.text('');

st.markdown('''
    This app is a deep dive analysis of the air quality parameters over the years in India. Also it provides a detailed view of 
    current live air quality stats and allow user to compare them in different ways. The app tried to show 
    every dimension possible to understand the air quality in India.
            
    There are many hidden facts and relations that were found during the analysis of the data and which are vital to
    have a better understanding of the air quality in India. The app is designed to show all those facts and relations
    in a very intuitive way.

''')

st.markdown('# Features of the app')

col1, col2 = st.columns(2)
with col1:
    st.markdown("## ✏️ Live Air Quality Stats")
    st.markdown('''The app provides live air quality stats of all the states in India. The stats are updated every hour.
            It also provide the control to the user to apply filters on the data and compare the stats in different ways''')

with col2:
    st.markdown("## 🧪 Historical Analysis")
    st.markdown('''The app provides a historical analysis of the air quality parameters over the years in India.The 
            analysis is done at state level and city level. The analysis is done in a very intuitive way to show the trends and 
            relations between the parameters.''')

col3, col4 = st.columns(2)
with col3:
    st.markdown("## 💹 Forecasting")
    st.markdown('''The app provides a forecasting of the air quality parameters months wise for the next 2 years. The forecasting
            is done at state level. Different models were tried and tested to come up with the best model for forecasting.''')
with col4:
    st.markdown("## 📃 External Factors")
    st.markdown('''The app provides a detailed analysis and comparision of factors like **health**, **energy**, **automobiles**, 
            were done to understand their impact on the air quality''')

st.text('');
st.text('');
st.markdown('# Data Sources')

st.markdown('''
1. **Air Quality Data** - The air quality data is taken from [data.gov.in](https://data.gov.in/). The data is collected using their
            live API and is updated every hour.
2. **Historical Data** - The historical data is taken from [data.gov.in](https://data.gov.in/). The data is collected in various
            files and then compiled into a single dataset for analysis.
3. **Health Data** - The health data is taken from [data.gov.in](https://data.gov.in/). Which was then cleaned and processed to
            make it usable for analysis.
4. **Energy and Demographics Data** - The energy and demographics data are taken from snowflake marketplace. The data 
            is accessed using the snowflake connector and then modified and processed to make it usable for analysis.
            
''')
