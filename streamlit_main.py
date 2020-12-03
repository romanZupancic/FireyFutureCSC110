# Third-party modules
import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
from PIL import Image
import altair as alt

# Custom modules
import assemble_data
import data_processing

FIRE_POINT = assemble_data.read_fire_disturbance_point()
FIRE_AREA = assemble_data.read_fire_disturbance_area()

FIRE_POINT_CAUSE = data_processing.fire_point_cause_count(FIRE_POINT)

img = Image.open('forest_fire2.jpg')
st.image(img, use_column_width=True)
st.title('Abstract')

st.write('''Forest fires have always posed a major threat to Canadian forests.
        The National Forestry database estimates that Canada endures on average
        8000 forest fires per year, which burn through 2.1 million acres of
        landscape \cite{cite_key2}. These fires have become especially
        concerning within the past century, as the Canadian population has
        boomed and rural areas have become popular destinations for both home
        owners and cottagers. The looming threat of climate change has brought
        an even greater uncertainty to already frightening threat of forest
        fires in Canada, and it is logical to assume that hotter and drier
        weather increases the risk of forest fires occurring. Also, The
        National Forestry database reports that forest fires that begin as a
        result of a lightning strikes account for 85% of all land burned by
        fires \cite{cite_key2}. Thus, if climate change were to impact either
        the frequency of lightning storms, or increase the severity of
        droughts, forest fires could be impacted greatly. This project will
        determine if and how climate change affects forest fires in Ontario. We
        have procured a data set containing forest fire data for Ontario dating
        back to 2003. Our goal is to analyze how changes in weather patterns
        have affected the number and size of forest fires in Ontario.''')

st.title('Preliminary Data Analysis')

st.write('''The following map displays data for fires in ontario between 1998
to 2020 that are under 40 hectares in size. Areas in red are where fires are
more likely to occur. These fires are more likely to occur in the
southern-mid section of Ontario, but is still quite frequent throughout the
non-northern parts of the province.''')
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
        latitude=50,
        longitude=-86,
        zoom=4,
        pitch=30
    ),
    layers=[
        pdk.Layer(
            "HeatmapLayer",
            data=FIRE_POINT,
            opacity=0.3,
            get_position='[longitude, latitude]'
        )
        #  pdk.Layer(
        #     'HexagonLayer',
        #     data=map_data.fires,
        #     radius=8000,
        #     get_position='[longitude, latitude]',
        #     elevation_scale=30,
        #     elevation_range=[0, 1000],
        #     extruded=True
        #  ),
        # pdk.Layer(
        #     "ContourLayer",
        #     data=map_data.fires,
        #     get_position='[longitude, latitude]',
        #     cell_size=8000

        # )
    ]
))

st.write('''The following map displays the location of forest fires in
Ontario from 1998 to 2020 that were greater than 40 hectares in size. Areas
highlighted in red indicate areas where these fires most frequently occur. 
The importance of separating fires by size is clearly illustrated by the 
discrepancy between the two maps. In the map bellow, we can see that larger forest
fires tend to occur in the north western region of Ontario, while the map above
shows that smaller fires most often occur further south.''')
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
        latitude=50,
        longitude=-86,
        zoom=4,
        pitch=30
    ),
    layers=[
        pdk.Layer(
            "HeatmapLayer",
            data=FIRE_AREA,
            opacity=0.3,
            get_position='[LONGITUDE, LATITUDE]'
        )
        #  pdk.Layer(
        #     'HexagonLayer',
        #     data=map_data.fires,
        #     radius=8000,
        #     get_position='[longitude, latitude]',
        #     elevation_scale=30,
        #     elevation_range=[0, 1000],
        #     extruded=True
        #  ),
        # pdk.Layer(
        #     "ContourLayer",
        #     data=map_data.fires,
        #     get_position='[longitude, latitude]',
        #     cell_size=8000

        # )
    ]
))

st.write('''The data sets we aquired have a few built in metrics we can
compare fire intensity, size, and frequency to. For example, over our
datasets, the common causes are:''')

st.bar_chart(data=FIRE_POINT_CAUSE)

cause_alt = alt.Chart(data_processing.fire_area_cause_count(FIRE_AREA)).mark_bar().encode(
    x='Cause of Fire', y='Occurances in dataset')

st.altair_chart(cause_alt)