import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import map_data
import pydeck as pdk

img = Image.open('forest_fire.jpg')
st.image(img, use_column_width=True)
st.title('Ontario\'s Fiery Future')

st.write("""Forest fires have always posed a major threat to Canadian forests.
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
        have affected the number and size of forest fires in Ontario.""")

st.map(map_data.fires)
st.pydeck_chart(pdk.Deck(
     map_style='mapbox://styles/mapbox/dark-v9',
     initial_view_state=pdk.ViewState(
         latitude=46,
         longitude=-79,
         zoom=5,
         pitch=30,
     ),
     layers=[
        #  pdk.Layer(
        #     'HexagonLayer',
        #     data=map_data.fires,
        #     radius=8000,
        #     get_position='[longitude, latitude]',
        #     elevation_scale=30,
        #     elevation_range=[0, 1000],
        #     extruded=True
        #  ),
        pdk.Layer(
           "HeatmapLayer",
           data=map_data.fires,
           opacity=0.8,
           get_position='[longitude, latitude]',
        )
        # pdk.Layer(
        #     "ContourLayer",
        #     data=map_data.fires,
        #     get_position='[longitude, latitude]',
        #     cell_size=8000
            
        # )
     ],
 ))
