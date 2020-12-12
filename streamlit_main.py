"""
The main file of our project. This calls all the code that is required for our app to function,
as long as the datasets have already been built.

Most of this file is streamlit function calls, because that is the interface that
our app uses to display the data. Other functions (data retrival and processing) functions
are also called in this file, and, most of the time, passed on to streamlit.
"""
# Third-party modules
import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

import logging

import time

# Custom modules
import assemble_data
import data_processing

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s: %(message)s')

logging.info('---Start of file---')

FIRE_POINT = assemble_data.read_fire_disturbance_point()
FIRE_AREA = assemble_data.read_fire_disturbance_area()

FIRE_POINT_PROCESSED = assemble_data.read_fire_disturbance_point_processed()
FIRE_AREA_PROCESSED = assemble_data.read_fire_disturbance_area_processed()

NO_FIRE_WEATHER_SEQUENCES = assemble_data.read_no_fire_weather_sequences()

WEATHER_DATA = assemble_data.read_all_station_weather_data()

data_processing.predict_future_weather_data(WEATHER_DATA)

def generate_regressive_plot(data: pd.DataFrame,
                             x: str, y: str, title: str) -> go.Figure():
    regression = data_processing.linear_regression(data, x, y)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data[x], y=data[y], mode='markers', name='Data'))
    linear_x = [data[x].min(), data[x].max()]
    linear_y = [data_processing.evaluate_linear_equation(regression[0],
                                                         regression[1], linear_x[0]),
                data_processing.evaluate_linear_equation(regression[0],
                                                         regression[1], linear_x[1])]
    figure.add_trace(go.Scatter(x=linear_x, y=linear_y, mode='lines', name='Regression Line'))
    figure.update_layout(title=title, xaxis_title=x, yaxis_title=y)

    return figure


logging.info('Starting UI')

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

logging.info('Presenting maps')

st.header('Forest Fire Locations Since 1998')
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
            get_position='[LONGITUDE, LATITUDE]'
        )
    ]
))

# st.pydeck_chart(pdk.Deck(
#     map_style='mapbox://styles/mapbox/light-v9',
#     initial_view_state=pdk.ViewState(
#         latitude=50,
#         longitude=-86,
#         zoom=4,
#         pitch=30
#     ),
#     layers=[
#         pdk.Layer(
#             "HeatmapLayer",
#             data=FIRE_POINT2,
#             opacity=0.3,
#             get_position='[LONGITUDE, LATITUDE]'
#         )
#     ]
# ))

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
    ]
))

# st.map(data_processing.read_weather_data())

logging.info('Calculating Fire Causes')

st.header('Fire Causes Since 1998')
st.write('''The data sets we aquired have a few built in metrics we can
compare fire intensity, size, and frequency to. For example, over
the Fire Disturbance Point dataset, we have:''')

st.bar_chart(data=data_processing.fire_cause_count(FIRE_POINT))

st.write('''And over the Fire Disturbance Area dataset, we have:''')

st.bar_chart(data=data_processing.fire_cause_count(FIRE_AREA))

st.write('''Generally, we see fires started by lightining a considerable
amount more than any other cause. This might be an interesting factor to
watch in the event that climate change effects the frequency and intensity of
lightning strikes: we should see an increase of lightning related fires over
the years.''')

st.write('''Additionaly, we see that human-caused fires
(recreational, logging, incendiary, etc.) are relatively low in number. Even
so, the climate still effects the frequency of these types of fires:
prolonged periods of dry and hot weather can increase the chance of workplace
and recreational accidents to grow into reported fires. It is evident,
however, that these fires are not that impactful in the grand scale of
Ontario's forests: human-caused fires are far more likely to be smaller (and
end up in the Fire Point database) than other fire causes.''')

logging.info('Calculating Causes over time')

st.sidebar.title('Causes over time')
st.sidebar.header('Fire causes')
fire_cause_over_time_values = {key: True for key in data_processing.CAUSE_REFERENCES.keys()}

for cause in fire_cause_over_time_values:
    fire_cause_over_time_values[cause] = \
                                st.sidebar.checkbox(data_processing.CAUSE_REFERENCES[cause], True)
st.sidebar.header('Timescale')
cause_timescale = st.sidebar.select_slider('Timescale', ['Year', 'Month', 'Day'], 'Month')

st.header('Causes over time')
st.write('''The following graph displays the number of Fire Disturbance Point
causes over time. What data is displayed, and over what timescales, can be
configured in the side bar.''')

enabled_cause_values = [key for key in fire_cause_over_time_values
                        if fire_cause_over_time_values[key]]
st.line_chart(data_processing.fire_cause_over_time(FIRE_POINT, enabled_cause_values,
                                                   cause_timescale))
st.write('And over the Fire Disturbance Area dataset:')
st.line_chart(data_processing.fire_cause_over_time(FIRE_AREA, enabled_cause_values,
                                                   cause_timescale))

st.write('''It is important to note the difference in scale between the two
datasets (where the point dataset is about 10 - 15 times bigger than the area
dataset). It is also helpful to examine the data together. For example, the
peaks and troughs between the two sets are nearly identical for lightning
causes on the yearly scale, but the similarities begin to break down as we
consider other causes (E.g miscellaneous on the yearly scale has an entirely different
shape on each of the graphs). Some of the variation in data, and perhaps why some of the
other causes diverge in similarities, could be because of the lack of data in the Fire
Disturbance Area dataset - perhaps not because the set is missing fires, but that the fires
do not occur as regularly, as they are bigger. With less data, the causes attributed to
the Fire Disturbance Area dataset have less room to "smooth" out: a change of 5-to-4 seems much
greater than a change of 123-to-128.''')

logging.info('Calculating Weather and Fire severity')

st.sidebar.title('Weather and fire severity')
maximum_area = st.sidebar.slider('Maximum fire area burned', 0, 150000, 150000)

st.header('''Weather and fire severity''')
st.write('''The following graphs display the correlation between Average temperatures and
         average precipitations in the 21 days before a fire occured.''')
# Area
weather_v_area_area = data_processing.area_burned_vs_weather(FIRE_AREA_PROCESSED, maximum_area)
# st.plotly_chart(px.scatter(weather_v_area_area, x='TEMPERATURE', y='AREA BURNED',
#                            title='Fire Disturbance Area: Temperature v. Area Burned'))
# temperature_v_area_area = go.Figure()
temperature_v_area_area = generate_regressive_plot(weather_v_area_area,
                                                   'TEMPERATURE',
                                                   'AREA BURNED',
                                                   'Fire Disturbance Area: Temperature v. Area Burned')
st.plotly_chart(temperature_v_area_area)
st.write('''One of the most important takeaways from this visualization is
         that fires have the potential to grow large at higher temperatures
         than at lower temperatures. This is evident with just the base graph
         (displaying a maximum area of 150000 hectares), where the highest
         points (indicating the most area burned) were at the farther end of
         the graph (indicating higher temperatures). The trend continues as
         we "zoom in" to the graph, where higher points are always more
         frequent with larger temperatures. The density of the points is also
         an indicator of the frequency of fires. While at temperatures of 0
         degrees celcius to (at the very least) 10 degrees celcius see few
         fires, temperatures from 15 degrees celcius onward sees a far
         greater concentration.''')

st.plotly_chart(px.scatter(weather_v_area_area, x='PRECIPITATION', y='AREA BURNED',
                           title='Fire Disturbance Area: Precipitation v. Area Burned'))
st.write('''This graph shows the negative correlation between precipitation
and area burned: where higher precipitation results in weaker and less
frequent fires. ''')

st.plotly_chart(px.scatter_3d(weather_v_area_area, z='PRECIPITATION', y='AREA BURNED',
                              x='TEMPERATURE', height=800,
                              title='Fire Disturbance Area: Precipitation v. Area Burned ' + \
                              'v. Precipitation'))
st.write('''Finally, we have a 3D graph plotting temperature, precipitation, and fire area burned
         on 3 separate axis. It (unsurprisingly) seems that fires tend to spread the farthest
         under low precipitations and high temperatures. However, the opposite is not true:
         just because a fire happens under dry and hot conditions does not mean it will grow
         very large: hence, the large blob of datapoints at the lower ends of the temperature and
         precipitation axis.''')
st.write('''It also seems that precipitation has a much larger effect on fire frequency and
         strength than weather does: there is a far narrower range of large fires on the precipitation
         axis than on the temperature axis.''')

# Point
weather_v_area_point = data_processing.area_burned_vs_weather(FIRE_POINT_PROCESSED,
                                                                     10000)
st.plotly_chart(px.scatter(weather_v_area_point, x='TEMPERATURE', y='AREA BURNED',
                           title='Fire Disturbance Point: Temperature v. Area Burned'))

st.plotly_chart(px.scatter(weather_v_area_point, x='PRECIPITATION', y='AREA BURNED',
                           title='Fire Disturbance Point: Precipitation v. Area Burned'))

st.plotly_chart(px.scatter_3d(weather_v_area_point, z='PRECIPITATION', y='AREA BURNED',
                              x='TEMPERATURE', height=800,
                              title='Fire Disturbance Point: Precipitation v. Area Burned ' + \
                              'v. Precipitation'))

logging.info('Calculating precipitation and temperature correlations')

st.title('Data Modelling Part 1:')
st.write("""The primary goal of this project is to identify how climate change has affected the frequency of forest fires in Ontario,
            and predict how it might affect forest fires in the future. In our preliminary data analysis, we concluded that we do not
            have sufficient evidence to prove that climate change has affected forest fires in Ontario. We showed that the fires dataset only
            includes the fires which occured from 1998 to 2020, and over this time period there was no tangible change in the frequency of
            forest fires. However, this result does not proclude us from further investigating how climate change might affect forest fires
            in the future. We have assembled a dataset containing approaximately thirty-thousand 21 day sequences of max temperature and precipitation
            data, and an indicator which designates whether or not a fire occured after the day following that 21 day sequence.""")
st.header('Correlations between temperature and precipitation')
fire_point_temp_v_weather = data_processing.graph_weather(FIRE_POINT_PROCESSED)
no_fire_temp_v_weather = data_processing.graph_weather(NO_FIRE_WEATHER_SEQUENCES)

precipitation_vs_temperature = px.density_heatmap(fire_point_temp_v_weather, x='TEMPERATURE', y='PRECIPITATION', nbinsx=30, nbinsy=30, marginal_x='histogram', marginal_y='histogram')
precipitation_vs_temperature_no_fire = px.density_heatmap(no_fire_temp_v_weather, x='TEMPERATURE', y='PRECIPITATION', nbinsx=30, nbinsy=30, marginal_x='histogram', marginal_y='histogram')
# precipitation_vs_temperature = px.scatter(fire_point_temp_v_weather, x='TEMPERATURE', y='PRECIPITATION')
# precipitation_vs_temperature_no_fire = px.scatter(no_fire_temp_v_weather, x='TEMPERATURE', y='PRECIPITATION')
st.plotly_chart(precipitation_vs_temperature)
st.write('No Fire')
st.plotly_chart(precipitation_vs_temperature_no_fire)

logging.info('---End of file---')