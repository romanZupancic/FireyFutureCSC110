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
import random

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

PRESENT_WEATHER_DATA = assemble_data.read_modern_weather_data()


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
    figure.update_layout(title=f'{title}', xaxis_title=x, yaxis_title=y)

    return figure

    
def combine_frequency_frames(data1: pd.DataFrame, data2: pd.DataFrame, x: str, y: str) -> go.Figure():
    figure = go.Figure(x=data1['FIRE_START_DATE'], y=data1['# of fires'], mode='lines', name='data1')
    return figure
    


# a = 4
# b = 10

# things = pd.DataFrame({'x': [x for x in range(200)], 'y': [0] * 100 + [a + x*b + random.uniform(-200, 200) for x in range(100)]})

# st.plotly_chart(generate_regressive_plot(things, 'x', 'y', 'TITLE'))

img = Image.open('./images/forest_fire.jpg')
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

logging.info('Fire frequency over time')
st.title('Fire Frequency over Time')
st.write('''Analyzing the historical frequency of fires can also be valuable. Here, we accumulate
         the number of fires along the given timescale in an effort to see if there are trends
         in the number of fires as the years go on. The following controls the resolution 
         at which the fires are accumulated.''')
freq_timescale = st.select_slider('Frequency Timescale', ['Year', 'Month', 'Day'], 'Year')
st.line_chart(data_processing.fire_frequency_over_time([FIRE_AREA_PROCESSED, 
                                                        FIRE_POINT_PROCESSED], freq_timescale))
st.write('''Interestingly, the year-by-year rates of fires appears to be decreasing: 
         at the very least, the points of maximum are decreasing, but there always seems to
         be a minimum amount of fires every year. When we switch to the monthly timescale, however,
         any trends seem to dissolve: while the number of fires may fluctuate, it is much
         harder to find a correlation between time and number of fires, on the whole. 
         Although we do find it interesting that the peaks in the last 7 years seem to be
         smaller than the peaks from 1998 - 2013.''')

logging.info('Calculating Weather and Fire severity')

st.sidebar.title('Weather and fire severity')
maximum_area = st.sidebar.slider('Maximum fire area burned', 0, 150000, 150000)

st.title('''Weather and fire severity''')
st.write('''Before we get into modelling and making predictions for future
forest fires, we need to analyse the factors that are most closely associated
with the start of a forest fire. According to the Center for Climate and
Energy Solutions, wildfire risk can depend on temperature, presence of trees
and shrubs, and soil moisture. Given the data we've collected so far, we've decided 
to organize these factors into two main groups: Temperature and Precipitation (where precipitation
is rain or snow).''')
# SOURCE
# https://www.c2es.org/content/wildfires-and-climate-change/#:~:text=Wildfire%20risk%20depends%20on%20a,climate%20variability%20and%20climate%20change.
st.write('''The following graphs display the correlation between Average temperatures and
         average precipitations in the 21 days before a fire occured.''')
st.header('''Fire Disturbance Area:''')
# Area
weather_v_area_area = data_processing.area_burned_vs_weather(FIRE_AREA_PROCESSED, maximum_area)
temperature_v_area_area = generate_regressive_plot(weather_v_area_area,
                                                   'TEMPERATURE',
                                                   'AREA BURNED',
                                                   'Fire Disturbance Area: Temperature vs. ' + 
                                                   'Area Burned')
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

precipitation_v_area_area = generate_regressive_plot(weather_v_area_area,
                                                   'PRECIPITATION',
                                                   'AREA BURNED',
                                                   'Fire Disturbance Area: Precipitation vs. ' + 
                                                   'Area Burned')
st.plotly_chart(precipitation_v_area_area)
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
st.header('''Fire Disturbance Point''')
st.write('''The classification of Fire Disturbance Point (i.e. under 40 hectares) allow for 
         fires identified by it to happen more often, and under slightly different (but not so
         different) conditions than it's > 40 hectare counterpart''')
weather_v_area_point = data_processing.area_burned_vs_weather(FIRE_POINT_PROCESSED,
                                                                     10000)

temperature_v_area_point = generate_regressive_plot(weather_v_area_point,
                                                   'TEMPERATURE',
                                                   'AREA BURNED',
                                                   'Fire Disturbance Point: Temperature vs. ' + 
                                                   'Area Burned')
st.plotly_chart(temperature_v_area_point)
st.write('''Of particular not about this data is the temperature that it
ranges from: fires were recording in temperatures as low as -16 degrees
Celcius, and in temperatures as high as 31 degrees celcius. As a reference, the disturbance area
fires only started after about 5 degrees celcius. Another quirk of this data is that the regression
line slopes downward, even though we can 'see' more area being burned the higher the temperatures.
This is likely because, even though some fires get bigger, the majority do not and weigh the entire
regression line down. The key takeaway here is that temperature does not make ALL fires bigger,
but it increases the chances for larger fires to occur.''')

precipitation_v_area_point = generate_regressive_plot(weather_v_area_point,
                                                   'PRECIPITATION',
                                                   'AREA BURNED',
                                                   'Fire Disturbance Point: Precipitation vs. ' + 
                                                   'Area Burned')
st.plotly_chart(precipitation_v_area_point)
st.write('''Again, the greater the precipitation, the less chance that a large fire will occur,
         but this does not mean that no fire will occur. Just as in our last graph, the regression
         line is infuriatingly low and shallow: and largely for the same reasons, where large fires
         are the exception, not the rule, but they occur with greater possibility none the less''')

st.plotly_chart(px.scatter_3d(weather_v_area_point, z='PRECIPITATION', y='AREA BURNED',
                              x='TEMPERATURE', height=800,
                              title='Fire Disturbance Point: Precipitation v. Area Burned ' + \
                              'v. Precipitation'))

st.write('''Here we display the point data on the three axis graph, so we can get a better 
         understanding into how our two factors interact with eachother and fire area.''')

logging.info('Explaining the models')

st.title('Forest Fire Modelling')
st.write("""The primary goal of this project is to identify how climate change has affected the frequency of forest fires in Ontario,
            and predict how it might affect forest fires in the future. In our preliminary data analysis, we concluded that we do not
            have sufficient evidence to prove that climate change has affected forest fires in Ontario over the time period from 1998 to 2020.
            We showed that over the time period from 1998 to 2020,there was no tangible change in the frequency of
            forest fires. However, this result does not proclude us from further investigating how climate change might affect forest fires
            in the future. We have assembled a dataset containing approaximately thirty-thousand 21 day sequences of max temperature and precipitation
            data, and an indicator which designates whether or not a fire occured after the day following that 21 day sequence. 
            Using this dataset, we will train 3 different models with the intention of using the best model to predict the occurence of future forest fires
            given a sequence of future weather data.""")
st.header('Model 1: Support Vector Machine')
st.write("""
The first model we have developed is a support vector machine (SVM). The data we used to train this model was a slightly modified version of the dataset explained in the above paragraph.
Instead of passing in entire 21 day weather sequences to the model, we only passed in the average temperature as well as the total precipitation over the 21 day sequence. 
Doing this drastically reduced the dimensionality of the training dataset, and resulted in a simpler decision boundary for the SVM to model. The following heatmap histograms show the obvious
correlation between average temperature/total precipitation and the occurence of a forest fire:
""")
fire_point_temp_v_weather = data_processing.graph_weather(FIRE_POINT_PROCESSED)
no_fire_temp_v_weather = data_processing.graph_weather(NO_FIRE_WEATHER_SEQUENCES)

precipitation_vs_temperature = px.density_heatmap(fire_point_temp_v_weather, x='TEMPERATURE', y='PRECIPITATION', nbinsx=30, nbinsy=30, marginal_x='histogram', marginal_y='histogram', title='TEMPERATURE vs PRECIPITATION in the 21 days before a fire')
precipitation_vs_temperature_no_fire = px.density_heatmap(no_fire_temp_v_weather, x='TEMPERATURE', y='PRECIPITATION', nbinsx=30, nbinsy=30, marginal_x='histogram', marginal_y='histogram', title='TEMPERATURE vs PRECIPITATION in the 21 days without a fire')

st.plotly_chart(precipitation_vs_temperature)
st.plotly_chart(precipitation_vs_temperature_no_fire)

st.write("""The final trained SVM achieved an accuracy of 79.7% on the the test dataset
 (i.e. given the average temperature and total precipitation over a 21 day period, 
 the model is able to predict whether or not a fire will occur at the end of that 
 21 day sequence with 79.7% accuracy""")

st.header('Model 2: Artificial Neural Network')
st.write("""
The second model we have developed is an artificial neural network (ANN). 
This model was trained on 42 by 1 arrays containing both temperature and precipitation
data for each of the 21 days in the sequence. The following is a diagram of the architecture
of this ANN:
""")
ann_graph_png = Image.open('./images/ann_graph.png')
st.image(ann_graph_png, use_column_width=False)
st.write("""The final trained ANN achieved an accuracy of 81.67% on the the test dataset
 (i.e. the model is able to predict whether or not a fire will occur at the end of a
 21 day sequence with 81.67% accuracy""")

st.header('Model 3: Double LSTM')
st.write("""
The third model we have developed contains two LSTM branches, hence why we refer to it as a double lstsm (DLSTM). 
This model model takes 2 inputs, one input containing a 21 day sequence of temperature data, and a second input
containing a 21 day sequence of precipitation data. Each input gets passed into a separate LSTM branch, and then 
the outputs of these LSTM branches get concatenated together and then passed through a dense layer. The following is
a diagram of the DLSTM architecture. """)
dlstm_graph_png = Image.open('./images/dlstm_graph.png')
st.image(dlstm_graph_png, use_column_width=True)
st.write("""The final trained DLSTM achieved an accuracy of 85% on the the test dataset
 (i.e. given two sequences, one containing 21 days temperature data and the other containing 21 days of precipitation
 data, the model is able to predict whether or not a fire will occur at the end of that 
 21 day sequence with 81.67% accuracy.""")

logging.info('Fire Predictions for 2019 vs Actual Results')
st.title('''Fire Predictions for 2019 vs Actual Results''')
st.write('''So how well does our model really preform? Here, we'll run it
against temperature and precipitation data from 2019 and see how closely our
predictions match the actual results.''')
date_vs_weather_data = PRESENT_WEATHER_DATA.set_index('Date/Time')
st.write('''Here is the temperature data for 2019.''')
st.line_chart(date_vs_weather_data['Max Temp (°C)'])
st.write('''And the total precipitation data.''')
st.line_chart(date_vs_weather_data['Total Precip (mm)'])

st.write('''From our early analysis, we can expect that more forest fires will happen in the
         summer months, because that is when temperature peaks across all our weather stations.''')
st.write('''Here are the locations of fires for 2019 in Ontario, from our
         aquired dataset:''')

fire_predictions_2019 = data_processing.predict_fires(PRESENT_WEATHER_DATA, 2019)
fire_locations_2019 = data_processing.get_2019_fire_locations()

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
            data=fire_locations_2019[['lon', 'lat']],
            opacity=0.3,
            get_position='[lon, lat]'
        )
    ]
))
st.write('''And here are the fires predicted by the dlstm model, where each fire was mapped
         to the weather station from which the weather sequence was retrieved from 
         (each fire was offset by a random longitude and latitude value):''')
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
            data=fire_predictions_2019[['lon', 'lat']],
            opacity=0.3,
            get_position='[lon, lat]'
        )
    ]
))

st.write('''Here is the actual distribution of fires:''')
fire_2019_amount  = data_processing.fire_frequency_over_time([fire_locations_2019])
st.line_chart(fire_2019_amount)
#st.write(f'{str(sum(fire_2019_amount))}')
st.write('''And here is the distribution of our fires as predicted by our model:''')
fire_predictions_2019_amount = data_processing.fire_frequency_over_time([fire_predictions_2019])
#st.write(f'{sum(fire_predictions_2019_amount)}')
st.line_chart(fire_predictions_2019_amount)

logging.info('Future Weather Predictions')
st.title('''Fire Predictions for 2050''')
st.write('''
A 2019 report released by the Government of Canada titled "Canada's Changing Climate Report", provided predictions for how
temperature and precipitation will change in Ontario between 2031 and 2050. Using this information as well as the weather data
from 2019, we have predicted daily temperature and precipitation values for each of our weather stations in the year 2050.
The following graph compares our model's predictions for fires in the year 2050 compared to the number of fires in the year 2019.
The default values of the sliders represent the values described in the report. The values of the sliders are used to generate weather
data for 2050, and this data is then fed into the dlstm model. The predicted weather data is generated by adjusting each daily temperature
reading from 2019 by the amount specified in the temperature slider, and daily precipitation measurement from 2019 by the percentage
specified by the precipitation slider.
''')

future_weather_temperature = st.slider('Temperature Offset', -10.0, 10.0, 1.6)
future_weather_precipitation = st.slider('Precipitation Multiplier', -2.0, 2.0, 1.055)
future_date_vs_weather_data = data_processing.predict_future_weather_data(PRESENT_WEATHER_DATA,
                                                                          future_weather_temperature,
                                                                          future_weather_temperature)
future_date_vs_weather_data = future_date_vs_weather_data.set_index('Date/Time')
st.write('''Future Temperatures''')
st.line_chart(future_date_vs_weather_data['Max Temp (°C)'])
st.write('''Future Precipitation''')
st.line_chart(future_date_vs_weather_data['Total Precip (mm)'])


# fire_predictions_2050 = data_processing.predict_fires(future_date_vs_weather_data, 2050)
# print(fire_predictions_2050)
# combined_frequencies = combine_frequency_frames(future_fires_data, pd.DataFrame(), 'x', 'y')
# st.plotly_chart(combined_frequencies)s
future_fires_data = data_processing.future_fires_per_month_graph_data(future_weather_temperature, future_weather_temperature)

future_fires_per_month_graph = px.line(future_fires_data, x='FIRE_START_DATE', y='# of fires', color='YEAR')
future_fires_per_month_graph.show()
logging.info('---End of file---')