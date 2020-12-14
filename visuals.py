"""
Handles visualization of all data, and explanations and analysis of that data.

As little data processing as possible is done in this file, in order to separate processing
and UI. However, in some cases UI and processing are coupled - sliders and checkboxes - making it
much harder to organize the components. In these cases, we've just combined them.

Copyright and Usage Information
===============================
This file is Copyright (c) 2020 Daniel Hocevar and Roman Zupancic. 

This files contents may not be modified or redistributed without written
permission from Daniel Hocevar and Roman Zupancic
"""

from typing import Dict
import streamlit as st
import pydeck as pdk
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

import data_processing
import assemble_data

def section_introduction() -> None:
    """
    The Project Introduction.
    """
    img = Image.open('./images/forest_fire.jpg')
    st.image(img, use_column_width=True)
    st.title('Introduction')
    st.write('''Forest fires have always posed a major threat to Canadian forests.
            These fires have become especially concerning within the past century, as the Canadian population has
            boomed and rural areas have become popular destinations for both home
            owners and cottagers. The looming threat of climate change has brought
            an even greater uncertainty to already frightening threat of forest
            fires in Canada, and it is logical to assume that hotter and drier
            weather increases the risk of forest fires occurring. If climate 
            change were to increase the severity of droughts - or increase any of the factors that 
            cause fires in the first place - the frequency and severity of forest 
            fires could be impacted greatly. This project will
            determine if and how climate change affects forest fires in Ontario. We
            have procured a data set containing forest fire data for Ontario dating
            back to 1998. First, we will manually analyze the data through graphs and 
            simple projections. Then, we'll showcase some machine learning models
            we've developed to help predict the possibility of forest fires
            given some weather data. We will end with an interactive demo.''')

    st.write('''Please use the sidebar to navigate throughout this website. For best
             results, visit each section in order.''')


def subsection_fire_locations(fire_point: pd.DataFrame, fire_area: pd.DataFrame) -> None:
    """
    This section talks about fire location data. It plots fire location data for 
    area and point data on maps.
    """
    st.header('Forest Fire Locations Since 1998')
    st.write('''The following map displays data for fires in ontario between 1998
    to 2020 that are under 40 hectares in size. Areas in red are where fires are
    more likely to occur. These fires are more likely to occur in the
    southern-mid section of Ontario, but are still quite frequent throughout the
    other non-northern parts of the province.''')

    st.pydeck_chart(heatmap_ontario(fire_point, '[LONGITUDE, LATITUDE]'))

    st.write('''The following map displays the location of forest fires in
    Ontario from 1998 to 2020 that were greater than 40 hectares in size. Areas
    highlighted in red indicate areas where these fires most frequently occur.
    The importance of separating fires by size is clearly illustrated by the
    discrepancy between the two maps. In the map below, we can see that larger forest
    fires tend to occur in the north western region of Ontario, while the map above
    shows that smaller fires most often occur further south.''')

    st.pydeck_chart(heatmap_ontario(fire_area, '[LONGITUDE, LATITUDE]'))


def subsection_fire_cause_counts(cause_point: pd.DataFrame, cause_area: pd.DataFrame) -> None:
    """
    Handles fire causes over time. Charts cause_point and cause_area on bar charts.
    """
    st.header('Fire Causes Since 1998')
    st.write('''The data sets we aquired have a few built in metrics we can
    compare fire intensity, size, and frequency to. For example, over
    the Fire Disturbance Point dataset, we have:''')

    st.bar_chart(data=cause_point)

    st.write('''And over the Fire Disturbance Area dataset, we have:''')

    st.bar_chart(data=cause_area)

    st.write('''Generally, we see fires started by lightning a considerable
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

    
def subsection_causes_over_time(cause_mapping: Dict[str, str],
                                fire_point: pd.DataFrame,
                                fire_area: pd.DataFrame) -> None:
    """
    Displays UI for firce causes over time.
    Contains interactive sliders in the body of the section.
    Data processing is done in this function.
    """

    st.header('Causes over time')
    st.write('''The following graph displays the number of Fire Disturbance Point
    causes over time. What data is displayed, and over what timescales, can be
    configured below.''')

    fire_cause_over_time_values = {key: True for key in cause_mapping.keys()}

    for cause in fire_cause_over_time_values:
        fire_cause_over_time_values[cause] = st.checkbox(cause_mapping[cause], True)

    cause_timescale = st.select_slider('Timescale', ['Year', 'Month', 'Day'], 'Month')
    enabled_cause_values = [key for key in fire_cause_over_time_values
                            if fire_cause_over_time_values[key]]

    st.write('Fire Disturbance Point:')
    st.line_chart(data_processing.fire_cause_over_time(fire_point, 
                                                       enabled_cause_values, cause_timescale))
    st.write('And over the Fire Disturbance Area dataset:')
    st.line_chart(data_processing.fire_cause_over_time(fire_area, 
                                                       enabled_cause_values, cause_timescale))

    st.write('''It is important to note the difference in scale between the
    two datasets (where the point dataset is about 10 - 15 times bigger than
    the area dataset). It is also helpful to examine the data together. For
    example, the peaks and troughs between the two sets are nearly identical
    when comparing lightning causes on the yearly scale, but the similarities begin to
    break down as we consider other causes (E.g miscellaneous on the yearly
    scale has an entirely different shape on each of the graphs). Some of the
    variation in data, and perhaps why some of the other causes diverge in
    similarities, could be because of the lack of data in the Fire
    Disturbance Area dataset - perhaps not because the set is missing fires,
    but that the fires do not occur as regularly, as they are bigger. With
    less data, the causes attributed to the Fire Disturbance Area dataset
    have less room to "smooth" out: a change of 5-to-4 seems much greater
    than a change of 123-to-128.''')


def section_frequency_over_time(fire_area, fire_point) -> None:
    """
    Displays UI for fire frequencies over time. 
    Combines fire_area and fire_point dataframe and plots them on a line graph.
    """
    st.title('Fire Frequency over Time')
    st.write('''Analyzing the historical frequency of fires can also be
    valuable. Here, we accumulate the number of fires along the given
    timescale in an effort to see if there are trends in the number of fires
    as the years go on. The following controls the resolution at which the
    fires are accumulated.''')
    freq_timescale = st.select_slider('Frequency Timescale', ['Year', 'Month', 'Day'], 'Year')
    st.line_chart(data_processing.fire_frequency_over_time([fire_area, 
                                                            fire_point], freq_timescale))
    st.write('''Interestingly, the year-by-year rates of fires appears to be
    decreasing: at the very least, the points of maximum are decreasing, but
    there always seems to be a minimum amount of fires every year. When we
    switch to the monthly timescale, however, any trends seem to dissolve:
    while the number of fires may fluctuate, it is much harder to find a
    correlation between time and number of fires, on the whole. Although we
    do find it interesting that the peaks in the last 7 years seem to be
    smaller than the peaks from 1998 - 2013.''')

    
def section_weather_fire_severity(fire_point, fire_area) -> None:
    """
    Displays UI for fire severity vs temperature and precipitation for the data in 
    fire_point and fire_area.
    """
    st.title('''Weather and fire severity''') 
    st.write('''Before we get into
    modelling and making predictions for future forest fires, we need to
    analyse the factors that are most closely associated with the start of a
    forest fire. According to the Center for Climate and Energy Solutions,
    wildfire risk can depend on temperature, presence of trees and shrubs,
    and soil moisture [\[citation\]](https://www.c2es.org/content/wildfires-and-climate-change/#:~:text=Wildfire%20risk%20depends%20on%20a,climate%20variability%20and%20climate%20change).
    Given the data we've collected so far, we've decided to organize these
    factors into two main groups: Temperature and Precipitation (where
    precipitation is rain or snow). ''')
    st.write('''The following graphs display the correlation between Average temperatures and
            average precipitations in the 21 days before a fire occured.''')
    st.header('''Fire Disturbance Area:''')
    # Area
    weather_v_area_area = fire_area
    temperature_v_area_area = generate_regressive_plot(weather_v_area_area,
                                                       'TEMPERATURE',
                                                       'AREA BURNED',
                                                       'Fire Disturbance Area: Temperature vs. ' + 
                                                       'Area Burned')
    st.plotly_chart(temperature_v_area_area)
    st.write('''One of the most important takeaways from this visualization
    is that fires have the potential to grow large at higher temperatures
    than at lower temperatures. The highest points (indicating the most area
    burned) are at the farther end of the graph (indicating higher
    temperatures). The trend continues as we "zoom in" to the graph, where
    higher points are always more frequent with larger temperatures. The
    density of the points is also an indicator of the frequency of fires.
    While at temperatures of 0 degrees celcius to (at the very least) 10
    degrees celcius see few fires, temperatures from 15 degrees celcius
    onward sees a far greater concentration.''')

    precipitation_v_area_area = generate_regressive_plot(weather_v_area_area,
                                                         'PRECIPITATION',
                                                         'AREA BURNED',
                                                         'Fire Disturbance Area: Precipitation ' + 
                                                         'vs. Area Burned')
    st.plotly_chart(precipitation_v_area_area)
    st.write('''This graph shows the negative correlation between precipitation
    and area burned: where higher precipitation results in weaker and less
    frequent fires. ''')

    st.plotly_chart(px.scatter_3d(weather_v_area_area, z='PRECIPITATION', y='AREA BURNED',
                                x='TEMPERATURE', height=800,
                                title='Fire Disturbance Area: Precipitation v. Area Burned ' + \
                                'v. Precipitation'))
    st.write('''Finally, we have a 3D graph plotting temperature,
    precipitation, and fire area burned on 3 separate axis. It
    (unsurprisingly) seems that fires tend to spread the farthest under low
    precipitations and high temperatures. However, the opposite is not true:
    just because a fire happens under dry and hot conditions does not mean it
    will grow very large: hence, the large blob of datapoints at the lower
    ends of the temperature and precipitation axis.''')
    st.write('''It also seems that precipitation has a much larger effect on
    fire frequency and strength than weather does: there is a far narrower
    range of large fires on the precipitation axis than on the temperature
    axis.''')

    # Point
    st.header('''Fire Disturbance Point''')
    st.write('''The classification of Fire Disturbance Point (i.e. under 40 hectares) allow for 
            fires identified by it to happen more often, and under slightly different (but not so
            different) conditions than it's > 40 hectare counterpart.''')
    weather_v_area_point = fire_point

    temperature_v_area_point = generate_regressive_plot(weather_v_area_point,
                                                    'TEMPERATURE',
                                                    'AREA BURNED',
                                                    'Fire Disturbance Point: Temperature vs. ' + 
                                                    'Area Burned')
    st.plotly_chart(temperature_v_area_point)
    st.write('''Of particular note about this data is the temperature that it
    ranges from: fires were recording in temperatures as low as -16 degrees
    Celcius, and in temperatures as high as 31 degrees celcius. As a
    reference, the disturbance area fires only started after about 5 degrees
    celcius. Another quirk of this data is that the regression line slopes
    downward, even though we can 'see' more area being burned the higher the
    temperatures. This is likely because, even though some fires get bigger,
    the majority do not and weigh the entire regression line down. The key
    takeaway here is that temperature does not make ALL fires bigger, but it
    increases the chances for larger fires to occur and for more 
    fires to occur.''')

    precipitation_v_area_point = generate_regressive_plot(weather_v_area_point,
                                                    'PRECIPITATION',
                                                    'AREA BURNED',
                                                    'Fire Disturbance Point: Precipitation vs. ' + 
                                                    'Area Burned')
    st.plotly_chart(precipitation_v_area_point)
    st.write('''Again, the greater the precipitation, the less chance that a
    large fire will occur, but this does not mean that no fire will occur.
    Just as in our last graph, the regression line is infuriatingly low and
    shallow: and largely for the same reasons, where large fires are the
    exception, not the rule, but they occur with greater possibility none the
    less.''')

    st.plotly_chart(px.scatter_3d(weather_v_area_point, z='PRECIPITATION', y='AREA BURNED',
                                x='TEMPERATURE', height=800,
                                title='Fire Disturbance Point: Precipitation v. Area Burned ' + \
                                'v. Precipitation'))

    st.write('''Here we display the point data on the three axis graph, so we can get a better 
            understanding into how our two factors interact with eachother and fire area.''')

            
def section_model_explanation(fire_point_temp_v_weather, no_fire_temp_v_weather) -> None:
    """
    The UI for the model expanation. Uses both graphs and images to explain how various models
    work.
    """
    st.title('Forest Fire Modelling')
    st.write("""The primary goal of this project is to identify how climate
    change has affected the frequency of forest fires in Ontario, and predict
    how it might affect forest fires in the future. In our preliminary data
    analysis, we concluded that we do not have sufficient evidence to prove
    that climate change has affected forest fires in Ontario over the time
    period from 1998 to 2020. We showed that over the time period from 1998
    to 2020, there was no tangible change in the frequency of forest fires.
    However, this result does not proclude us from further investigating how
    climate change might affect forest fires in the future. We have assembled
    a dataset containing approaximately thirty-thousand 21 day sequences of
    max temperature and precipitation data, and an indicator which designates
    whether or not a fire occured after the day following that 21 day
    sequence. Using this dataset, we will train 3 different models with the
    intention of using the best model to predict the occurence of future
    forest fires given a sequence of future weather data.""")
    st.header('Model 1: Support Vector Machine')
    st.write("""
    The first model we have developed is a support vector machine (SVM). The
    data we used to train this model was a slightly modified version of the
    dataset explained in the above paragraph. Instead of passing in entire 21
    day weather sequences to the model, we only passed in the average
    temperature as well as the total precipitation over the 21 day sequence.
    Doing this drastically reduced the dimensionality of the training
    dataset, and resulted in a simpler decision boundary for the SVM to
    model. The following heatmap histograms show the obvious correlation
    between average temperature/total precipitation and the occurence of a
    forest fire:
    """)
    
    precipitation_vs_temperature = \
        px.density_heatmap(fire_point_temp_v_weather, 
                           x='TEMPERATURE', y='PRECIPITATION', nbinsx=30, nbinsy=30, 
                           marginal_x='histogram', marginal_y='histogram', 
                           title='TEMPERATURE vs PRECIPITATION in the 21 days before a fire')
    precipitation_vs_temperature_no_fire = \
        px.density_heatmap(no_fire_temp_v_weather, x='TEMPERATURE', y='PRECIPITATION', 
                           nbinsx=30, nbinsy=30, marginal_x='histogram', marginal_y='histogram',
                           title='TEMPERATURE vs PRECIPITATION in the 21 days without a fire')

    st.plotly_chart(precipitation_vs_temperature)
    st.plotly_chart(precipitation_vs_temperature_no_fire)

    st.write("""The final trained SVM achieved an accuracy of 80.44% on the the test dataset
    (i.e. given the average temperature and total precipitation over a 21 day period, 
    the model is able to predict whether or not a fire will occur at the end of that 
    21 day sequence with 79.7% accuracy)""")

    st.header('Model 2: Artificial Neural Network')
    st.write("""
    The second model we have developed is an artificial neural network (ANN). 
    This model was trained on 42 by 1 arrays containing both temperature and precipitation
    data for each of the 21 days in the sequence. The following is a diagram of the architecture
    of this ANN:
    """)
    ann_graph_png = Image.open('./images/ann_graph.png')
    st.image(ann_graph_png, use_column_width=False)
    st.write("""The final trained ANN achieved an accuracy of 82.66% on the the test dataset
    (i.e. the model is able to predict whether or not a fire will occur at the end of a
    21 day sequence with 82.66% accuracy""")

    st.header('Model 3: Double LSTM')
    st.write("""
    The third model we have developed contains two LSTM branches, hence why
    we refer to it as a double lstsm (DLSTM). This model model takes 2
    inputs, one input containing a 21 day sequence of temperature data, and a
    second input containing a 21 day sequence of precipitation data. Each
    input gets passed into a separate LSTM branch, and then the outputs of
    these LSTM branches get concatenated together and then passed through a
    dense layer. The following is a diagram of the DLSTM architecture. """)
    dlstm_graph_png = Image.open('./images/dlstm_graph.png')
    st.image(dlstm_graph_png, use_column_width=True)
    st.write("""The final trained DLSTM achieved an accuracy of 85% on the
    the test dataset (i.e. given two sequences, one containing 21 days
    temperature data and the other containing 21 days of precipitation data,
    the model is able to predict whether or not a fire will occur at the end
    of that 21 day sequence with 81.67% accuracy.""")

    
def section_predictions_vs_actual(present_weather_data) -> None:
    """
    The UI for analysis of the efficacy of our dlstm model.
    """
    st.title('''Fire Predictions for 2019 vs Actual Results''')
    st.write('''So how well does our model really preform? Here, we'll run it
    against temperature and precipitation data from 2019 and see how closely our
    predictions match the actual results.''')
    date_vs_weather_data = present_weather_data.set_index('Date/Time')
    st.write('''Here is the temperature data for 2019.''')
    st.line_chart(date_vs_weather_data['Max Temp (°C)'])
    st.write('''And the total precipitation data.''')
    st.line_chart(date_vs_weather_data['Total Precip (mm)'])

    st.write('''From our early analysis, we can expect that more forest fires
    will happen in the summer months, because that is when temperature peaks
    across all our weather stations.''')
    st.write('''Here are the locations of fires for 2019 in Ontario, from our
    aquired dataset:''')

    fire_predictions_2019 = data_processing.predict_fires(present_weather_data, 2019)
    fire_locations_2019 = data_processing.get_2019_fire_locations()

    st.pydeck_chart(heatmap_ontario(fire_locations_2019[['lon', 'lat']], '[lon, lat]'))

    st.write('''And here are the fires predicted by the dlstm model, where each fire was mapped
            to the weather station from which the weather sequence was retrieved from 
            (each fire was offset by a random longitude and latitude value):''')
    st.pydeck_chart(heatmap_ontario(fire_predictions_2019[['lon', 'lat']], '[lon, lat]'))

    st.write('''While this is a largely superficial glance at the data our model produces,
             the maps do give a little insight into the correctness of the model. Since 
             stronger areas are identified in about the same locations on both maps, 
             we can make the conclusion that our model has at least been able to identify relative
             chances for fires to break out (i.e some areas will recieve more fires than others).''')

    st.write('''The following two graphs is an assessment of total fires over time.''')
    st.write('''Here is the actual distribution of fires:''')
    fire_2019_amount  = data_processing.fire_frequency_over_time([fire_locations_2019])
    st.line_chart(fire_2019_amount)
    #st.write(f'{str(sum(fire_2019_amount))}')
    st.write('''And here is the distribution of our fires as predicted by our model:''')
    fire_predictions_2019_amount = data_processing.fire_frequency_over_time([fire_predictions_2019])
    #st.write(f'{sum(fire_predictions_2019_amount)}')
    st.line_chart(fire_predictions_2019_amount)

    st.write('''While certainly not perfect, our model has developed an intuition for the data it
             is processing: it mirrors the expected output not exactly, but is fairly close in
             shape and size.''')


def section_2050_predictions(present_weather_data) -> None:
    """The UI for making fire predictions about arbitrary weather data in 2020.
    This function uses data_processing.
    """
    st.title('''Fire Predictions for 2050''')
    st.write('''
    A 2019 report released by the Government of Canada titled "Canada's
    Changing Climate Report", provided predictions for how temperature and
    precipitation will change in Ontario between 2031 and 2050. Using this
    information as well as the weather data from 2019, we have predicted
    daily temperature and precipitation values for each of our weather
    stations in the year 2050. The following graph compares our model's
    predictions for fires in the year 2050 compared to the number of fires in
    the year 2019. The default values of the sliders represent the values
    described in the report. The values of the sliders are used to generate
    weather data for 2050, and this data is then fed into the dlstm model.
    The predicted weather data is generated by adjusting each daily
    temperature reading from 2019 by the amount specified in the temperature
    slider, and daily precipitation measurement from 2019 by the percentage
    specified by the precipitation slider.
    ''')

    st.write('''The following sliders allow you to configure your own offsets to the 2019 data.
             The default numbers (1.6 for Temperature Offset, and 5.5 for Precipitation Percentage
             Change) are the offsets calculated by "Canada's Changing Climate Report". ''')
    future_weather_temperature = st.slider('Temperature Offset (degrees Celsius)', -5.0, 5.0, 1.6)
    future_weather_precipitation = st.slider('Precipitation Percentage Change', -10.0, 10.0, 5.5)
    future_date_vs_weather_data = \
        data_processing.predict_future_weather_data(present_weather_data,
                                                    future_weather_temperature,
                                                    future_weather_precipitation)
    future_date_vs_weather_data = future_date_vs_weather_data.set_index('Date/Time')
    future_fire_locations = data_processing.predict_fires(
                                          data_processing.predict_future_weather_data(
                                              assemble_data.read_modern_weather_data(),
                                              future_weather_temperature, future_weather_precipitation),
                                          2019)
    st.write('''Here is the data you have configured:''')
    st.write('''Future Temperatures''')
    st.line_chart(future_date_vs_weather_data['Max Temp (°C)'])
    st.write('''Future Precipitation''')
    st.line_chart(future_date_vs_weather_data['Total Precip (mm)'])

    fires_2019_data = \
        data_processing.future_fires_per_month_graph_data(future_weather_temperature, 
                                                          future_weather_precipitation)
    fires_2019_data[0] = fires_2019_data[0].set_index('MONTH')
        
    future_fires_data = \
        data_processing.future_fires_per_month_graph_data(future_weather_temperature, 
                                                          future_weather_precipitation)
    future_fires_data[1] = future_fires_data[1].set_index('MONTH')

    st.write('''Again, here is the actual 2019 data:''')
    st.line_chart(fires_2019_data[0]['# of fires'])
    st.write('''And here is the predicted fire weather data for the offesets:''')
    st.line_chart(future_fires_data[1]['# of fires'])
    st.write('''Additionally, here is the 2019 location data:''')
    st.pydeck_chart(heatmap_ontario(data_processing.get_2019_fire_locations()[['lon', 'lat']],
                                    '[lon, lat]'))
    st.write('''Finally, here is the predicted heat map of fire locations:''')
    st.pydeck_chart(heatmap_ontario(future_fire_locations[['lon', 'lat']], '[lon, lat]'))
    
    st.header('''Analysing the results of the Climate Scientist's numbers''')
    st.write('''Under the offsets calculated by actual climate scientists, we might be able to draw
             some interesting conclusions about both the correctness of our model and the 
             frequency and locality of the fires of the future.''')


def heatmap_ontario(data: pd.DataFrame, positions: str) -> pdk.Deck:
    """Return a heatmap of data over Ontario. The coordinates of data
    are specified through positions, a string that describes a list of the longitude
    and latitude column names: e.g '[lon, lat]'."""
    map_ontario = pdk.Deck(
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
                data=data,
                opacity=0.3,
                get_position=positions
            )
        ]
    )
    return map_ontario


def generate_regressive_plot(data: pd.DataFrame,
                             x: str, y: str, title: str) -> go.Figure():
    """Return a plotly scatter plot with a linear regression line along the data.
    
    data is the data passed in, and x and y are the axis lables (that exist as columns in the data).
    
    title is the title of the graph.
    """
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

