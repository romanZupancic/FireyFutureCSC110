"""Helper functions to preform data processing/manipulation/rearranging."""
import pandas as pd
import streamlit as st
import assemble_data
import datetime
import statistics
import math
from dataclasses import dataclass
import tensorflow as tf
import numpy as np
import random

from typing import List, Optional, Tuple, Dict

from assemble_data import to_list, get_used_weather_files, remove_nan_weather, read_fire_disturbance_area_processed, read_fire_disturbance_point_processed

CAUSE_REFERENCES = {'IDF': 'Foresting', 'IDO': 'Industrial',
                    'INC': 'Incedniary', 'LTG': 'Lightning',
                    'MIS': 'Miscellaneous', 'REC': 'Recreation',
                    'RES': 'Resident', 'RWY': 'Railway', 'UNK': 'Unkown'}


# The st.cache decorator allows streamlit to cach the result of the operation, so
# the page can be reloaded quickly when the inputs to this function are not changed
def fire_cause_count(fire_data: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of times each type of fire cause appears in the dataset

    Preconditions:
        - fire_data is a pd.DataFrame that follows the expected format of the Fire Disturbance
        area dataset or Fire Disturbance Point dataset
    """
    cause_series = fire_data['FIRE_GENERAL_CAUSE_CODE']
    cause_dict = {}
    for _, fire_cause in cause_series.items():
        # If the cause has been encountered before
        if CAUSE_REFERENCES[fire_cause] in cause_dict:
            cause_dict[CAUSE_REFERENCES[fire_cause]][0] += 1
        else: # If the cause is not yet in the dictionary
            cause_dict[CAUSE_REFERENCES[fire_cause]] = [1]
    cause_df = pd.DataFrame(cause_dict)
    transposed_df = cause_df.transpose()
    transposed_df.index.name = 'Cause'
    transposed_df.columns = ['Count']
    return transposed_df


@st.cache
def fire_cause_over_time(fire_data: pd.DataFrame, causes: List[str],
                         scale: Optional[str] = '') -> pd.DataFrame:
    """
    Return a pd.DataFrame that lists the Fire Causes found in the list causes over a timescale scale
    from a fire disturbance dataset.

    Preconditions:
        - fire_data is of the format that is expected of fire disturbance point and area datasets
        - causes is a list of causes that are keys in the CAUSE_REFERENCES dictionary
    """
    data_so_far = {}
    for _, row in fire_data.iterrows():
        date = string_date_to_date(row['FIRE_START_DATE'], scale)
        if date not in data_so_far:
            data_so_far[date] = [0 for _ in CAUSE_REFERENCES]

        cause_code = row['FIRE_GENERAL_CAUSE_CODE']
        if cause_code in causes:
            fire_cause_idx = list(CAUSE_REFERENCES.keys()).index(cause_code)
            data_so_far[date][fire_cause_idx] += 1

    cause_df = pd.DataFrame(data_so_far)
    transposed_df = cause_df.transpose()
    transposed_df.columns = list(CAUSE_REFERENCES.keys())
    return transposed_df


def string_date_to_date(date_time: str, scale: Optional[str] = '') -> datetime.datetime:
    """
    Converts a date in the form of a string, as found in the fire data sets, to a datetime
    object. The item passed through scale is NOT case sensitive.

    The datetime object has a resolution according to scale, where:
        - 'year' limits the datetime to year
        - 'month' limits the datetime to month and year
        - other attributes are set to 1

    >>> string_date_to_datetime('2001/06/09 00:00:00+00')
    """
    date = date_time.split(' ')
    year, month, day = date[0].split('/')
    if scale.upper() == 'YEAR':
        return datetime.datetime(int(year), 1, 1)
    if scale.upper() == 'MONTH':
        return datetime.datetime(int(year), int(month), 1)
    else:
        return datetime.datetime(int(year), int(month), int(day))


def fire_frequency_over_time(frames: List[pd.DataFrame], 
                             scale: Optional[str] = 'month') -> pd.DataFrame:
    ''''''
    data_so_far = {}
    for data in frames:
        for _, row in data.iterrows():
            date = string_date_to_date(row['FIRE_START_DATE'], scale)
            if date not in data_so_far:
                data_so_far[date] = [1]
            else:
                data_so_far[date][0] += 1

    frequency_df = pd.DataFrame(data_so_far)
    transposed_df = frequency_df.transpose()
    transposed_df.columns = ['# of fires']
    return transposed_df


@st.cache
def area_burned_vs_weather(processed_data: pd.DataFrame,
                           area_max_threshold: Optional[float] = 10000000) -> pd.DataFrame:
    """
    Return a pandas dataframe with the area burned by a fire and the
    average temperatue and precipitation in the 21 days leading up to that fire.

    Does not process fires greater than area_max_threshold.

    The factor is averaged over the 21 days.

    The processed_data argument has to take in data with the _Processed suffix.
    """
    factors = {'AREA BURNED': [],
               'TEMPERATURE': [],
               'PRECIPITATION': []}

    for _, row in processed_data.iterrows():
        if row['FIRE_FINAL_SIZE'] <= area_max_threshold:
            factors['AREA BURNED'].append(row['FIRE_FINAL_SIZE'])
            for factor in ['TEMPERATURE', 'PRECIPITATION']:
                factor_full = [float(temp) for temp in row[factor.upper()].split(',')]
                factors[factor].append(statistics.mean(factor_full))

    frame = pd.DataFrame(factors)
    return frame


def linear_regression(data: pd.DataFrame, x: str, y: str) -> Tuple[float, float]:
    """
    Return the a and b values (constant and slope) of the simple linear regression over
    data for columns x and y.
    """
    x_mean = data[x].mean()
    y_mean = data[y].mean()

    b_numerator = sum((data[x][i] - x_mean) * (data[y][i] - y_mean) for i in range(0, len(data)))
    b_denominator = sum((data[x][i] - x_mean) ** 2 for i in range(0, len(data)))
    b = b_numerator / b_denominator

    a = y_mean - b * x_mean

    return (a, b)


def evaluate_linear_equation(a: int, b: int, x: int) -> int:
    """
    Return the value of the linear equation a + bx evalueated at x.
    """
    return a + b * x


@st.cache
def graph_weather(data: pd.DataFrame) -> pd.DataFrame:
    fire_df = data.copy()
    temp_list = []
    precip_list = []
    for row, value in fire_df.iterrows():
        temp_str = value['TEMPERATURE'].split(',')
        precip_str = value['PRECIPITATION'].split(',')
        temp_list.append(statistics.mean([float(x) for x in temp_str]))
        precip_list.append(sum([float(x) for x in precip_str]))
    frame = pd.DataFrame({'TEMPERATURE': temp_list, 'PRECIPITATION': precip_list})
    return frame


def predict_future_weather_data(weather_data: pd.DataFrame, 
                                delta_temp: Optional[float] = 1.6, 
                                multi_precip: Optional[float] = 1.055) -> pd.DataFrame:
    """
    Apply a shift to the input weather data by adding delta_temp to the temperature 
    and multiplying by multiplying total precipitation by multi_precip.

    The default values are estimated by scientists for 2050.

    Preconditions:
        - data must have collumn headers 'Max Temp (°C)' and 'Total Precip (mm)' 
    """
    data = weather_data.copy()
    for _, row in data.iterrows():
        new_temp = row['Max Temp (°C)'] + delta_temp
        row['Max Temp (°C)'] = new_temp
        new_precip = row['Total Precip (mm)'] * multi_precip
        row['Total Precip (mm)'] = new_precip
    return data


@st.cache
def predict_fires_2019(weather_data: Optional[pd.DataFrame] = pd.DataFrame()) -> List[pd.DataFrame]:
    #data = assemble_data.read_all_station_weather_data()
    print('Here1')
    data = pd.read_csv('./data/processed_data/weather_station_data_2019.csv')
    #data = weather_data.copy()    
    print('Here2')
    weather_sequences = ([], [])
    weather_sequences_loc_reference = ([], [])
    weather_sequences_date = []
    stations = set(data['Station Name'])
    print('Here3')
    for station in stations:
        station_data = data.loc[data['Station Name'] == station]
        temp_list = list(station_data['Max Temp (°C)'])
        precip_list = list(station_data['Total Precip (mm)'])
        for i in range(len(temp_list) - 20):
            #print(station_path)
            temperature = remove_nan_weather(temp_list[i:i+21])
            precipitation = remove_nan_weather(precip_list[i:i+21])
            #print(any([math.isnan(x) for x in temperature]))
            #print(any([math.isnan(x) for x in precipitation]))
            if temperature != ['INVALID'] and precipitation != ['INVALID']:
                weather_sequences[0].append(temperature)
                weather_sequences[1].append(precipitation)
                weather_sequences_loc_reference[0].append(list(station_data['Longitude (x)'])[0] + random.uniform(-0.5, 0.5))
                weather_sequences_loc_reference[1].append(list(station_data['Latitude (y)'])[0] + random.uniform(-0.5, 0.5))          
                weather_sequences_date.append(i + 21)
    print('Here4')
    model = tf.keras.models.load_model('./data/models/dlstm_v5.h5')
    predictions = model.predict({'TEMPERATURE': np.array(weather_sequences[0]), 'PRECIPITATION': np.array(weather_sequences[1])})
    fire_loc_reference = ([], [])
    fire_date_reference = []
    for i in range(len(predictions)):
        if predictions[i] >= 0.94:
            fire_loc_reference[0].append(weather_sequences_loc_reference[0][i])
            fire_loc_reference[1].append(weather_sequences_loc_reference[1][i])
            fire_date_reference.append(weather_sequences_date[i])
    print(sum([int(x >= 0.94) for x in predictions]))
    fire_point = pd.read_csv('./data/training_data/Fire_Disturbance_Point_Processed.csv')
    fire_area = pd.read_csv('./data/training_data/Fire_Disturbance_Area_Processed.csv')
    print(fire_frequency_over_time([fire_point, fire_area], 'year').loc['2019-01-01', :])
    #print(len(fire_point.loc[fire_point['FIRE_YEAR'] == 2011]))# + len(fires2.loc[fires['FIRE_YEAR'] == 2011]))
    actual_fire_longitudes = list(fire_point.loc[fire_point['FIRE_YEAR'] == 2019]['LONGITUDE']) #+ list(fire_area.loc['2019' in fire_area['FIRE_START_DATE']]['LONGITUDE'])
    actual_fire_latitudes = list(fire_point.loc[fire_point['FIRE_YEAR'] == 2019]['LATITUDE']) #+ list(fire_area.loc['2019' in fire_area['FIRE_START_DATE']]['LATITUDE'])

    return [pd.DataFrame({'lon': actual_fire_longitudes, 'lat': actual_fire_latitudes}), 
            pd.DataFrame({'lon': fire_loc_reference[0], 
                          'lat': fire_loc_reference[1],
                          'date': fire_date_reference})]


