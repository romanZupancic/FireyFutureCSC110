"""Helper functions to preform data processing/manipulation/rearranging."""
import random
from typing import List, Optional, Tuple, Union
import datetime
import statistics
import pandas as pd
import streamlit as st
import tensorflow as tf
import numpy as np

from assemble_data import read_modern_weather_data, remove_nan_weather

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
        else:  # If the cause is not yet in the dictionary
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
            data_so_far[date] = [0 for x in CAUSE_REFERENCES]

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


@st.cache
def fire_frequency_over_time(frames: List[pd.DataFrame],
                             scale: Optional[str] = 'month') -> pd.DataFrame:
    """
    Return the amount of fires over the time the dataset is collected for.
    Accumulate the amount of fires base on scale.

    Preconditions:
        - scale must be 'year', 'month', or 'day', case-insensitive
    """
    data_so_far = {}
    for data in frames:
        for _, row in data.iterrows():
            date = scale_date(row['FIRE_START_DATE'], scale)
            if date not in data_so_far:
                data_so_far[date] = [1]
            else:
                data_so_far[date][0] += 1

    frequency_df = pd.DataFrame(data_so_far)
    transposed_df = frequency_df.transpose()
    transposed_df.columns = ['# of fires']
    return transposed_df


def scale_date(in_date: Union[datetime.date, str],
               scale: Optional[str] = 'month') -> datetime.datetime:
    """
    Returns the input date as a datetime.datetime object whose actual values are only set
    at scale. Values smaller than scale are set to 1.

    in_date: Either a string of the format '2001/06/09 00:00:00+00' or a
    datetime.datetime object

    scale: either 'DAY', 'MONTH', or 'YEAR'. Case insensitive
    """
    if isinstance(in_date, datetime.date):
        if scale.upper() == 'YEAR':
            date = datetime.datetime(year=in_date.year, month=1, day=1)
        elif scale.upper() == 'MONTH':
            date = datetime.datetime(year=in_date.year, month=in_date.month, day=1)
        else:  # scale.upper() == 'DAY':
            date = in_date
    else:
        date = string_date_to_date(in_date, scale)

    return date


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
    """
    Return a dataframe with two columns
    The first column contains the average temperature for each row in data
    The second column contains the total precipitation for each row in data
    """
    fire_df = data.copy()
    temp_list = []
    precip_list = []
    for _, value in fire_df.iterrows():
        temp_str = value['TEMPERATURE'].split(',')
        precip_str = value['PRECIPITATION'].split(',')
        temp_list.append(statistics.mean([float(x) for x in temp_str]))
        precip_list.append(sum([float(x) for x in precip_str]))
    frame = pd.DataFrame({'TEMPERATURE': temp_list, 'PRECIPITATION': precip_list})
    return frame


def predict_future_weather_data(weather_data: pd.DataFrame,
                                delta_temp: Optional[float] = 1.6,
                                multi_precip: Optional[float] = 5.5) -> pd.DataFrame:
    """
    Apply a shift to the input weather data by adding delta_temp to the temperature
    and multiplying by multiplying total precipitation by multi_precip.

    The default values are estimated by scientists for 2050.

    Preconditions:
        - data must have collumn headers 'Max Temp (°C)' and 'Total Precip (mm)'
    """
    data = weather_data.copy()
    for idx, row in data.iterrows():
        new_temp = row['Max Temp (°C)'] + delta_temp
        data.at[idx, 'Max Temp (°C)'] = new_temp
        new_precip = row['Total Precip (mm)'] * (1 + (multi_precip / 100))
        data.at[idx, 'Total Precip (mm)'] = new_precip
    return data


@st.cache
def predict_fires(weather_data: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Based on a year's worth on input weather data, predict the number of fires
    which will occur un that year

    This function uses the dlstm model to make predictions
    """
    data = weather_data.copy()
    weather_sequences = ([], [])
    weather_sequences_loc_reference = ([], [])
    weather_sequences_date = []
    stations = set(data['Station Name'])
    for station in stations:
        station_data = data.loc[data['Station Name'] == station]
        temp_list = list(station_data['Max Temp (°C)'])
        precip_list = list(station_data['Total Precip (mm)'])
        for i in range(len(temp_list) - 20):
            temperature = remove_nan_weather(temp_list[i:i + 21])
            precipitation = remove_nan_weather(precip_list[i:i + 21])
            if temperature != ['INVALID'] and precipitation != ['INVALID']:
                weather_sequences[0].append(temperature)
                weather_sequences[1].append(precipitation)
                weather_sequences_loc_reference[0].append(list(station_data['Longitude (x)'])[0]
                                                          + random.uniform(-0.5, 0.5))
                weather_sequences_loc_reference[1].append(list(station_data['Latitude (y)'])[0]
                                                          + random.uniform(-0.5, 0.5))          
                weather_sequences_date.append(i + 21)
    model = tf.keras.models.load_model('./data/models/dlstm_v5.h5')
    predictions = model.predict({'TEMPERATURE': np.array(weather_sequences[0]),
                                 'PRECIPITATION': np.array(weather_sequences[1])})
    fire_loc_reference = ([], [])
    fire_date_reference = []
    for i in range(len(predictions)):
        if predictions[i] >= 0.94:
            fire_loc_reference[0].append(weather_sequences_loc_reference[0][i])
            fire_loc_reference[1].append(weather_sequences_loc_reference[1][i])
            fire_date_reference.append(datetime.date(year=year, month=1, day=1) 
                                       + datetime.timedelta(days=weather_sequences_date[i]))
    return pd.DataFrame({'lon': fire_loc_reference[0], 'lat': fire_loc_reference[1],
                         'FIRE_START_DATE': fire_date_reference})


@st.cache
def get_2019_fire_locations() -> pd.DataFrame:
    """
    Return a dataframe containing the longitude and latitude coordinates of every fire
    which occured in 2019
    """
    fire_point = pd.read_csv('./data/training_data/Fire_Disturbance_Point_Processed.csv')
    fire_area = pd.read_csv('./data/training_data/Fire_Disturbance_Area_Processed.csv')
    actual_fire_longitudes = list(fire_point.loc[fire_point['FIRE_YEAR'] == 2019]['LONGITUDE'])
    actual_fire_latitudes = list(fire_point.loc[fire_point['FIRE_YEAR'] == 2019]['LATITUDE'])
    actual_fire_dates = list(fire_point.loc[fire_point['FIRE_YEAR'] == 2019]['FIRE_START_DATE'])
    for _, row in fire_area.iterrows():
        year = row['FIRE_START_DATE'].split(' ')[0].split('/')[0]
        if int(year) == 2019:
            actual_fire_longitudes.append(row['LONGITUDE'])
            actual_fire_latitudes.append(row['LATITUDE'])
            actual_fire_dates.append(row['FIRE_START_DATE'])
    return pd.DataFrame({'lon': actual_fire_longitudes,
                         'lat': actual_fire_latitudes,
                         'FIRE_START_DATE': actual_fire_dates})


def get_2019_fires_per_month() -> pd.DataFrame:
    """
    Return a dataframe connecting the month of 2019 to the number of fires which occured
    in that month
    """
    fire_point = pd.read_csv('./data/training_data/Fire_Disturbance_Point_Processed.csv')
    fire_area = pd.read_csv('./data/training_data/Fire_Disturbance_Area_Processed.csv')
    fires_per_month = fire_frequency_over_time([fire_point, fire_area], 'month')
    fires_2019_per_month = []
    fires_2019_start_dates = []
    for row, value in fires_per_month.iterrows():
        if row.year == 2019:
            fires_2019_per_month.append(value['# of fires'])
            fires_2019_start_dates.append(row)
    return pd.DataFrame({'MONTH': fires_2019_start_dates, '# of fires': fires_2019_per_month})


def future_fires_per_month_graph_data(temp_adjust: float,
                                      precip_adjust: float) -> List[pd.DataFrame]:
    """
    Predict the number of fires per month in 2050 using the dlstm model and predicted 
    weather data for 2050

    Return a list containing two dataframes. One dataframe contains the number of fires
    per month in 2019 and the other contains the number of fires per month in 2050.
    """
    fires_2019 = get_2019_fires_per_month()
    fires_2050 = fire_frequency_over_time([predict_fires(
                                           predict_future_weather_data(read_modern_weather_data(),
                                                                       temp_adjust, precip_adjust),
                                                                       2019)], 'month')
    combined_fires = {'MONTH': [], '# of fires': [], 'YEAR': []}
    combined_fires2 = {'MONTH': [], '# of fires': [], 'YEAR': []}
    for row, value in fires_2019.iterrows():
        combined_fires['MONTH'].append(value['MONTH'])
        combined_fires['# of fires'].append(value['# of fires'])
        combined_fires['YEAR'].append('2019')
    for row, value in fires_2050.iterrows():
        combined_fires2['MONTH'].append(row)
        combined_fires2['# of fires'].append(value['# of fires'])
        combined_fires2['YEAR'].append('2050')
    return [pd.DataFrame(combined_fires), pd.DataFrame(combined_fires2)]


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['pandas', 'streamlit', 'datetime', 'assemble_data',
                          'statistics', 'tensorflow', 'numpy', 'random', 'typing'],  
        'allowed-io': [], 
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })
    