"""Helper functions to preform data processing/manipulation/rearranging."""
import pandas as pd
import streamlit as st
import assemble_data
import datetime
import statistics
from dataclasses import dataclass

from typing import List, Optional, Tuple, Dict

CAUSE_REFERENCES = {'IDF': 'Foresting', 'IDO': 'Industrial',
                    'INC': 'Incedniary', 'LTG': 'Lightning',
                    'MIS': 'Miscellaneous', 'REC': 'Recreation',
                    'RES': 'Resident', 'RWY': 'Railway', 'UNK': 'Unkown'}


@dataclass
class WeatherAreaRegression():
    """A dataclass containing a three-column pd.DataFrame and linear regression constants
    for area burned vs temperature and weather

    Instance Attributes:
        - data: the DataFrame holding the data
        - temp_regres: the regression constants against temperature data
        - precip_regres: the regression constants agains the precipitation data
    """
    data: pd.DataFrame
    regressions: Dict[str, Tuple[int, int]]


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
    datetime.date()
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
def area_burned_vs_weather(processed_data: pd.DataFrame,
                           area_max_threshold: Optional[float] = 10000000) -> WeatherAreaRegression:
    """
    Return a pandas dataframe with the area burned of the fire on one axis, and the 
    factor specified by factor on the other axis.

    The factor is averaged over the time period.

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
    regressions = {'TEMPERATURE': linear_regression(frame, 'TEMPERATURE', 'AREA BURNED'),
                   'PRECIPITATION': linear_regression(frame, 'PRECIPITATION', 'AREA BURNED')}
    return WeatherAreaRegression(frame, regressions)
        

def linear_regression(data: pd.DataFrame, x: str, y: str) -> Tuple[float, float]:
    """
    Return the a and b values (constant and slope) of the simple linear regression over
    data for columns x and y.
    """
    x_mean = data[x].mean()
    y_mean = data[y].mean()
    
    b_numerator = sum((data[x][i] - x_mean) * (data[y][i] - y_mean) for i in range(1, data.shape[0]))
    b_denominator = sum((data[x][i] - x_mean) ** 2 for i in range(1, data.shape[0]))
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
