"""Helper functions to preform data processing/manipulation/rearranging."""
import pandas as pd
import streamlit as st
import assemble_data
import datetime

from typing import List, Optional

CAUSE_REFERENCES = {'IDF': 'Foresting', 'IDO': 'Industrial',
                    'INC': 'Incedniary', 'LTG': 'Lightning',
                    'MIS': 'Miscellaneous', 'REC': 'Recreation',
                    'RES': 'Resident', 'RWY': 'Railway', 'UNK': 'Unkown'}


# The st.cache decorator allows streamlit to cach the result of the operation, so
# the page can be reloaded quickly when the inputs to this function are not changed
@st.cache
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



WEATHER_STATION_COORDINATES = [[-83.09, 42.1], [-76.11, 45.19], [-88.91, 50.29], [-91.63, 48.76], [-77.88, 45.07], [-93.97, 48.63], [-77.39, 44.15], [-82.12, 49.38], [-75.67, 44.6], [-79.8, 43.3], [-88.34, 49.15], [-85.83, 47.33], [-76.91, 44.4], [-77.37, 46.05], [-78.17, 43.95], [-78.18, 43.97], [-79.54, 44.63], [-80.22, 44.5], [-74.75, 45.02], [-81.73, 45.33], [-80.55, 42.87], [-76.25, 45.03], [-81.9, 42.25], [-80.38, 43.7], [-80.33, 43.73], [-78.97, 42.88], [-79.88, 43.64], [-81.72, 43.77], [-82.95, 45.63], [-75.85, 44.42], [-78.53, 45.03], [-79.91, 43.29], [-76.69, 44.43], [-75.63, 45.0], [-81.48, 45.97], [-81.62, 44.17], [-82.67, 42.04], [-80.0, 48.15], [-79.22, 44.55], [-87.94, 52.2], [-81.15, 43.03], [-80.05, 42.53], [-76.08, 44.52], [-82.02, 46.19], [-80.65, 51.27], [-80.75, 43.98], [-84.16, 49.75], [-81.64, 42.51], [-76.78, 45.05], [-79.44, 44.6], [-78.83, 43.87], [-75.72, 45.38], [-85.43, 54.98], [-77.32, 45.95], [-77.25, 45.88], [-90.22, 51.45], [-77.15, 43.83], [-79.25, 42.87], [-79.25, 42.88], [-79.22, 43.25], [-93.72, 49.65], [-79.33, 43.04], [-80.47, 43.35], [-94.76, 49.47], [-79.63, 44.4], [-81.21, 42.77], [-75.06, 45.29], [-81.64, 42.98], [-80.72, 42.86], [-79.47, 43.78], [-77.53, 44.12], [-79.16, 44.26], [-90.47, 49.03], [-89.12, 48.37], [-80.37, 45.03], [-82.93, 42.33], [-80.77, 43.14], [-81.15, 43.86]]
