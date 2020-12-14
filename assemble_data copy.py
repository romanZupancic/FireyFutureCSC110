"""Build the data sets from a variety of sources.
Additionally, provide functions that read the datasets from their csvs.

Functions with prefix "assemble" build the datasets into pandas dataframes.
Functions with prefix "make" convert dataframes to csvs.
Functions with prefux "read" read csvs into dataframes.
"""

import requests
import streamlit as st
import pandas as pd
import json
import os
from typing import List, Set, Optional
import math
import datetime
import statistics


DELIMITER = '","'  # Since the government likes microsoft, we have to use weird delimiters
DATA_DIRECTORY = "./data"

# Raw input files (from source)
FIRE_AREA_RAW = f'{DATA_DIRECTORY}/Fire_Disturbance_Area.geojson'
FIRE_POINT_RAW = f'{DATA_DIRECTORY}/Fire_Disturbance_Point.csv'
WEATHER_STATION_RAW = f'{DATA_DIRECTORY}/Station Inventory EN.csv'

# Output files
# Segregated sets
FIRE_AREA_FILTERED =f'{DATA_DIRECTORY}/processed_data/Fire_Disturbance_Area_Filtered.csv'
FIRE_POINT_FILTERED = f'{DATA_DIRECTORY}/processed_data/Fire_Disturbance_Point_Filtered.csv'
CULLED_WEATHER_DATA = f'{DATA_DIRECTORY}/processed_data/weather_station_data_2019.csv'
WEATHER_STATION_LOCATIONS = f'{DATA_DIRECTORY}/processed_data/weather_station_location_info.csv'

# Sets of combined data
FIRE_POINT_WEATHER = f'{DATA_DIRECTORY}/processed_data/Fire_Disturbance_Point_Weather.csv'
FIRE_AREA_WEATHER = f'{DATA_DIRECTORY}/processed_data/Fire_Disturbance_Area_Weather.csv'

# Invalid data removed
FIRE_POINT_PROCESSED = f'{DATA_DIRECTORY}/training_data/Fire_Disturbance_Point_Processed.csv'
FIRE_AREA_PROCESSED = f'{DATA_DIRECTORY}/training_data/Fire_Disturbance_Area_Processed.csv'
NO_FIRE_WEATHER_SEQUENCES = f'{DATA_DIRECTORY}/training_data/No_Fire_Weather_Sequences.csv'
NO_FIRE_WEATHER_SEQUENCES_SMALL = f'{DATA_DIRECTORY}/training_data/No_Fire_Weather_Sequences_Small.csv'

# Model Training data
SVM_DATA = f'{DATA_DIRECTORY}/training_data/svm_training_data.csv'
ANN_DATA = f'{DATA_DIRECTORY}/training_data/ann_training_data.csv'
DLSTM_DATA = f'{DATA_DIRECTORY}/training_data/dlstm_training_data.csv'


def assemble_all_data() -> None:
    """
    Assemble all data from just the three source (RAW) input files.

    Requires the files pointed to by the RAW constants to exist in the appropriate directory.
    """
    # Early filtering of the data sets
    make_fire_disturbance_point()
    make_fire_disturbance_area()

    # Organize weather station metadata and download weather data for individual stations
    make_individual_weather_data()
    make_weather_station_locations()

    make_fire_data_with_weather()

    make_processed_fire_weather_data()


def read_fire_disturbance_point() -> pd.DataFrame:
    """
    Read the csv containing fire disturbance point data, and return a dataframe
    representation of its content
    """
    fires = pd.read_csv(FIRE_POINT_FILTERED)
    return fires


def read_fire_disturbance_area() -> pd.DataFrame:
    """
    Return a pandas dataframe containing the data read from the fire disturbance area csv
    """
    fires = pd.read_csv(FIRE_AREA_FILTERED)
    return fires


def read_modern_weather_data() -> pd.DataFrame:
    """
    Read the weather station data from the csv, and return it as a pandas DataFrame
    """
    data = pd.read_csv(CULLED_WEATHER_DATA)
    data['Date/Time'] = pd.to_datetime(data['Date/Time'])
    return data


def read_fire_disturbance_point_processed() -> pd.DataFrame:
    """
    Read the csv containing fire disturbance point data, and return a dataframe
    representation of its content
    """
    fires = pd.read_csv(FIRE_POINT_PROCESSED)
    return fires


def read_fire_disturbance_area_processed() -> pd.DataFrame:
    """
    Return a pandas dataframe containing the data read from the fire disturbance area processed csv
    """
    fires = pd.read_csv(FIRE_AREA_PROCESSED)
    return fires


def read_no_fire_weather_sequences() -> pd.DataFrame:
    """
    Read the small-sized weather sequences from the csv, and return it as a pandas DataFrame
    """
    data = pd.read_csv(NO_FIRE_WEATHER_SEQUENCES_SMALL)
    return data


def assemble_fire_disturbance_point() -> pd.DataFrame:
    """
    Save a new csv containing only fire disturbance point data for fires which
    took place after the beginning of the year 1998

    The return value of this function is stored at:
    ./data/Fire_Disturbance_Point_Clean.csv
    """
    fires = pd.read_csv(FIRE_POINT_RAW)  # Get the raw data file
    fires = fires.rename(columns={'X': 'LONGITUDE', 'Y': 'LATITUDE'})
    remove_indicies = []
    # Filter out data based on the year it was collected for
    for row, item in fires.iterrows():
        print(row)
        if int(item['FIRE_START_DATE'].split('/')[0]) < 1998:
            remove_indicies.append(row)
    print('remove indicies')
    # Drop everything at once
    for row in remove_indicies:
        print(row)
        fires = fires.drop(row)
    return fires


def make_fire_disturbance_point() -> None:
    """
    Assemble and save fire disturbance area data to a csv (makes for faster loading in the
    future)
    """
    data = assemble_fire_disturbance_area()
    data.to_csv(FIRE_AREA_FILTERED)


def assemble_fire_disturbance_area() -> pd.DataFrame:
    """
    Filter and build a dataset from the raw 'Fire_Disturbance_Area.geojson' file.

    This function does not take any parameters because it is expected to be rewritten directly.

    The function extracts the desired properties of area fires from the file, as well
    as caculates the approximate location of a given area fire by averaging it's shape.
    """
    # Open the file
    with open(FIRE_AREA_RAW) as json_file:
        data = json.load(json_file)

        properties_so_far = []

        # Select data row-by-row
        for row in data['features']:
            if int(row['properties']['FIRE_START_DATE'].split('/')[0]) >= 1998:
                properties = [row['properties']['OGF_ID'],
                              row['properties']['FIRE_START_DATE'],
                              row['properties']['FIRE_GENERAL_CAUSE_CODE'],
                              row['properties']['FIRE_FINAL_SIZE']
                              ]

                # Analyse and retrieve the shape of the fire
                if row['geometry']['type'] == 'MultiPolygon':
                    collapsed = [coord for sub1 in row['geometry']['coordinates']
                                    for sub2 in sub1
                                    for coord in sub2]

                if row['geometry']['type'] == 'Polygon':
                    collapsed = [coord for sub1 in row['geometry']['coordinates']
                                    for coord in sub1]
                # Average the shape of the fire into one coordinate
                avg_long = sum([coord[0] for coord in collapsed]) / len(collapsed)
                avg_lat = sum([coord[1] for coord in collapsed]) / len(collapsed)
                properties.append(avg_long)
                properties.append(avg_lat)

                # Append the properties for that row
                properties_so_far.append(properties)

        # Name the headers for each column
        columnLabels = ['OGF_ID',
                        'FIRE_START_DATE',
                        'FIRE_GENERAL_CAUSE_CODE',
                        'FIRE_FINAL_SIZE',
                        'LONGITUDE',
                        'LATITUDE']
        # Build the pandas dataframe from the extracted data, and the column labels
        disturbance_area_fires = pd.DataFrame(properties_so_far, columns = columnLabels)
        return disturbance_area_fires


def make_fire_disturbance_area() -> None:
    """
    Assemble and save fire disturbance area data to a csv (makes for faster loading in the
    future)
    """
    data = assemble_fire_disturbance_area()
    data.to_csv(FIRE_AREA_FILTERED)


#  Assemble the weather data
def filter_stations(min_year: int) -> pd.DataFrame:
    """
    Return the rows of all weather stations that operate throughout years min_year and 2020
    """
    unfiltered = pd.read_csv(WEATHER_STATION_RAW, header=[2])

    # Filter data based on year collected for
    filtered_first_year = unfiltered[unfiltered['First Year'] <= min_year]
    filtered_last_year = filtered_first_year[unfiltered['Last Year'] == 2020]
    #Only look for Ontario data
    filtered_province = filtered_last_year[filtered_last_year['Province'] == 'ONTARIO']
    return filtered_province


def assemble_weather_data_by_station(stationID: int, min_year: int, max_year: int) -> pd.DataFrame:
    """
    Query an api for weather data of a given stationID throughout the years min_year - max_year
    """
    print(f'min_year: {min_year}, max_year: {max_year}')
    data_frames = []  # Accumulator

    # The api must be queried separately for each year. Government tech at its finest.
    for year in range(min_year, max_year + 1):
        print(year)
        data = requests.get(f'https://climate.weather.gc.ca/climate_data/bulk_data_e.html? \
                              format=csv&stationID={stationID}&Year={year}&Month=1&\
                              Day=14&timeframe=2&submit= Download+Data')

        # Convert the line endings characters to ones python-compatable
        fixed_data = data.text.replace('\r\n', '\n')
        fixed_data = [row.strip('"') for row in fixed_data.splitlines()]

        # Balance each row so that it is at least the length of the header
        for row in range(len(fixed_data)):
            fixed_data[row] += DELIMITER * (len(fixed_data[0].split(DELIMITER)) -
                                            len(fixed_data[row].split(DELIMITER)))

        print(fixed_data[0])

        # Build and append the dataframe to the accumulator
        data_frames.append(pd.DataFrame([x.split(DELIMITER) for x in fixed_data[1:]],
                                        columns=fixed_data[0].split(DELIMITER)))

    # Combine all rows into one dataframe, and return it
    return pd.concat(data_frames)


def make_individual_weather_data(year: int = 1998) -> None:
    """
    Produces csv files for every eligable weather station operating throughout year to 2020.
    """
    stations = filter_stations(year) # Get eligable stations

    # For every station that is eligable, get the weather data associated for that station
    # Within the tarket period (from year to present)
    for _, station in stations.iterrows():
        print(station)
        data = assemble_weather_data_by_station(station['Station ID'],
                                                year, station['Last Year'])

        # Export an individual station to CSV
        data.to_csv(f'{DATA_DIRECTORY}/weather_data/{station["Name"]}_{station["Station ID"]}.csv')


def assemble_weather_data_year(year: Optional[int] = 2019) -> pd.DataFrame:
    """
    Make a csv out of just weather data.
    """
    fire_point = read_fire_disturbance_point_processed() 
    fire_point = fire_point.loc[fire_point['FIRE_YEAR'] == year]
    stations = get_used_weather_files(fire_point, fire_point)
    # Gather all the data into one list
    # data_frames = [pd.read_csv(f'{DATA_DIRECTORY}/weather_data/{file}')
    #                for file in os.listdir(f'{DATA_DIRECTORY}/weather_data/') if '.csv' in file]
    data_frames = [pd.read_csv(f'{DATA_DIRECTORY}/weather_data/{file}')
                   for file in stations]

    data = pd.concat(data_frames)

    data = data.loc[data['Year'] == year]

    return data

    
def make_weather_data_year(year: Optional[int]) -> None:
    data = assemble_weather_data_year(year)
    data.to_csv(CULLED_WEATHER_DATA)


def assemble_weather_station_locations() -> pd.DataFrame:
    """
    Return a dataframe of the location of weather stations, by name.
    """
    coordinates = []
    # Iterate through all the weather station csvs
    for file in os.listdir(f'{DATA_DIRECTORY}/weather_data'):
        df = pd.read_csv(f'{DATA_DIRECTORY}/weather_data/' + file)
        long = set(df['Longitude (x)'])  # Extract longitude data
        lat = set(df['Latitude (y)'])  # Extract latitude data
        coordinates.append([file, list(lat)[0], list(long)[0]])
    final_coordinates = pd.DataFrame(coordinates, columns=['name', 'lat', 'lon'])
    return final_coordinates


def make_weather_station_locations() -> None:
    """
    Save the location of all weather stations, and their names, to a csv
    """
    data = assemble_weather_station_locations()
    data.to_csv(WEATHER_STATION_LOCATIONS)


def assemble_fire_data_with_weather(fire_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Return one dataframe with each fire location and the associated weather data in
    its area. This prevents us from haveing to make multiple lookups whenever we want weather
    data.
    """
    fire_data = fire_dataset.copy()
    #fire_data = fire_data.iloc[:15]  # For debugging
    temperature_list = []  # Accumulator
    precipitation_list = []  # Accumulator
    stations_list = []  # Accumulator
    for idx, row in fire_data.iterrows():
        date = row['FIRE_START_DATE']
        lat = row['LATITUDE']
        lon = row['LONGITUDE']
        temp, precip, station = get_closest_weather_data(date, lon, lat)
        temperature_list.append(temp)
        precipitation_list.append(precip)
        stations_list.append(station)
        print(idx)
    fire_data['WEATHER_STATION_PATH'] = stations_list
    fire_data['TEMPERATURE'] = temperature_list
    fire_data['PRECIPITATION'] = precipitation_list

    return fire_data


def make_fire_data_with_weather() -> None:
    """
    Save both the point and area fire data csvs with their weather information
    """
    fire_point_data = assemble_fire_data_with_weather(read_fire_disturbance_point())
    fire_point_data.to_csv(FIRE_POINT_WEATHER)
    fire_area_data = assemble_fire_data_with_weather(read_fire_disturbance_area())
    fire_area_data.to_csv(FIRE_AREA_WEATHER)


def get_closest_weather_data(date: str, lon: float, lat: float) -> List[str]:
    """
    Return the weather station data from the closest station to the given longitude and latitude
    coordinates.

    The return list is a string of temperatures, a string of precipitations, and the file path
    of the weather station.
    """
    station_file_path = choose_closest_station(lon, lat)
    station = pd.read_csv(f'{DATA_DIRECTORY}/weather_data/' + station_file_path)
    fire_date = date.split(' ')[0].split('/')

    # Format the date into a string that matches the csv
    fire_date = f'{fire_date[0]}-{fire_date[1]}-{fire_date[2]}'

    # Get the index of the data on the target date
    date_index = station.index[station['Date/Time'] == fire_date].tolist()[0]

    # Get the data on the date in question
    weather_data = station.iloc[date_index - 21: date_index]
    precipitation = list(weather_data['Total Precip (mm)'])
    temperature = list(weather_data['Max Temp (°C)'])

    # Remove invalid data (Nan data)
    temperature = remove_nan_weather(temperature)
    precipitation = remove_nan_weather(precipitation)

    # Format the data into strings
    temperature_str = ','.join([str(x) for x in temperature])
    precipitation_str = ','.join([str(x) for x in precipitation])

    return [temperature_str, precipitation_str, station_file_path]


def choose_closest_station(lon: float, lat: float) -> str:
    """
    Return the file path for the closest weather station to the given longitude and latitude
    coordinates
    """
    possible_weather_stations = pd.read_csv(WEATHER_STATION_LOCATIONS)
    # Initialize accumulator variable
    min_dist = [10000000000000000000, []]  # An impossible large number and an empty list
    for _, station in possible_weather_stations.iterrows():
        distance = get_distance([station['lon'], station['lat']], [lon, lat])
        if distance < min_dist[0]:
            min_dist = [distance, [station['name'], station['lon'], station['lat']]]
    return min_dist[1][0]


def get_distance(station: List[float], fire: List[float]) -> float:
    """
    Return the distance between two longitude and latitude coordinates, given by station
    (the location of the weather station), and fire (the location of the fire)
    """
    radius = 6371.009

    latitude_discrepancy = math.radians(station[1] - fire[1])
    longitude_discrepancy = math.radians(station[0] - fire[0])

    # This formula was given by Tutorial 11
    angle = 2 * math.asin(math.sqrt(math.sin(latitude_discrepancy / 2) ** 2 +
                          math.cos(math.radians(station[1])) * math.cos(math.radians(fire[1])) *
                          math.sin(longitude_discrepancy) ** 2))
    return angle * radius


def remove_nan_weather(weather_input: List[int]) -> List:
    """Remove nan values and replace them with the average value of the sequence"""
    weather = weather_input.copy()
    non_nan = []
    for i in range(len(weather)):
        try:
            if math.isnan(weather[i]) == False:
                non_nan.append(weather[i])
        except:
            # The data file contains an invalid character ("M")
            pass
    if (len(non_nan) < 14):
        weather = ['INVALID']  # Rewrite the weather data with a more clear indicator
    else:
        avg_temp = statistics.mean(non_nan)
        for i in range(len(weather)):
            try:
                if math.isnan(weather[i]) == True:
                    weather[i] = avg_temp
            except:
                if type(weather[i]) == str:
                    weather[i] = avg_temp
    return weather


def assemble_processed_fire_weather_data(input_path: str) -> pd.DataFrame:
    """
    Remove all rows from the input path that contain
    INVALID values for temperature or precipitation
    """
    fires = pd.read_csv(input_path)
    fires = fires.drop(index=fires.index[fires['TEMPERATURE'] == 'INVALID'])
    fires = fires.drop(index=fires.index[fires['PRECIPITATION'] == 'INVALID'])
    return fires


def make_processed_fire_weather_data() -> None:
    """
    Produces two csv's (for point and area data) that contains all fires with legitament weather
    information (without INVALID entries).
    """
    processed_point = assemble_processed_fire_weather_data(FIRE_POINT_WEATHER)
    processed_point.to_csv(FIRE_POINT_PROCESSED)

    processed_area = assemble_processed_fire_weather_data(FIRE_AREA_WEATHER)
    processed_area.to_csv(FIRE_AREA_PROCESSED)


def get_used_weather_files(fire_point, fire_area) -> Set[str]:
    """
    Return a set of all the weather stations that we used to get weather data for each of the fires
    """
    used_stations = set.union(set(fire_point['WEATHER_STATION_PATH']),
                              set(fire_area['WEATHER_STATION_PATH']))
    return used_stations


def create_no_fire_weather_data() -> None:
    """
    Get a dataset containing sequences of weather data which did not result in there being a fire
    """
    no_fire_weather_df = pd.DataFrame(columns=['TEMPERATURE', 'PRECIPITATION'])
    fire_point = pd.read_csv(FIRE_POINT_PROCESSED)
    fire_area = pd.read_csv(FIRE_AREA_PROCESSED)
    used_stations = get_used_weather_files(fire_point, fire_area)
    for station in list(used_stations):
        # if station != 'CARIBOU ISLAND (AUT)_7582.csv':
        print(station)
        fire_point_days = fire_point.loc[fire_point['WEATHER_STATION_PATH'] == station]
        fire_area_days = fire_area.loc[fire_area['WEATHER_STATION_PATH'] == station]
        fire_point_days = fire_point_days['FIRE_START_DATE']
        fire_area_days = fire_area_days['FIRE_START_DATE']
        fire_days = set.union(set(fire_point_days), set(fire_area_days))
        print(len(fire_days))
        no_fire_weather_df = pd.concat([no_fire_weather_df, get_no_fire_weather_sequences(fire_days, station)])
    no_fire_weather_df.to_csv(NO_FIRE_WEATHER_SEQUENCES)


def get_no_fire_weather_sequences(fire_days, station_path) -> List:
    """Create a csv file containing 21 day sequences of weather data in which a fire did not occur at the end of the 21 days"""
    weather_sequences = []
    weather_data = pd.read_csv('./data/weather_data/' + station_path)
    weather_data = weather_data.reset_index()
    index_list = []
    for day in fire_days:
        date = day.split(' ')[0]
        date = date.replace('/', '-')
        index_list.append(weather_data.index[weather_data['Date/Time'] == date][0])
    list.sort(index_list)
    valid_indices = []
    for i in range(0, len(index_list) - 1):
        if index_list[i + 1] - index_list[i] > 28:
            valid_indices.append([index_list[i] + 3, index_list[i + 1] - 3])
    for pair in valid_indices:
        weather_interval = weather_data.iloc[pair[0]:pair[1]]
        temp_list = list(weather_interval['Max Temp (°C)'])
        precip_list = list(weather_interval['Total Precip (mm)'])
        precip_list = remove_nan_weather(precip_list)
        for i in range(len(temp_list) - 20):
            #print(station_path)
            temperature = remove_nan_weather(temp_list[i:i+21])
            precipitation = remove_nan_weather(precip_list[i:i+21])
            if temperature != ['INVALID'] and precipitation != ['INVALID']:
                temperature = to_list_of_str(temperature)
                precipitation = to_list_of_str(precipitation)
                weather_sequences.append([','.join(temperature), ','.join(precipitation)])
    return pd.DataFrame(weather_sequences, columns=['TEMPERATURE', 'PRECIPITATION'])


def to_list_of_str(lst: List) -> List[str]:
    """
    Take in a list of int or floats and return a list containing
    string representations of the floats or ints
    """
    new_lst = []
    for value in lst:
        new_lst.append(str(value))
    return new_lst


def shrink_no_fire_weather_csv():
    """
    Save a smaller csv containing randomly sampled data from weather_df
    """
    weather_df = pd.read_csv(NO_FIRE_WEATHER_SEQUENCES)
    weather_df = weather_df.sample(frac=0.09, random_state=1)
    weather_df.to_csv(NO_FIRE_WEATHER_SEQUENCES_SMALL)


def assemble_svm_training_csv() -> None:
    """Save a csv file containing the data required to train the support vector machine"""
    fire_point = read_fire_disturbance_point_processed()
    fire_area = read_fire_disturbance_area_processed()
    no_fire_weather = read_no_fire_weather_sequences()
    temperature_data = list(fire_point['TEMPERATURE']) + list(fire_area['TEMPERATURE']) + list(no_fire_weather['TEMPERATURE'])
    precipitation_data = list(fire_point['PRECIPITATION']) + list(fire_area['PRECIPITATION']) + list(no_fire_weather['PRECIPITATION'])
    # fire_indicator shows whether or not a fire occured after the 21 day sequence (1 = fire, 0 = no fire)
    fire_indicator = [1] * len(fire_point) + [1] * len(fire_area) + [0] * len(no_fire_weather)
    final_data = pd.DataFrame({'TEMPERATURE': get_averages(temperature_data), 'PRECIPITATION': get_sums(precipitation_data), 'FIRE': fire_indicator})
    final_data.to_csv(SVM_DATA)


def get_averages(lst: List):
    """
    Return a list containing the average of each string sequence
    Precondintions:
        - len(lst) > 0
    """
    avg_list = []
    for string in lst:
        splitted = string.split(',')
        avg_list.append(statistics.mean([float(x) for x in splitted]))
    return avg_list


def get_sums(lst: List):
    """
    Return a list containing the sum of each string sequence
    Precondintions:
        - len(lst) > 0
    """
    avg_list = []
    for string in lst:
        splitted = string.split(',')
        avg_list.append(sum([float(x) for x in splitted]))
    return avg_list


def assemble_ann_training_csv() -> None:
    """Save a csv file containing the data required to train the support vector machine"""
    fire_point = read_fire_disturbance_point_processed()
    fire_area = read_fire_disturbance_area_processed()
    no_fire_weather = read_no_fire_weather_sequences()
    temperature_data = list(fire_point['TEMPERATURE']) + list(fire_area['TEMPERATURE']) + list(no_fire_weather['TEMPERATURE'])
    precipitation_data = list(fire_point['PRECIPITATION']) + list(fire_area['PRECIPITATION']) + list(no_fire_weather['PRECIPITATION'])
    # fire_indicator shows whether or not a fire occured after the 21 day sequence (1 = fire, 0 = no fire)
    fire_indicator = [1] * len(fire_point) + [1] * len(fire_area) + [0] * len(no_fire_weather)
    final_data = pd.DataFrame({'WEATHER': get_lists(temperature_data, precipitation_data), 'FIRE': fire_indicator})
    final_data.to_csv(ANN_DATA)


def get_lists(temp: List, precip: List):
    """
    Return a list of lists containing float representation for each value in each string sequence
    Concatenate the temperature and precipitation lists together
    Precondintions:
        - len(temp) > 0
        - len(precip) > 0
    """
    new_list = []
    for i in range(len(temp)):
        splitted_temp = temp[i].split(',')
        splitted_precip = precip[i].split(',')
        splitted = splitted_temp + splitted_precip
        new_list.append([float(x) for x in splitted])
    return new_list


def assemble_dlstm_training_csv() -> None:
    """Save a csv file containing the data required to train the support vector machine"""
    fire_point = read_fire_disturbance_point_processed()
    fire_area = read_fire_disturbance_area_processed()
    no_fire_weather = read_no_fire_weather_sequences()
    temperature_data = list(fire_point['TEMPERATURE']) + list(fire_area['TEMPERATURE']) + list(no_fire_weather['TEMPERATURE'])
    precipitation_data = list(fire_point['PRECIPITATION']) + list(fire_area['PRECIPITATION']) + list(no_fire_weather['PRECIPITATION'])
    # fire_indicator shows whether or not a fire occured after the 21 day sequence (1 = fire, 0 = no fire)
    fire_indicator = [1] * len(fire_point) + [1] * len(fire_area) + [0] * len(no_fire_weather)
    final_data = pd.DataFrame({'TEMPERATURE': to_list(temperature_data), 'PRECIPITATION': to_list(precipitation_data), 'FIRE': fire_indicator})
    final_data.to_csv(DLSTM_DATA)


def to_list(lst: List):
    """
    Return a list of lists containing float representation for each value in each string sequence
    This function is different from get_lists() because it does not concatenate 2 input lists
    Precondintions:
        - len(lst) > 0
    """
    new_list = []
    for i in range(len(lst)):
        splitted = lst[i].split(',')
        new_list.append([float(x) for x in splitted])
    return new_list


if __name__ == '__main__':
    # print(answer)
    pass

