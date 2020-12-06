"""Build the data sets from a variety of sources.
Additionally, provide functions that read the datasets from their csvs.

Functions with prefix "assemble" build the datasets into pandas dataframes.
Functions with prefix "make" convert dataframes to csvs.
Functions with prefux "read" read csvs into dataframes.
"""

import requests
import pandas as pd
import json
import os
from typing import List
import math
import datetime
import statistics
DELIMITER = '","'  # Since the government likes microsoft, we have to use weird delimiters
DATA_DIRECTORY = "./data"

# Input files
FIRE_AREA_INPUT = f'{DATA_DIRECTORY}/Fire_Disturbance_Area.geojson'
FIRE_POINT_INPUT = f'{DATA_DIRECTORY}/fire_point.csv'
WEATHER_DATA_INPUT = f'{DATA_DIRECTORY}/Station Inventory EN.csv'

# Output files
FIRE_AREA_OUTPUT =f'{DATA_DIRECTORY}/processed_data/Fire_Disturbance_Area_Clean.csv'
FIRE_POINT_OUTPUT = f'{DATA_DIRECTORY}/processed_data/Fire_Disturbance_Point_Clean.csv'
WEATHER_STATION_OUT = f'{DATA_DIRECTORY}/processed_data/weather_station_data.csv'
WEATHER_STATION_LOCATIONS = f'./data/processed_data/weather_station_location_info.csv'

def assemble_fire_disturbance_point() -> pd.DataFrame:
    """
    Save a new csv containing only fire disturbance point data for fires which
    took place after the beginning of the year 1998

    The return value of this function is stored at:
    ./data/Fire_Disturbance_Point_Clean.csv
    """
    fires = pd.read_csv(FIRE_POINT_INPUT)
    fires = fires.rename(columns={'X': 'longitude', 'Y': 'latitude'})
    remove_indicies = []
    for row, item in fires.iterrows():
        print(row)
        if int(item['FIRE_START_DATE'].split('/')[0]) < 1998:
            remove_indicies.append(row)
    print('remove indicies')
    for row in remove_indicies:
        print(row)
        fires = fires.drop(row)
    return fires


def make_fire_disturbance_point() -> None:
    """
    Save the assembled fire disturbance point data to a csv.
    """
    print('Saving Data')
    data = assemble_fire_disturbance_point()
    data.to_csv(FIRE_POINT_OUTPUT)


def read_fire_disturbance_point() -> pd.DataFrame:
    """
    Read the csv containing fire disturbance point data, and return a dataframe
    representation of its content
    """
    fires = pd.read_csv(FIRE_POINT_OUTPUT)
    return fires


def assemble_fire_disturbance_area() -> pd.DataFrame:
    """
    Filter and build a dataset from the raw 'Fire_Disturbance_Area.geojson' file.

    This function does not take any parameters because it is expected to be rewritten directly.

    The function extracts the desired properties of area fires from the file, as well
    as caculates the approximate location of a given area fire by averaging it's shape.
    """
    # Open the file
    with open(FIRE_AREA_INPUT) as json_file:
        data = json.load(json_file)

        properties_so_far = []

        # Select data row-by-row
        for row in data['features']:
            if int(row['properties']['FIRE_START_DATE'].split('/')[0]) >= 1998:
                properties = [row['properties']['OGF_ID'],
                          row['properties']['FIRE_TYPE_CODE'],
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
                        'FIRE_TYPE_CODE',
                        'FIRE_START_DATE',
                        'FIRE_GENERAL_CAUSE_CODE',
                        'FIRE_FINAL_SIZE',
                        'LONGITUDE',
                        'LATITUDE']
        # Build the pandas dataframe from the extracted data, and the column labels
        return pd.DataFrame(properties_so_far, columns = columnLabels)


def make_fire_disturbance_area_csv() -> None:
    """
    Assemble and save fire disturbance area data to a csv (makes for faster loading in the
    future)
    """
    data = assemble_fire_disturbance_area()
    data.to_csv(FIRE_AREA_OUTPUT)


def read_fire_disturbance_area() -> pd.DataFrame:
    """
    Return a pandas dataframe containing the data read from the fire disturbance area csv
    """
    fires = pd.read_csv(FIRE_AREA_OUTPUT)
    return fires


#  Assemble the weather data
def filter_stations(min_year: int) -> pd.DataFrame:
    """
    Return the rows of all weather stations that operate throughout years min_year and 2020
    """
    unfiltered = pd.read_csv(WEATHER_DATA_INPUT, header=[2])

    filtered_first_year = unfiltered[unfiltered['First Year'] <= min_year]
    filtered_last_year = filtered_first_year[unfiltered['Last Year'] == 2020]
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
    Produces csv files for every eligable weather station operating throught 1998 to 2020.
    """
    stations = filter_stations(year) # Get eligable stations

    for _, station in stations.iterrows():
        print(station)
        data = assemble_weather_data_by_station(station['Station ID'],
                                                year, station['Last Year'])

        data.to_csv(f'{DATA_DIRECTORY}/weather_data/{station["Name"]}_{station["Station ID"]}.csv')


def make_weather_station_data() -> None:
    """
    Makes a giant csv file with all the weather station data.
    Requires the existance of individual station data in {DATA_DIRECTORY}/weather_data/ in
    the form of csv's.
    """
    data_frames = [pd.read_csv(f'{DATA_DIRECTORY}/weather_data/{file}')
                   for file in os.listdir(f'{DATA_DIRECTORY}/weather_data/') if '.csv' in file]

    pd.concat(data_frames).to_csv(WEATHER_STATION_OUT)


def read_weather_station_data() -> pd.DataFrame:
    """
    Read the weather station data from the csv, and return it as a pandas DataFrame
    """
    data = pd.read_csv(WEATHER_STATION_OUT)
    return data


def assemble_weather_station_locations() -> pd.DataFrame:
    """
    Return a dataframe of the location of weather stations, by name.
    """
    coordinates = []
    for file in os.listdir('./data/weather_data'):
        df = pd.read_csv('./data/weather_data/' + file)
        long = set(df['Longitude (x)'])
        lat = set(df['Latitude (y)'])        
        coordinates.append([file, list(lat)[0], list(long)[0]])
    final_coordinates = pd.DataFrame(coordinates, columns=['name', 'lat', 'lon'])
    return final_coordinates
    

def make_weather_station_locations() -> None:
    """
    Save the location of all weather stations, and their names, to a csv
    """
    data = assemble_weather_station_locations()
    data.to_csv(WEATHER_STATION_LOCATIONS)


def make_fire_point_with_weather():
    fire_point_data = read_fire_disturbance_point()
    fire_point_data = fire_point_data.iloc[:15]
    temperature_list = []
    precipitation_list = []
    stations_list = []
    for _, row in fire_point_data.iterrows():
        date = row['FIRE_START_DATE']
        lat = row['latitude']
        lon = row['longitude']
        temp, precip, station = get_closest_weather_data(date, lon, lat)
        temperature_list.append(temp)
        precipitation_list.append(precip)
        stations_list.append(station)    
    fire_point_data['WEATHER_STATION_PATH'] = stations_list
    fire_point_data['TEMPERATURE'] = temperature_list
    fire_point_data['PRECIPITATION'] = precipitation_list
    print(fire_point_data)
    

def get_closest_weather_data(date: str, lon: float, lat: float) -> List[str]:
    """
    Get the weather station data from the closest station to the given longitude and latitude
    coordinates
    """
    station_file_path = choose_closest_station(lon, lat)
    station = pd.read_csv('./data/weather_data/' + station_file_path)
    fire_date = date.split(' ')[0].split('/')
    fire_date = datetime.datetime(year=int(fire_date[0]), month=int(fire_date[1]), day=int(fire_date[2]))

    date_index = 0
    for row, data in station.iterrows():
        # print(int(data['Year']))
        # print(int(data['Month']))
        # print(int(data['Day']))
        current_date = datetime.datetime(year=int(data['Year']), month=int(data['Month']), day=int(data['Day']))
        
        if current_date > fire_date:
            date_index = row
            break
    weather_data = station.iloc[date_index - 21: date_index]    
    precipitation = list(weather_data['Total Precip (mm)'])
    temperature = list(weather_data['Max Temp (°C)'])


    temperature_str = ','.join([str(x) for x in temperature])
    precipitation_str = ','.join([str(x) for x in precipitation])

    return [temperature_str, precipitation_str, station_file_path]
        

def choose_closest_station(lon: float, lat: float) -> str:
    """
    Return the file path for the closest weather station to the given longitude and latitude 
    coordinates
    """
    possible_weather_stations = pd.read_csv('./data/processed_data/weather_station_location_info.csv')
    min_dist = [10000000000000000000, []]
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
    
    angle = 2 * math.asin(math.sqrt(math.sin(latitude_discrepancy / 2) ** 2 +
                          math.cos(math.radians(station[1])) * math.cos(math.radians(fire[1])) *
                          math.sin(longitude_discrepancy) ** 2))
    return angle * radius


def remove_nan_weather(weather: List[int]) -> List:
    """Remove nan values and replace them with the average value of the sequence"""
    non_nan = []
    for i in range(len(temperature)):
        if math.isnan(temperature[i]) == False:
            non_nan.append(temperature[i])
    if (len(non_nan) < 14):
        temperature == 'INVALID'
    else:
        avg_temp = statistics.mean(non_nan)
        for i in range(len(temperature)):
            if math.isnan(temperature[i]) == True:
                temperature[i] = avg_temp



if __name__ == '__main__':
    # print(answer)
    pass

