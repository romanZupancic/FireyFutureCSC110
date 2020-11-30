"""Build the data sets from a variety of sources"""

import requests
import pandas as pd
import json
import os

# 58526
DELIMITER = '","'  # Since the government likes microsoft, we have to use weird delimiters
DATA_DIRECTORY = "./data"

geojson = f'{DATA_DIRECTORY}/Fire_Disturbance_Area.json'

def assemble_fire_disturbance_point() -> pd.DataFrame:
    fires = pd.read_csv('fires_point.csv')
    fires = fires.rename(columns={'X': 'longitude', 'Y': 'latitude'})
    for row, item in fires.iterrows():
        print(row)
        if int(item['FIRE_START_DATE'].split('/')[0]) < 1998:
            fires.drop(row)
    return fires


def assemble_fire_disturbance_area() -> pd.DataFrame:
    with open(geojson) as json_file:
        data = json.load(json_file)
        
        properties_so_far = []
        
        for row in data['features']:
            if int(row['properties']['FIRE_START_DATE'].split('/')[0]) >= 1998:                
                properties = [row['properties']['OGF_ID'],
                          row['properties']['FIRE_TYPE_CODE'],
                          row['properties']['FIRE_START_DATE'],
                          row['properties']['FIRE_GENERAL_CAUSE_CODE'],
                          row['properties']['FIRE_FINAL_SIZE']
                          ]

                if row['geometry']['type'] == 'MultiPolygon':
                    collapsed = [coord for sub1 in row['geometry']['coordinates']
                                    for sub2 in sub1
                                    for coord in sub2]

                if row['geometry']['type'] == 'Polygon':
                    collapsed = [coord for sub1 in row['geometry']['coordinates']
                                    for coord in sub1]
                avg_long = sum([coord[0] for coord in collapsed]) / len(collapsed)
                avg_lat = sum([coord[1] for coord in collapsed]) / len(collapsed)
                properties.append(avg_long)   
                properties.append(avg_lat)
                #print(collapsed)
                
            
                properties_so_far.append(properties)

        columnLabels = ['OGF_ID', 
                        'FIRE_TYPE_CODE', 
                        'FIRE_START_DATE', 
                        'FIRE_GENERAL_CAUSE_CODE', 
                        'FIRE_FINAL_SIZE',
                        'LONGITUDE',
                        'LATITUDE']
        return pd.DataFrame(properties_so_far, columns = columnLabels)
    
def make_fire_disturbance_area_csv():
    data = assemble_fire_disturbance_area()
    data.to_csv(f'{DATA_DIRECTORY}/processed_data/fire_disturbance_point.csv')


#  Assemble the weather data
def filter_stations(min_year: int) -> pd.DataFrame:
    unfiltered = pd.read_csv(f'{DATA_DIRECTORY}/Station Inventory EN Full.csv', header=[2])

    filtered_first_year = unfiltered[unfiltered['First Year'] <= min_year]
    filtered_last_year = filtered_first_year[unfiltered['Last Year'] == 2020]
    filtered_province = filtered_last_year[filtered_last_year['Province'] == 'ONTARIO']
    return filtered_province


def assemble_weather_data_by_station(stationID: int, min_year: int, max_year: int) -> pd.DataFrame:
    print(f'min_year: {min_year}, max_year: {max_year}')
    data_frames = []

    for year in range(min_year, max_year + 1):
        print(year)
        data = requests.get(f'https://climate.weather.gc.ca/climate_data/bulk_data_e.html? \
                              format=csv&stationID={stationID}&Year={year}&Month=1&\
                              Day=14&timeframe=2&submit= Download+Data')

        fixed_data = data.text.replace('\r\n', '\n')
        fixed_data = fixed_data.splitlines()

        for row in range(len(fixed_data)):
            fixed_data[row] += DELIMITER * (len(fixed_data[0].split(DELIMITER)) -
                                            len(fixed_data[row].split(DELIMITER)))

        data_frames.append(pd.DataFrame([x.split(DELIMITER) for x in fixed_data[1:]],
                                        columns=fixed_data[0].split(DELIMITER)))

    return pd.concat(data_frames)


def make_relevant_weather_data(year: int = 1998) -> None:
    stations = filter_stations(year)

    for _, station in stations.iterrows():
        print(station)
        data = assemble_weather_data_by_station(station['Station ID'],
                                                year, station['Last Year'])

        data.to_csv(f'{DATA_DIRECTORY}/weather_data/{station["Name"]}_{station["Station ID"]}.csv')
        

fire_point_data = assemble_fire_disturbance_point()
fire_area_data = assemble_fire_disturbance_area()


def make_big_weather_data():
    data_frames = [pd.read_csv(f'{DATA_DIRECTORY}/weather_data/{file}') 
                   for file in os.listdir(f'{DATA_DIRECTORY}/weather_data/') if '.csv' in file]
    
    return pd.concat(data_frames)



if __name__ == '__main__':
    # print(answer)
    pass
    
