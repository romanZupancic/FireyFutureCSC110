"""Build the data sets from a variety of sources"""

import requests
import pandas as pd


DELIMITER = '","'  # Since the government likes microsoft, we have to use wired delimiters
DATA_DIRECTORY = "./data"


#  Assemble the weather data
def get_station_data(min_year: int) -> pd.DataFrame:
    unfiltered = pd.read_csv('./data/Station Inventory EN Full.csv', header=[2])

    filtered_first_year = unfiltered[unfiltered['First Year'] <= min_year]
    filtered_last_year = filtered_first_year[unfiltered['Last Year'] == 2020]
    filtered_province = filtered_last_year[filtered_last_year['Province'] == 'ONTARIO']
    return filtered_province


def get_weather_data_by_station(stationID: int, min_year: int, max_year: int):
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


def get_relevant_weather_data():
    stations = get_station_data(1990)

    for _, station in stations.iterrows():
        print(station)
        data = get_weather_data_by_station(station['Station ID'],
                                           station['First Year'], station['Last Year'])

        data.to_csv(f'./data/weather_data/{station["Station ID"]}.csv')


# Assemble the Fire Disturbance Area Data
def get_fire_data():
    pure_fire_area = pd.read_csv(f'{DATA_DIRECTORY}/Fire_Disturbance_Area.csv')
    ogf_ids = pd.read_csv(f'{DATA_DIRECTORY}/Geographic_Names_Ontario.csv')

    return pd.merge(pure_fire_area, ogf_ids, on='OGF_ID')


if __name__ == '__main__':
    answer = get_fire_data()
    print(answer)
    breakpoint()
