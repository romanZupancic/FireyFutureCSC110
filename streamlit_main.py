"""
The main file of our project. This calls all the code that is required for our app to function,
as long as the datasets have already been built.

The function calls in this file either: 
    - load stored csv data (into constants) by calling into assemble_data.py
    - further process stored data by calling into data_processing.py
    - call section visualizations from visuals.py

This file also preforms basic logging of streamlit progress.

Copyright and Usage Information
===============================
This file is Copyright (c) 2020 Daniel Hocevar and Roman Zupancic. 

This files contents may not be modified or redistributed without written
permission from Daniel Hocevar and Roman Zupancic
"""

# Third-party modules
import streamlit as st
import logging

# Custom modules
import assemble_data
import data_processing
import visuals

SECTIONS = ('Introduction', 
            'Preliminary Analysis', 
            'Fire Frequency', 
            'Weather and Fire Severity', 
            'The Models', 
            'Prediction Demo', 
            'Future Demo')

FIRE_POINT = assemble_data.read_fire_disturbance_point()
FIRE_AREA = assemble_data.read_fire_disturbance_area()

FIRE_POINT_PROCESSED = assemble_data.read_fire_disturbance_point_processed()
FIRE_AREA_PROCESSED = assemble_data.read_fire_disturbance_area_processed()

NO_FIRE_WEATHER_SEQUENCES = assemble_data.read_no_fire_weather_sequences()

PRESENT_WEATHER_DATA = assemble_data.read_modern_weather_data()


def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s: %(message)s')

    logging.info('---Start of file---')

    st.sidebar.title('Section selector')
    st.sidebar.write('''Please select which section you would like to view''')
    section_enabled = st.sidebar.radio('Section', SECTIONS)
    
    if section_enabled == 'Introduction':
        visuals.section_introduction()

    if section_enabled == 'Preliminary Analysis':
        logging.info('Presenting maps')
        st.title('Preliminary Data Analysis')
        visuals.subsection_fire_locations(FIRE_POINT, FIRE_AREA)

        logging.info('Calculating Fire Causes')
        visuals.subsection_fire_cause_counts(data_processing.fire_cause_count(FIRE_POINT),
                                            data_processing.fire_cause_count(FIRE_AREA))

        logging.info('Causes over time')
        visuals.subsection_causes_over_time(data_processing.CAUSE_REFERENCES, FIRE_POINT, FIRE_AREA)

    if section_enabled == 'Fire Frequency':
        logging.info('Fire frequency over time')
        visuals.section_frequency_over_time(FIRE_POINT_PROCESSED, FIRE_AREA_PROCESSED)

    if section_enabled == 'Weather and Fire Severity':
        logging.info('Calculating Weather and Fire severity')
        visuals.section_weather_fire_severity(data_processing.area_burned_vs_weather(FIRE_POINT_PROCESSED),
                                          data_processing.area_burned_vs_weather(FIRE_AREA_PROCESSED))
    if section_enabled == 'The Models':
        logging.info('Explaining the models')
        visuals.section_model_explanation(data_processing.graph_weather(FIRE_POINT_PROCESSED), 
                                        data_processing.graph_weather(NO_FIRE_WEATHER_SEQUENCES))

    if section_enabled == 'Prediction Demo':
        logging.info('Fire Predictions for 2019 vs Actual Results')
        visuals.section_predictions_vs_actual(PRESENT_WEATHER_DATA)

    if section_enabled == 'Future Demo':
        logging.info('Future Weather Predictions')
        visuals.section_2050_predictions(PRESENT_WEATHER_DATA)

    logging.info('---End of file---')


if __name__ == '__main__':
    main()