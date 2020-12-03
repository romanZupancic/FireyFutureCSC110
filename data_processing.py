"""Helper functions to preform data processing/manipulation/rearranging."""
import pandas as pd
import assemble_data

CAUSE_REFERENCES = {'IDF': 'Foresting', 'IDO': 'Industrial',
                    'INC': 'Incedniary', 'LTG': 'Lightning',
                    'MIS': 'Miscellaneous', 'REC': 'Recreation',
                    'RES': 'Resident', 'RWY': 'Railway', 'UNK': 'Unkown'}


def fire_point_cause_count(fire_point_data: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of times each type of fire cause appears in the dataset

    Preconditions:
        - fire_point_data is a pd.DataFrame that follows the expected format of the Fire Disturbance 
        area dataset
    """
    cause_series = fire_point_data['FIRE_GENERAL_CAUSE_CODE']
    cause_dict = {}
    for _, fire_cause in cause_series.items():
        if CAUSE_REFERENCES[fire_cause] in cause_dict:
            cause_dict[CAUSE_REFERENCES[fire_cause]][0] += 1
        else:
            cause_dict[CAUSE_REFERENCES[fire_cause]] = [1]
    cause_df = pd.DataFrame(cause_dict).transpose()
    print(cause_df.columns)
    return cause_df


def fire_area_cause_count(fire_area_data: pd.DataFrame) -> pd.DataFrame:
    """Count the number of times each type of fire cause appears in the dataset
    
    Preconditions:
        - fire_area_data is a pd.DataFrame that follows the expected format of the Fire Disturbance 
        area dataset
    """
    cause_series = fire_area_data['FIRE_GENERAL_CAUSE_CODE']
    appearances_so_far = {}

    for _, fire_cause in cause_series.items():
        if CAUSE_REFERENCES[fire_cause] in appearances_so_far:
            appearances_so_far[CAUSE_REFERENCES[fire_cause]][0] += 1
        else:
            appearances_so_far[CAUSE_REFERENCES[fire_cause]] = [1]
            
    return pd.DataFrame(appearances_so_far).transpose()
