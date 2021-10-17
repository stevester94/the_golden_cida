#! /usr/bin/env python3

import unittest
import os
from typing import List
import re

import steves_utils.utils_v2 as steves_utils_v2

# A file pair being (sigmf-data, sigmf-meta)
NUMBER_OF_ORIGINAL_FILE_PAIRS = 352
NUMBER_OF_DEVICES = 16
ORIGINAL_PAPER_SAMPLES_PER_CHUNK = 128
NUM_SAMPLES_PER_ORIGINAL_FILE = 20006400

"""All distances in the dataset (This is distance from Tx to Rx)"""
ALL_DISTANCES_FEET = [
    2,
    8,
    14,
    20,
    26,
    32,
    38,
    44,
    50,
    56,
    62,
]

"""All the serial numbers of the USRP X310 transmitters in the dataset"""
ALL_SERIAL_NUMBERS = [
    "3123D52",
    "3123D65",
    "3123D79",
    "3123D80",
    "3123D54",
    "3123D70",
    "3123D7B",
    "3123D89",
    "3123D58",
    "3123D76",
    "3123D7D",
    "3123EFE",
    "3123D64",
    "3123D78",
    "3123D7E",
    "3124E4A",
]

"""All runs in the dataset"""
ALL_RUNS = [
    1,
    2,
]

SERIAL_NUMBER_MAPPING = {
    "3123D52": 0,
    "3123D65": 1,
    "3123D79": 2,
    "3123D80": 3,
    "3123D54": 4,
    "3123D70": 5,
    "3123D7B": 6,
    "3123D89": 7,
    "3123D58": 8,
    "3123D76": 9,
    "3123D7D": 10,
    "3123EFE": 11,
    "3123D64": 12,
    "3123D78": 13,
    "3123D7E": 14,
    "3124E4A": 15,
}


INVERSE_SERIAL_NUMBER_MAPPING = {v: k for k, v in SERIAL_NUMBER_MAPPING.items()}

def id_to_serial_number(serial_number_id:int)->str:
    return INVERSE_SERIAL_NUMBER_MAPPING[serial_number_id]

def serial_number_to_id(serial_number: str) -> int:
    return SERIAL_NUMBER_MAPPING[serial_number]



def get_oracle_dataset_path():
    return os.path.join(steves_utils_v2.get_datasets_base_path(), "KRI-16Devices-RawData")

def get_oracle_data_files_based_on_criteria(
        desired_serial_numbers,
        desired_runs,
        desired_distances,
):
    all_paths = []
    base_path = get_oracle_dataset_path()

    for distance in desired_distances:
        paths = steves_utils_v2.get_files_with_suffix_in_dir(os.path.join(base_path, str(distance)+"ft"), "sigmf-data")

        paths = filter_paths(
            paths=paths,
            distances_to_get=desired_distances,
            serial_numbers_to_get=desired_serial_numbers,
            runs_to_get=desired_runs
        )
        all_paths.extend(paths)

    return all_paths

def filter_paths(
        paths: List[str], 
        distances_to_get: List[int] = ALL_DISTANCES_FEET, 
        serial_numbers_to_get: List[str] =ALL_SERIAL_NUMBERS,
        runs_to_get: List[int] = ALL_RUNS):
    """Filters a list of paths to ORACLE dataset files based on desired attributes
    Args: 
        paths: list of paths of ORACLE dataset files that will be filtered
        distances_to_get: The distances to get, based on the filtering string 'xft'. Defaults to ALL_DISTANCES_FEET
        serial_numbers_to_get: The serial number to get, based on the filtering string 'x'. Defaults to ALL_SERIAL_NUMBERS
        runs_to_get: The runs to get, based on the filtering string 'runx'. Defaults to ALL_RUNS
        
    returns:
        The list of paths that match the filter criteria
    """
    assert(isinstance(distances_to_get, list))
    assert(isinstance(serial_numbers_to_get, list))
    assert(isinstance(runs_to_get, list))


    def is_any_pattern_in_string(list_of_patterns: List[str], string: str):
        """Helper function, returns a bool if any of the regex patterns in list_of_patterns is in the input string"""
        for p in list_of_patterns:
            if re.search(p, string) != None:
                return True
        return False

    # Trying to make this extensible
    all_filter_classes = [
        ["(?<![0-9]){}ft(?![0-9])".format(f) for f in distances_to_get],
        ["{}".format(f) for f in serial_numbers_to_get],
        ["run{}(?![0-9])".format(f) for f in runs_to_get],
    ]

    filtered_paths = paths

    for filter_class in all_filter_classes:
        filtered_paths = [p for p in filtered_paths if is_any_pattern_in_string(filter_class, p)]
        
    return filtered_paths


class Test_filter_paths(unittest.TestCase):
    def test_get_paths(self):
        paths = steves_utils_v2.get_files_with_suffix_in_dir(get_oracle_dataset_path(), "sigmf-data")
        self.assertEqual(352, len(paths))

    def test_get_everything(self):
        paths = steves_utils_v2.get_files_with_suffix_in_dir(get_oracle_dataset_path(), "sigmf-data")
        filtered_paths = filter_paths(
            paths,
        )
        self.assertEqual(
            len(filtered_paths),
            NUMBER_OF_ORIGINAL_FILE_PAIRS
        )

        self.assertEqual(
            paths, 
            filtered_paths
        )

    def test_get_one_run(self):
        paths = steves_utils_v2.get_files_with_suffix_in_dir(get_oracle_dataset_path(), "sigmf-data")
        filtered_paths = filter_paths(
            paths,
            runs_to_get=[1]
        )
        self.assertEqual(
            float(len(filtered_paths)),
            NUMBER_OF_ORIGINAL_FILE_PAIRS/2
        )

    def test_get_one_serial(self):
        paths = steves_utils_v2.get_files_with_suffix_in_dir(get_oracle_dataset_path(), "sigmf-data")
        filtered_paths = filter_paths(
            paths,
            serial_numbers_to_get=["3123D52"]
        )
        self.assertEqual(
            float(len(filtered_paths)),
            len(ALL_DISTANCES_FEET) * len(ALL_RUNS)
        )
    
    def test_get_one_distance(self):
        paths = steves_utils_v2.get_files_with_suffix_in_dir(get_oracle_dataset_path(), "sigmf-data")
        filtered_paths = filter_paths(
            paths,
            distances_to_get=[44]
        )
        self.assertEqual(
            float(len(filtered_paths)),
            len(ALL_SERIAL_NUMBERS) * len(ALL_RUNS)
        )

    def test_get_nonesense(self):
        paths = steves_utils_v2.get_files_with_suffix_in_dir(get_oracle_dataset_path(), "sigmf-data")

        filtered_paths = filter_paths(
            paths,
            distances_to_get=[1000]
        )
        self.assertEqual(
            0,
            len(filtered_paths)
        )

        filtered_paths = filter_paths(
            paths,
            runs_to_get=[1000]
        )
        self.assertEqual(
            0,
            len(filtered_paths)
        )

        filtered_paths = filter_paths(
            paths,
            serial_numbers_to_get=["1000"]
        )
        self.assertEqual(
            0,
            len(filtered_paths)
        )

    def test_get_just_one(self):
        paths = steves_utils_v2.get_files_with_suffix_in_dir(get_oracle_dataset_path(), "sigmf-data")

        filtered_paths = filter_paths(
            paths,
            distances_to_get=[44],
            serial_numbers_to_get=["3123D7B"],
            runs_to_get=[1]
        )
        self.assertEqual(
            1,
            len(filtered_paths)
        )