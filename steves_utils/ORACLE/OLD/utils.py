#! /usr/bin/env python3

import re
from typing import List

from tensorflow.python.framework import dtypes
import steves_utils.utils
import sys
import os
import unittest
import numpy as np
import tensorflow as tf

"""All distances in the dataset (This is distance from Tx to Rx)"""
ALL_DISTANCES_FEET = [
    14,
    2,
    44,
    62,
    20,
    32,
    50,
    8,
    26,
    38,
    56,
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

# A file pair being (sigmf-data, sigmf-meta)
NUMBER_OF_ORIGINAL_FILE_PAIRS = 352
NUMBER_OF_DEVICES = 16
ORIGINAL_PAPER_SAMPLES_PER_CHUNK = 128
NUM_SAMPLES_PER_ORIGINAL_FILE = 20006400

def id_to_serial_number(serial_number_id:int)->str:
    return INVERSE_SERIAL_NUMBER_MAPPING[serial_number_id]

def serial_number_to_id(serial_number: str) -> int:
    return SERIAL_NUMBER_MAPPING[serial_number]

def get_oracle_dataset_path():
    return os.path.join(steves_utils.utils.get_datasets_base_path(), "KRI-16Devices-RawData")

def metadata_to_file_name(serial_number_id:int, distance_feet:int, run:int)->str:
    return "WiFi_air_X310_{serial_number}_{distance_feet}ft_run{run}".format(
        serial_number=id_to_serial_number(serial_number_id),
        distance_feet=distance_feet,
        run=run
    )

def metadata_from_path(path: str):
    match  = re.search("WiFi_air_X310_(.*)_([0-9]+)ft_run([0-9]+)", path)

    if match == None:
        raise Exception("Failed to parse metadata from path {}".format(path))

    (serial_number, distance_feet, run) = match.groups()

    return {
        "serial_number": serial_number,
        "distance_feet": int(distance_feet),
        "run": int(run)
    }

class Test_metadata_helpers(unittest.TestCase):
    def test_simple(self):
        path = "WiFi_air_X310_3123D64_44ft_run2.sigmf-meta"
        m = metadata_from_path(path)

        should_be = {
            "serial_number": "3123D64",
            "distance_feet": 44,
            "run": 2,
        }

        self.assertEqual(
            should_be,
            m
        )

    def test_all(self):
        """Actually a pretty good overall dataset integrity test"""
        import itertools

        paths = steves_utils.utils.get_files_with_suffix_in_dir(get_oracle_dataset_path(), "sigmf-data")
        all_meta = [metadata_from_path(p) for p in paths]
        all_meta = [(m["serial_number"], m["distance_feet"], m["run"]) for m in all_meta]

        self.assertEqual(len(all_meta), len(set(all_meta)))

        self.assertEqual(
            set(all_meta),
            set(itertools.product(ALL_SERIAL_NUMBERS, ALL_DISTANCES_FEET, ALL_RUNS))
        )

    def test_symmetry(self):
        path = "WiFi_air_X310_3123D64_44ft_run2"
        m = metadata_from_path(path)
        m_1 = metadata_to_file_name(
            serial_number_to_id(m["serial_number"]),
            m["distance_feet"],
            m["run"]
        )

        self.assertEqual(path, m_1)


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
        paths = steves_utils.utils.get_files_with_suffix_in_dir(get_oracle_dataset_path(), "sigmf-data")
        self.assertEqual(352, len(paths))

    def test_get_everything(self):
        paths = steves_utils.utils.get_files_with_suffix_in_dir(get_oracle_dataset_path(), "sigmf-data")
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
        paths = steves_utils.utils.get_files_with_suffix_in_dir(get_oracle_dataset_path(), "sigmf-data")
        filtered_paths = filter_paths(
            paths,
            runs_to_get=[1]
        )
        self.assertEqual(
            float(len(filtered_paths)),
            NUMBER_OF_ORIGINAL_FILE_PAIRS/2
        )

    def test_get_one_serial(self):
        paths = steves_utils.utils.get_files_with_suffix_in_dir(get_oracle_dataset_path(), "sigmf-data")
        filtered_paths = filter_paths(
            paths,
            serial_numbers_to_get=["3123D52"]
        )
        self.assertEqual(
            float(len(filtered_paths)),
            len(ALL_DISTANCES_FEET) * len(ALL_RUNS)
        )
    
    def test_get_one_distance(self):
        paths = steves_utils.utils.get_files_with_suffix_in_dir(get_oracle_dataset_path(), "sigmf-data")
        filtered_paths = filter_paths(
            paths,
            distances_to_get=[44]
        )
        self.assertEqual(
            float(len(filtered_paths)),
            len(ALL_SERIAL_NUMBERS) * len(ALL_RUNS)
        )

    def test_get_nonesense(self):
        paths = steves_utils.utils.get_files_with_suffix_in_dir(get_oracle_dataset_path(), "sigmf-data")

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
        paths = steves_utils.utils.get_files_with_suffix_in_dir(get_oracle_dataset_path(), "sigmf-data")

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

def get_chunk_of_IQ_based_on_metadata_and_index(
    serial_number_id:int, distance_feet:int, run:int,
    index:int, num_samps_in_chunk:int
):
    name = metadata_to_file_name(serial_number_id, distance_feet, run) + ".sigmf-data"
    path = os.path.join(get_oracle_dataset_path(), name)

    return steves_utils.utils.get_chunk_of_IQ_at_index_from_binary_file(
        path=path,
        index=index,
        num_samps_in_chunk=num_samps_in_chunk,
        I_or_Q_datatype=np.float64
    )

class Test_get_chunk_of_IQ_based_on_metadata_and_index(unittest.TestCase):
    def test_limited(self):
        print("Begin Test_get_chunk_of_IQ_based_on_metadata_and_index")
        path = get_oracle_dataset_path() + "/WiFi_air_X310_3123D64_44ft_run2.sigmf-data"
        ds,_ = binary_file_path_to_oracle_dataset(path, ORIGINAL_PAPER_SAMPLES_PER_CHUNK)
        ds = ds.shuffle(100000000).prefetch(1000)

        for e in ds:
            ds_iq = e["IQ"]
            fetched_iq = get_chunk_of_IQ_based_on_metadata_and_index(
                e["serial_number_id"].numpy(),
                e["distance_feet"].numpy(),
                e["run"].numpy(),
                e["index_in_file"].numpy(),
                ORIGINAL_PAPER_SAMPLES_PER_CHUNK
            )

            self.assertTrue(
                np.array_equal(
                    ds_iq.numpy(),
                    fetched_iq
                )
            )


def binary_file_path_to_oracle_dataset(
    path: str,
    num_samples_per_chunk: int
):
    metadata = metadata_from_path(path)
    ds, cardinality = steves_utils.utils.interleaved_IQ_binary_file_to_dataset(
        path,
        num_samples_per_chunk,
        tf.float64,
    )

    ds = ds.map(
        lambda index, IQ:
            {
                "IQ": IQ,
                "index_in_file": index,
                "serial_number_id": tf.constant(serial_number_to_id(metadata["serial_number"]), dtype=tf.uint8),
                "distance_feet": tf.constant(metadata["distance_feet"], dtype=tf.uint8),
                "run": tf.constant(metadata["run"], dtype=tf.uint8),
            }
    )

    return ds, cardinality

class Test_binary_file_path_to_oracle_dataset(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

        self.path = get_oracle_dataset_path() + "/WiFi_air_X310_3123D64_44ft_run2.sigmf-data"
        self.num_samples_per_chunk = 128
        self.expected_index = 0
        self.expected_serial = serial_number_to_id("3123D64")
        self.expected_distance = 44 
        self.expected_run = 2

    def test_metadata(self):
        ds, _ = binary_file_path_to_oracle_dataset(
            self.path,
            self.num_samples_per_chunk
        )

        first = next(ds.as_numpy_iterator())

        self.assertEqual(
            first["index_in_file"],
            self.expected_index
        )
        self.assertEqual(
            first["serial_number_id"],
            self.expected_serial
        )
        self.assertEqual(
            first["distance_feet"],
            self.expected_distance
        )
        self.assertEqual(
            first["run"],
            self.expected_run
        )

    def test_data_integrity(self):
        ds, _ = binary_file_path_to_oracle_dataset(
            self.path,
            self.num_samples_per_chunk
        )

        chunk_size = self.num_samples_per_chunk * 2 * 8 # 2 64bit floating points per sample

        with open(self.path, "rb") as f:
            for e in ds.take(100):
                buf = f.read(chunk_size)
                original_element = np.frombuffer(buf, dtype=np.complex128)

                for idx, X in enumerate(original_element):
                    self.assertEqual(
                        e["IQ"][0][idx],
                        X.real
                    )

                    self.assertEqual(
                        e["IQ"][1][idx],
                        X.imag
                    )

            # self.assertEqual(
            #     steves_utils.utils.get_file_size(self.path),
            #     f.tell()
            # )
    
    def test_cardinality(self):
        ds, cardinality = binary_file_path_to_oracle_dataset(
            self.path,
            self.num_samples_per_chunk
        )

        count = 0
        for e in ds.prefetch(10000):
            count += 1

        self.assertEqual(count, cardinality)



if __name__ == '__main__':
    unittest.main()

    ds = binary_file_path_to_oracle_dataset(
        get_oracle_dataset_path() + "/WiFi_air_X310_3123D64_44ft_run2.sigmf-data",
        128
    )

    for e in ds.take(1):
        print(e)

# if __name__ == "__main__":
#     import doctest
#     doctest.testmod()
#     p = steves_utils.utils.get_files_with_suffix_in_dir(get_oracle_dataset_path(), "sigmf-data")
#     print(len(filter_paths(p)))