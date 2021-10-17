#! /usr/bin/env python3

from tensorflow._api.v2 import data
from typing import List
import unittest

import steves_utils.utils
from steves_utils.ORACLE.utils import (
    get_oracle_dataset_path, 
    filter_paths,
    binary_file_path_to_oracle_dataset,
    ALL_DISTANCES_FEET,
    ALL_SERIAL_NUMBERS, 
    ALL_RUNS, 
    NUM_SAMPLES_PER_ORIGINAL_FILE, 
    NUMBER_OF_ORIGINAL_FILE_PAIRS, 
    NUM_SAMPLES_PER_ORIGINAL_FILE, 
    ORIGINAL_PAPER_SAMPLES_PER_CHUNK,
    serial_number_to_id
)


def Simple_ORACLE_Dataset_Factory(
    num_samples_per_chunk: int,
    base_path:str = get_oracle_dataset_path(),
    distances_to_get: List[int] = ALL_DISTANCES_FEET, 
    serial_numbers_to_get: List[str] =ALL_SERIAL_NUMBERS,
    runs_to_get: List[int] = ALL_RUNS):

    """Ok fine it's not a factory. Returns a consolidated dataset based on the criteria in the args
    
    This function is intended to build a simple dataset based on provided criteria. It does not do
    any fancy interleaving or shuffling. This is intended to be used on small subsets of the ORACLE datasets
    which can be fit in memory, or as a building block for more complex dataset pipelines.

    NOTE: The one tricky thing to this is that the serial numbers are encoded as ints according to 
          SERIAL_NUMBER_MAPPING

    TIP: Use                 lambda IQ,index,serial_number,distance_feet,run: 

    Returns:
        A tensorflow dataset
    """

    paths = steves_utils.utils.get_files_with_suffix_in_dir(base_path, "sigmf-data")

    paths = filter_paths(
        paths,
        distances_to_get = distances_to_get,
        serial_numbers_to_get = serial_numbers_to_get,
        runs_to_get = runs_to_get,
    )

    if len(paths) == 0:
        raise Exception("0 binaries matched requested criteria")

    datasets = [binary_file_path_to_oracle_dataset(p, num_samples_per_chunk) for p in paths]

    total_cardinality = sum([d[1] for d in datasets])

    ds = datasets[0][0]
    for d in datasets[1:]:
        ds = ds.concatenate(d[0])

    return ds, total_cardinality

class Test_Simple_ORACLE_Dataset_Factory(unittest.TestCase):
    def test_cardinality(self):
        ds, cardinality = Simple_ORACLE_Dataset_Factory(ORIGINAL_PAPER_SAMPLES_PER_CHUNK)


        self.assertEqual(
            NUMBER_OF_ORIGINAL_FILE_PAIRS * NUM_SAMPLES_PER_ORIGINAL_FILE / ORIGINAL_PAPER_SAMPLES_PER_CHUNK,
            cardinality
        )

if __name__ == "__main__":
    unittest.main()
