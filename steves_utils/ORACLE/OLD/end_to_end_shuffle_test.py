#! /usr/bin/env python3

from numpy.core.fromnumeric import sort
from steves_utils.ORACLE.simple_oracle_dataset_factory import Simple_ORACLE_Dataset_Factory
from steves_utils.ORACLE.utils import ALL_DISTANCES_FEET, ALL_RUNS, ALL_SERIAL_NUMBERS, get_chunk_of_IQ_based_on_metadata_and_index
from steves_utils.ORACLE.dataset_shuffler import Dataset_Shuffler
import tensorflow as tf
from typing import List
from steves_utils.ORACLE.utils import ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ALL_SERIAL_NUMBERS
from steves_utils.utils import get_all_in_dir
import unittest
from shutil import rmtree
import os
from steves_utils.ORACLE.shuffled_dataset_accessor import Shuffled_Dataset_Factory
import numpy as np
import copy

SCRATCH_DIR = "/mnt/wd500GB/derp/"



def clear_scrath_dir():
    for thing in get_all_in_dir(SCRATCH_DIR):
        rmtree(thing)

class Test_oracle_dataset_shuffler_safety_features(unittest.TestCase):
    @unittest.expectedFailure
    def test_too_few_piles(self):
        shuffler = Dataset_Shuffler(
            num_samples_per_chunk=ORIGINAL_PAPER_SAMPLES_PER_CHUNK,
            output_batch_size=1000,
            num_piles=1,
            output_format_str="shuffled_batchSize-{batch_size}_part-{part}.tfrecord_ds",
            output_max_file_size_MB=200,
            # output_max_file_size_MB=1,
            pile_dir=os.path.join(SCRATCH_DIR, "piles"),
            output_dir=os.path.join(SCRATCH_DIR, "output"),
            seed=1337,
        )

    @unittest.expectedFailure
    def test_non_empty_dirs(self):
        clear_scrath_dir()
        os.mkdir(os.path.join(SCRATCH_DIR, "piles"))
        with open(os.path.join(SCRATCH_DIR, "piles", "out"), "w") as f:
            f.write("lol")
        shuffler = Dataset_Shuffler(
            num_samples_per_chunk=ORIGINAL_PAPER_SAMPLES_PER_CHUNK,
            output_batch_size=1000,
            num_piles=1,
            output_format_str="shuffled_batchSize-{batch_size}_part-{part}.tfrecord_ds",
            output_max_file_size_MB=1,
            # output_max_file_size_MB=1,
            pile_dir=os.path.join(SCRATCH_DIR, "piles"),
            output_dir=os.path.join(SCRATCH_DIR, "output"),
            seed=1337,
            runs_to_get=[1],
            distances_to_get=[8],
            serial_numbers_to_get=[ALL_SERIAL_NUMBERS[0]]
        )

        shuffler.create_and_check_dirs()

    @unittest.expectedFailure
    def test_too_few_outputs(self):
        shuffler = Dataset_Shuffler(
            num_samples_per_chunk=ORIGINAL_PAPER_SAMPLES_PER_CHUNK,
            output_batch_size=1000,
            num_piles=1,
            output_format_str="shuffled_batchSize-{batch_size}_part-{part}.tfrecord_ds",
            output_max_file_size_MB=10000,
            pile_dir=os.path.join(SCRATCH_DIR, "piles"),
            output_dir=os.path.join(SCRATCH_DIR, "output"),
            seed=1337,
            runs_to_get=[1],
            distances_to_get=[8],
            serial_numbers_to_get=[ALL_SERIAL_NUMBERS[0]]
        )

    def test_too_few_outputs_no_fail_option(self):
        shuffler = Dataset_Shuffler(
            num_samples_per_chunk=ORIGINAL_PAPER_SAMPLES_PER_CHUNK,
            output_batch_size=1000,
            num_piles=1,
            output_format_str="shuffled_batchSize-{batch_size}_part-{part}.tfrecord_ds",
            output_max_file_size_MB=10000,
            pile_dir=os.path.join(SCRATCH_DIR, "piles"),
            output_dir=os.path.join(SCRATCH_DIR, "output"),
            seed=1337,
            runs_to_get=[1],
            distances_to_get=[8],
            serial_numbers_to_get=[ALL_SERIAL_NUMBERS[0]],
            fail_on_too_few_output_parts=False
        )
    

class Test_shuffler_end_to_end(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.pile_path = os.path.join(SCRATCH_DIR, "piles")
        self.output_path = os.path.join(SCRATCH_DIR, "output")

        self.runs_to_get = [1]
        self.distances_to_get = ALL_DISTANCES_FEET[:3]
        self.serial_numbers_to_get = ALL_SERIAL_NUMBERS[:3]

        self.train_val_test_splits=(0.6, 0.2, 0.2)

        self.shuffler = Dataset_Shuffler(
            num_samples_per_chunk=ORIGINAL_PAPER_SAMPLES_PER_CHUNK,
            output_batch_size=1000,
            num_piles=5,
            output_format_str="shuffled_batchSize-{batch_size}_part-{part}.tfrecord_ds",
            output_max_file_size_MB=5,
            pile_dir=self.pile_path,
            output_dir=self.output_path,
            seed=1337,
            runs_to_get=self.runs_to_get,
            distances_to_get=self.distances_to_get,
            serial_numbers_to_get=self.serial_numbers_to_get,
        )

        clear_scrath_dir()
        self.shuffler.create_and_check_dirs()
        print("Write piles")
        self.shuffler.write_piles()
        print("shuffle")
        self.shuffler.shuffle_piles()

        self.simple_ds, self.cardinality = Simple_ORACLE_Dataset_Factory(
            num_samples_per_chunk=ORIGINAL_PAPER_SAMPLES_PER_CHUNK,
            runs_to_get=self.runs_to_get,
            distances_to_get=self.distances_to_get,
            serial_numbers_to_get=self.serial_numbers_to_get,
        )

    # @unittest.skip("Skip cardinality to save time")
    def test_cardinality(self):
        # I believe because we are working with discrete files, some of them are being dropped. We allow up to 1% of data to be lost
        acceptable_cardinality_delta_percent = 0.01
        
        datasets = Shuffled_Dataset_Factory(self.output_path, train_val_test_splits=self.train_val_test_splits)

        train_ds = datasets["train_ds"]
        val_ds = datasets["val_ds"]
        test_ds = datasets["test_ds"]   

        train_count = 0
        for e in train_ds:
            train_count += e["index_in_file"].shape[0]

        val_count = 0
        for e in val_ds:
            val_count += e["index_in_file"].shape[0]

        test_count = 0
        for e in test_ds:
            test_count += e["index_in_file"].shape[0]

        expected_cardinality = self.cardinality
        expected_train_count  = expected_cardinality * self.train_val_test_splits[0]
        expected_val_count    = expected_cardinality * self.train_val_test_splits[1]
        expected_test_count   = expected_cardinality * self.train_val_test_splits[2]

        self.assertAlmostEqual(expected_cardinality, train_count+val_count+test_count, delta=expected_cardinality*acceptable_cardinality_delta_percent)
        self.assertAlmostEqual(train_count, expected_train_count, delta=expected_train_count*acceptable_cardinality_delta_percent)
        self.assertAlmostEqual(val_count, expected_val_count, delta=expected_val_count*acceptable_cardinality_delta_percent)
        self.assertAlmostEqual(test_count, expected_test_count, delta=expected_test_count*acceptable_cardinality_delta_percent)
    
    # @unittest.skip("Skip comparing chunks to save time")
    def test_compare_chunks_to_original(self):
        datasets = Shuffled_Dataset_Factory(self.output_path, train_val_test_splits=self.train_val_test_splits)

        train_ds = datasets["train_ds"]
        val_ds = datasets["val_ds"]
        test_ds = datasets["test_ds"] 

        all_ds = train_ds.concatenate(val_ds).concatenate(test_ds)

        for e in all_ds.unbatch():
            original_iq = get_chunk_of_IQ_based_on_metadata_and_index(
                serial_number_id=e["serial_number_id"].numpy(),
                distance_feet=e["distance_feet"].numpy(),
                run=e["run"].numpy(),
                index=e["index_in_file"].numpy(),
                num_samps_in_chunk=ORIGINAL_PAPER_SAMPLES_PER_CHUNK
            )
            
            self.assertTrue(
                np.array_equal(
                    e["IQ"].numpy(),
                    original_iq
                )
            )
    
    # @unittest.skip("Skip checking duplicates to save time")
    def test_for_duplicates(self):
        datasets = Shuffled_Dataset_Factory(self.output_path, train_val_test_splits=self.train_val_test_splits)

        train_ds = datasets["train_ds"]
        val_ds = datasets["val_ds"]
        test_ds = datasets["test_ds"] 

        all_ds = train_ds.concatenate(val_ds).concatenate(test_ds)

        train_hashes = []
        for e in train_ds.unbatch():
            train_hashes.append(
                hash((
                    int(e["serial_number_id"].numpy()),
                    int(e["distance_feet"].numpy()),
                    int(e["run"].numpy()),
                    int(e["index_in_file"].numpy()),
                ))
            )
        
        val_hashes = []
        for e in val_ds.unbatch():
            val_hashes.append(
                hash((
                    int(e["serial_number_id"].numpy()),
                    int(e["distance_feet"].numpy()),
                    int(e["run"].numpy()),
                    int(e["index_in_file"].numpy()),
                ))
            )

        test_hashes = []
        for e in test_ds.unbatch():
            test_hashes.append(
                hash((
                    int(e["serial_number_id"].numpy()),
                    int(e["distance_feet"].numpy()),
                    int(e["run"].numpy()),
                    int(e["index_in_file"].numpy()),
                ))
            )

        all_hashes = []
        for e in all_ds.unbatch():
            all_hashes.append(
                hash((
                    int(e["serial_number_id"].numpy()),
                    int(e["distance_feet"].numpy()),
                    int(e["run"].numpy()),
                    int(e["index_in_file"].numpy()),
                ))
            )

        self.assertTrue(
            len(all_hashes) == len(train_hashes+val_hashes+test_hashes)
        )

        self.assertTrue(
            len(train_hashes+val_hashes+test_hashes) == len(set(train_hashes+val_hashes+test_hashes))
        )

    def test_shuffling(self):
        """
        This one is a bit hard. How do you check for randomness?

        What I ended up doing is taking the 'index_in_file' metadata field, sorting it, and comparing it to the original.
        If they aren't the same then we should be good.
        """

        datasets = Shuffled_Dataset_Factory(self.output_path, train_val_test_splits=self.train_val_test_splits)

        train_ds = datasets["train_ds"]
        val_ds = datasets["val_ds"]
        test_ds = datasets["test_ds"]

        for ds in (train_ds, val_ds, test_ds):
            indices = []
            for e in ds:
                indices.extend(e["index_in_file"].numpy())
                        
            sorted_indices = copy.deepcopy(indices)
            sorted_indices.sort()

            self.assertFalse(
                np.array_equal(
                    indices,
                    sorted_indices
                )
            )



        

if __name__ == "__main__":
    unittest.main()



    # shuffler = Dataset_Shuffler(
    #     num_samples_per_chunk=ORIGINAL_PAPER_SAMPLES_PER_CHUNK,
    #     output_batch_size=1000,
    #     num_piles=5,
    #     output_format_str="shuffled_batchSize-{batch_size}_part-{part}.tfrecord_ds",
    #     output_max_file_size_MB=200,
    #     # output_max_file_size_MB=1,
    #     pile_dir="/mnt/wd500GB/derp/pile",
    #     output_dir="/mnt/wd500GB/derp/output",
    #     seed=1337,
    #     runs_to_get=[1],
    #     distances_to_get=[8],
    #     serial_numbers_to_get=[ALL_SERIAL_NUMBERS[0]]
    # )


    # shuffler.create_and_check_dirs()
    # print("Write piles")
    # shuffler.write_piles()
    # print("shuffle")
    # shuffler.shuffle_piles()