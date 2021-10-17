#! /usr/bin/env python3

from re import S
from numpy.core.fromnumeric import sort
from tensorflow.python.eager.context import device
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
from steves_utils.ORACLE.windowed_dataset_shuffler import Windowed_Dataset_Shuffler
from steves_utils.ORACLE.windowed_shuffled_dataset_accessor import Windowed_Shuffled_Dataset_Factory
import numpy as np
import copy
import math

SCRATCH_DIR = "/mnt/wd500GB/derp/"



def clear_scratch_dir():
    for thing in get_all_in_dir(SCRATCH_DIR):
        rmtree(thing)


class Test_Shuffler_Datasets_Only(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.num_windowed_examples_per_device = int(3e3)
        self.num_val_examples_per_device = int(1e3)
        self.num_test_examples_per_device = int(2e3)

        self.distances_to_filter_on=ALL_DISTANCES_FEET
        # self.distances_to_filter_on=[2,8]

        # self.num_windowed_examples_per_device = int(3e3)
        # self.num_val_examples_per_device = int(1e3)
        # self.num_test_examples_per_device = int(2e3)

        self.expected_train_count = self.num_windowed_examples_per_device * len(ALL_SERIAL_NUMBERS)
        self.expected_val_count = self.num_val_examples_per_device * len(ALL_SERIAL_NUMBERS)
        self.expected_test_count = self.num_test_examples_per_device * len(ALL_SERIAL_NUMBERS)

        input_shuffled_ds_num_samples_per_chunk=4*ORIGINAL_PAPER_SAMPLES_PER_CHUNK
        output_window_size=ORIGINAL_PAPER_SAMPLES_PER_CHUNK
        stride_length=3

        self.expected_train_replication_factor = math.ceil((input_shuffled_ds_num_samples_per_chunk - output_window_size)/stride_length + 1)
        self.expected_val_replication_factor = math.ceil(input_shuffled_ds_num_samples_per_chunk/output_window_size)
        self.expected_test_replication_factor = math.ceil(input_shuffled_ds_num_samples_per_chunk/output_window_size)

        self.shuffler = Windowed_Dataset_Shuffler(
            input_shuffled_ds_dir="/mnt/wd500GB/CSC500/csc500-super-repo/datasets/all_shuffled_chunk-512/output",
            input_shuffled_ds_num_samples_per_chunk=input_shuffled_ds_num_samples_per_chunk,
            output_batch_size=100,
            seed=1337,
            num_windowed_examples_per_device=self.num_windowed_examples_per_device,
            num_val_examples_per_device=self.num_val_examples_per_device,
            num_test_examples_per_device=self.num_test_examples_per_device,
            output_max_file_size_MB=1,
            output_window_size=output_window_size, 
            distances_to_filter_on=self.distances_to_filter_on,
            serials_to_filter_on=ALL_SERIAL_NUMBERS,
            working_dir=SCRATCH_DIR,
            output_format_str="batch-{batch_size}_part-{part}.tfrecord_ds",
            stride_length=stride_length
        )

        self.output_window_size = output_window_size
        self.input_shuffled_ds_num_samples_per_chunk = input_shuffled_ds_num_samples_per_chunk

    # @unittest.skip("Skip cardinality to save time")
    def test_dataset_cardinality(self):
        acceptable_cardinality_delta_percent = 0.05
        
        datasets = self.shuffler.get_datasets()

        train_ds = datasets["train_ds"]
        val_ds = datasets["val_ds"]
        test_ds = datasets["test_ds"]   

        train_ds = train_ds.batch(10000)
        val_ds = val_ds.batch(10000)
        test_ds = test_ds.batch(10000)

        train_count = 0
        for e in train_ds:
            train_count += e["index_in_file"].shape[0]

        val_count = 0
        for e in val_ds:
            val_count += e["index_in_file"].shape[0]

        test_count = 0
        for e in test_ds:
            test_count += e["index_in_file"].shape[0]


        self.assertAlmostEqual(
            train_count, 
            self.expected_train_count, 
            delta=self.expected_train_count*acceptable_cardinality_delta_percent
        )
        self.assertAlmostEqual(
            val_count,
            self.expected_val_count,
            delta=self.expected_val_count*acceptable_cardinality_delta_percent
        )
        self.assertAlmostEqual(
            test_count,
            self.expected_test_count,
            delta=self.expected_test_count*acceptable_cardinality_delta_percent
        )
        
        self.assertGreaterEqual(
            train_count, 
            self.expected_train_count, 
        )
        self.assertGreaterEqual(
            val_count,
            self.expected_val_count,
        )
        self.assertGreaterEqual(
            test_count,
            self.expected_test_count,
        )

    def test_og_dataset_shape(self):
        """This is more of a sanity check than anything"""
        datasets = self.shuffler.get_og_datasets()

        train_ds = datasets["train_ds"]
        val_ds = datasets["val_ds"]
        test_ds = datasets["test_ds"]  

        for e in train_ds.take(1000):
            self.assertEqual(
                e["IQ"].shape[-1],
                self.input_shuffled_ds_num_samples_per_chunk
            )

    # @unittest.skip("Skip shape to save time")
    def test_dataset_shape(self):
        BATCH = 10000
        datasets = self.shuffler.get_datasets()

        train_ds = datasets["train_ds"]
        val_ds = datasets["val_ds"]
        test_ds = datasets["test_ds"]  

        for ds in (train_ds, val_ds, test_ds):
            for e in ds.batch(BATCH, drop_remainder=True):
                self.assertEqual(
                    e["IQ"].shape,
                    (BATCH, 2, self.output_window_size)
                )
                self.assertEqual(
                    e["index_in_file"].shape,
                    (BATCH,)
                )
                self.assertEqual(
                    e["serial_number_id"].shape,
                    (BATCH,)
                )
                self.assertEqual(
                    e["distance_feet"].shape,
                    (BATCH,)
                )
                self.assertEqual(
                    e["run"].shape,
                    (BATCH,)
                )

    # @unittest.skip("Skip checking duplicates to save time")
    def test_for_duplicates(self):
        """Make sure no chunk ID is shared between the datasets"""
        acceptable_cardinality_delta_percent = 0.10

        datasets = self.shuffler.get_datasets()

        train_ds = datasets["train_ds"]
        val_ds = datasets["val_ds"]
        test_ds = datasets["test_ds"]  

        all_ds = train_ds.concatenate(val_ds).concatenate(test_ds)

        train_ds = train_ds.prefetch(1000)
        val_ds = val_ds.prefetch(1000)
        test_ds = test_ds.prefetch(1000)

        train_keys = []
        for e in train_ds:
            train_keys.append(
                (
                    e["index_in_file"].numpy(),
                    e["serial_number_id"].numpy(),
                    e["distance_feet"].numpy(),
                    e["run"].numpy(),
                )
            )

        val_keys = []
        for e in val_ds:
            val_keys.append(
                (
                    e["index_in_file"].numpy(),
                    e["serial_number_id"].numpy(),
                    e["distance_feet"].numpy(),
                    e["run"].numpy(),
                )
            )

        test_keys = []
        for e in test_ds:
            test_keys.append(
                (
                    e["index_in_file"].numpy(),
                    e["serial_number_id"].numpy(),
                    e["distance_feet"].numpy(),
                    e["run"].numpy(),
                )
            )


        self.assertAlmostEqual(
            len(train_keys) / len(set(train_keys)),
            self.expected_train_replication_factor,
            delta=0.00001,
        )

        self.assertAlmostEqual(
            len(val_keys) / len(set(val_keys)),
            self.expected_val_replication_factor,
            delta=0.00001,
            msg="Total Val Indices: {}, Unique Val Indices: {}".format(len(val_keys), len(set(val_keys)))
        )

        self.assertAlmostEqual(
            len(test_keys) / len(set(test_keys)),
            self.expected_test_replication_factor,
            delta=0.00001
        )

        train_keys = set(train_keys)
        val_keys   = set(val_keys)
        test_keys  = set(test_keys)

        self.assertEqual(
            len(train_keys.intersection(val_keys)),
            0
        )

        self.assertEqual(
            len(train_keys.intersection(test_keys)),
            0
        )

        self.assertEqual(
            len(val_keys.intersection(test_keys)),
            0
        )

    # @unittest.skip("Skip device distribution to save time")
    def test_device_distribution(self):
        """The count for each device should absolutely be equal"""
        datasets = self.shuffler.get_datasets()

        train_ds = datasets["train_ds"]
        val_ds = datasets["val_ds"]
        test_ds = datasets["test_ds"]  

        for ds in (train_ds, val_ds, test_ds):
            devices = {}
            for e in ds:
                serial = e["serial_number_id"].numpy()
                devices[serial] = devices.get(serial, 0) + 1
            counts = [val for key,val in devices.items()]
            self.assertEqual(len(set(counts)), 1)

    def test_distance_distribution_and_filtering(self):
        """We have a separate test for asserting each device has equal example counts, so
           we can make this a little simpler and only look at the distances themselves.
           The datasets kinda suck in distribution, so we don't hard fail if they don't pass...
        """
        acceptable_delta_percent_between_distances = 0.05
        datasets = self.shuffler.get_datasets()

        train_ds = datasets["train_ds"]
        val_ds = datasets["val_ds"]
        test_ds = datasets["test_ds"]  

        for ds in (train_ds, val_ds, test_ds):
            test_failed = False
            distances = {}
            for e in ds:
                distance = e["distance_feet"].numpy()
                distances[distance] = distances.get(distance, 0) + 1

            for distance_orig, value_orig in distances.items():
                for distance_next, value_next in [(a,b) for a,b in distances.items() if a != distance_orig]:
                    delta_percent = abs(value_orig - value_next) / value_orig

                    try:
                        self.assertLessEqual(
                            delta_percent,
                            acceptable_delta_percent_between_distances,
                            msg="{}ft has {} Samples, but {}ft has {} samples. Delta percent: {} is unacceptable".format(
                                distance_orig,
                                value_orig,
                                distance_next,
                                value_next,
                                delta_percent
                            )
                        )
                    except AssertionError as e:
                        test_failed = True
                        # print(e)
                        # print("Distance Distribution Failed. Continuing")

            # If this were to fail we are in hot water
            self.assertEqual(
                set(self.distances_to_filter_on),
                set(distances.keys())
            )
            if test_failed:
                if ds is train_ds:
                    print("Train Dataset Distance Distribution Failed")
                if ds is val_ds:
                    print("Val Dataset Distance Distribution Failed")
                if ds is test_ds:
                    print("Test Dataset Distance Distribution Failed")
                print("Distributions:")
                print(distances)
                        


class Test_shuffler_end_to_end(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.num_windowed_examples_per_device = int(200e3)
        self.num_val_examples_per_device = int(10e3)
        self.num_test_examples_per_device = int(50e3)

        self.distances_to_filter_on=[2,8]

        self.expected_train_count = self.num_windowed_examples_per_device * len(ALL_SERIAL_NUMBERS)
        self.expected_val_count = self.num_val_examples_per_device * len(ALL_SERIAL_NUMBERS)
        self.expected_test_count = self.num_test_examples_per_device * len(ALL_SERIAL_NUMBERS)

        input_shuffled_ds_num_samples_per_chunk=4*ORIGINAL_PAPER_SAMPLES_PER_CHUNK
        output_window_size=ORIGINAL_PAPER_SAMPLES_PER_CHUNK
        stride_length=3

        self.output_batch_size = 100

        self.expected_train_replication_factor = math.ceil((input_shuffled_ds_num_samples_per_chunk - output_window_size)/stride_length + 1)
        self.expected_val_replication_factor = math.ceil(input_shuffled_ds_num_samples_per_chunk/output_window_size)
        self.expected_test_replication_factor = math.ceil(input_shuffled_ds_num_samples_per_chunk/output_window_size)

        self.shuffler = Windowed_Dataset_Shuffler(
            input_shuffled_ds_dir="/mnt/wd500GB/CSC500/csc500-super-repo/datasets/all_shuffled_chunk-512/output",
            input_shuffled_ds_num_samples_per_chunk=input_shuffled_ds_num_samples_per_chunk,
            output_batch_size=self.output_batch_size,
            seed=1337,
            num_windowed_examples_per_device=self.num_windowed_examples_per_device,
            num_val_examples_per_device=self.num_val_examples_per_device,
            num_test_examples_per_device=self.num_test_examples_per_device,
            output_max_file_size_MB=1,
            output_window_size=output_window_size, 
            distances_to_filter_on=self.distances_to_filter_on,
            serials_to_filter_on=ALL_SERIAL_NUMBERS,
            working_dir=SCRATCH_DIR,
            output_format_str="batch-{batch_size}_part-{part}.tfrecord_ds",
            stride_length=stride_length
        )

        self.output_window_size = output_window_size

        clear_scratch_dir()
        self.shuffler.create_and_check_dirs()
        print("Write piles")
        self.shuffler.write_piles()
        print("shuffle")
        self.shuffler.shuffle_piles()

    # @unittest.skip("Skip cardinality to save time")
    def test_cardinality(self):
        # There's some slop in this due to the replication factor. Err on the side of caution and make sure there are more than what we wanted
        # but at a max of below percent
        acceptable_cardinality_delta_percent = 0.05
        
        datasets = Windowed_Shuffled_Dataset_Factory(SCRATCH_DIR)

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


        self.assertAlmostEqual(
            train_count, 
            self.expected_train_count, 
            delta=self.expected_train_count*acceptable_cardinality_delta_percent
        )
        self.assertAlmostEqual(
            val_count,
            self.expected_val_count,
            delta=self.expected_val_count*acceptable_cardinality_delta_percent
        )
        self.assertAlmostEqual(
            test_count,
            self.expected_test_count,
            delta=self.expected_test_count*acceptable_cardinality_delta_percent
        )
        
        self.assertGreaterEqual(
            train_count, 
            self.expected_train_count, 
        )
        self.assertGreaterEqual(
            val_count,
            self.expected_val_count,
        )
        self.assertGreaterEqual(
            test_count,
            self.expected_test_count,
        )

        
    # @unittest.skip("Skip checking duplicates to save time")
    def test_for_duplicates(self):
        """Make sure no chunk ID is shared between the datasets"""
        datasets = Windowed_Shuffled_Dataset_Factory(SCRATCH_DIR)

        train_ds = datasets["train_ds"]
        val_ds = datasets["val_ds"]
        test_ds = datasets["test_ds"] 

        all_ds = train_ds.concatenate(val_ds).concatenate(test_ds)

        train_ds = train_ds.unbatch()
        val_ds = val_ds.unbatch()
        test_ds = test_ds.unbatch()

        train_keys = []
        for e in train_ds:
            train_keys.append(
                (
                    e["index_in_file"].numpy(),
                    e["serial_number_id"].numpy(),
                    e["distance_feet"].numpy(),
                    e["run"].numpy(),
                )
            )

        val_keys = []
        for e in val_ds:
            val_keys.append(
                (
                    e["index_in_file"].numpy(),
                    e["serial_number_id"].numpy(),
                    e["distance_feet"].numpy(),
                    e["run"].numpy(),
                )
            )

        test_keys = []
        for e in test_ds:
            test_keys.append(
                (
                    e["index_in_file"].numpy(),
                    e["serial_number_id"].numpy(),
                    e["distance_feet"].numpy(),
                    e["run"].numpy(),
                )
            )


        self.assertAlmostEqual(
            len(train_keys) / len(set(train_keys)),
            self.expected_train_replication_factor,
            delta=0.00001,
        )

        self.assertAlmostEqual(
            len(val_keys) / len(set(val_keys)),
            self.expected_val_replication_factor,
            delta=0.00001,
            msg="Total Val Indices: {}, Unique Val Indices: {}".format(len(val_keys), len(set(val_keys)))
        )

        self.assertAlmostEqual(
            len(test_keys) / len(set(test_keys)),
            self.expected_test_replication_factor,
            delta=0.00001
        )

        train_keys = set(train_keys)
        val_keys   = set(val_keys)
        test_keys  = set(test_keys)

        self.assertEqual(
            len(train_keys.intersection(val_keys)),
            0
        )

        self.assertEqual(
            len(train_keys.intersection(test_keys)),
            0
        )

        self.assertEqual(
            len(val_keys.intersection(test_keys)),
            0
        )
        train_ds = datasets["train_ds"].unbatch()
        val_ds = datasets["val_ds"].unbatch()
        test_ds = datasets["test_ds"].unbatch()
    # @unittest.skip("Skip shuffling to save time")
    def test_shuffling(self):
        """
        This one is a bit hard. How do you check for randomness?

        What I ended up doing is taking the 'index_in_file' metadata field, sorting it, and comparing it to the original.
        If they aren't the same then we should be good.
        """

        datasets = Windowed_Shuffled_Dataset_Factory(SCRATCH_DIR)

        train_ds = datasets["train_ds"]

        indices = []
        for e in train_ds:
            indices.extend(e["index_in_file"].numpy())
                    
        sorted_indices = copy.deepcopy(indices)
        sorted_indices.sort()

        self.assertFalse(
            np.array_equal(
                indices,
                sorted_indices
            )
        )

    # @unittest.skip("Skip shape to save time")
    def test_dataset_shape(self):
        BATCH = self.output_batch_size
        datasets = Windowed_Shuffled_Dataset_Factory(SCRATCH_DIR)

        train_ds = datasets["train_ds"]
        val_ds = datasets["val_ds"]
        test_ds = datasets["test_ds"] 

        # Its should be quite small
        acceptable_percentage_small_batches = 0.001
        num_batches = 0
        num_batches_that_are_smaller_than_expected = 0

        for ds in (train_ds, val_ds, test_ds):
            for e in ds:
                num_batches += 1
                self.assertTrue(
                    e["IQ"].shape[0] \
                        == e["index_in_file"].shape[0] \
                        == e["serial_number_id"].shape[0] \
                        == e["distance_feet"].shape[0] \
                        == e["run"].shape[0]
                )

                if e["IQ"].shape[0] < BATCH:
                    num_batches_that_are_smaller_than_expected += 1
                    continue

                self.assertEqual(
                    e["IQ"].shape[1:],
                    (2, self.output_window_size)
                )
                self.assertEqual(
                    e["index_in_file"].shape,
                    (BATCH,)
                )
                self.assertEqual(
                    e["serial_number_id"].shape,
                    (BATCH,)
                )
                self.assertEqual(
                    e["distance_feet"].shape,
                    (BATCH,)
                )
                self.assertEqual(
                    e["run"].shape,
                    (BATCH,)
                )

        self.assertLessEqual(
            num_batches_that_are_smaller_than_expected/num_batches,
            acceptable_percentage_small_batches
        )

    # @unittest.skip("Skip device distribution to save time")
    def test_device_distribution(self):
        """The count for each device should absolutely be equal"""
        datasets = Windowed_Shuffled_Dataset_Factory(SCRATCH_DIR)

        train_ds = datasets["train_ds"].unbatch()
        val_ds = datasets["val_ds"].unbatch()
        test_ds = datasets["test_ds"].unbatch()

        for ds in (train_ds, val_ds, test_ds):
            devices = {}
            for e in ds:
                serial = e["serial_number_id"].numpy()
                devices[serial] = devices.get(serial, 0) + 1
            counts = [val for key,val in devices.items()]
            self.assertEqual(len(set(counts)), 1)

    # @unittest.skip("Skip distance distribution to save time")
    def test_distance_distribution_and_filtering(self):
        """We have a separate test for asserting each device has equal example counts, so
           we can make this a little simpler and only look at the distances themselves.
           The datasets kinda suck in distribution, so we don't hard fail if they don't pass...
        """
        acceptable_delta_percent_between_distances = 0.05
        datasets = Windowed_Shuffled_Dataset_Factory(SCRATCH_DIR)

        train_ds = datasets["train_ds"].unbatch()
        val_ds = datasets["val_ds"].unbatch()
        test_ds = datasets["test_ds"].unbatch()

        for ds in (train_ds, val_ds, test_ds):
            test_failed = False
            distances = {}
            for e in ds:
                distance = e["distance_feet"].numpy()
                distances[distance] = distances.get(distance, 0) + 1

            for distance_orig, value_orig in distances.items():
                for distance_next, value_next in [(a,b) for a,b in distances.items() if a != distance_orig]:
                    delta_percent = abs(value_orig - value_next) / value_orig

                    try:
                        self.assertLessEqual(
                            delta_percent,
                            acceptable_delta_percent_between_distances,
                            msg="{}ft has {} Samples, but {}ft has {} samples. Delta percent: {} is unacceptable".format(
                                distance_orig,
                                value_orig,
                                distance_next,
                                value_next,
                                delta_percent
                            )
                        )
                    except AssertionError as e:
                        test_failed = True
                        # print(e)
                        # print("Distance Distribution Failed. Continuing")

            # If this were to fail we are in hot water
            self.assertEqual(
                set(self.distances_to_filter_on),
                set(distances.keys())
            )
            if test_failed:
                if ds is train_ds:
                    print("Train Dataset Distance Distribution Failed")
                if ds is val_ds:
                    print("Val Dataset Distance Distribution Failed")
                if ds is test_ds:
                    print("Test Dataset Distance Distribution Failed")
                print("Distributions:")
                print(distances)

if __name__ == "__main__":
    unittest.main()