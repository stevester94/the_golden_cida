#! /usr/bin/env python3

from numpy.lib import stride_tricks
from tensorflow.python.framework import dtypes
import steves_utils.ORACLE.serialization as oracle_serialization
from steves_utils.ORACLE.simple_oracle_dataset_factory import Simple_ORACLE_Dataset_Factory
from steves_utils.ORACLE.utils import ALL_DISTANCES_FEET, ALL_RUNS, ALL_SERIAL_NUMBERS, serial_number_to_id
from steves_utils.ORACLE.shuffled_dataset_accessor import Shuffled_Dataset_Factory
import steves_utils.dataset_shuffler
import tensorflow as tf
import steves_utils.utils as utils

from functools import reduce

from typing import List
import math

import os,sys,json

class Windowed_Dataset_Shuffler:
    """Alright this one is a real doozy
    Will take an already shuffled and chunked dataset (basically use all_shuffled_chunk-512) and
        output the following:
        - A windowed and shuffled train dataset
        - A monolithic validation dataset file (not windowed)
        - A monolithic test dataset file (not windowed)
    
    The total desired number of examples _PER DEVICE_ for the train/test/val datasets are specified.
    
    The train/test/val datasets are guaranteed to not contain examples that are related to each other 
        at all (IE they will not share chunks at _all_)

    Args:
        distances_to_filter_on: We will take only these distances. Does not imply we take X amount of them
                                though. Instead we simply rely on the underlying dataset being thorughly
                                shuffled.
        serials_to_filter_on: We will take num_windowed_examples_per_device from each of these serials.

    Workflow (See method docstring for details):
            shuffler.create_and_check_dirs()
                This is optional. Creates the directory structure (pile and output dir)
            shuffler.write_piles()
                Write the piles to the output dir
            shuffler.shuffle_piles()
                Shuffle the piles in memory, concatenate them to the output dir.
    """

    def __init__(
        self,
        input_shuffled_ds_dir,
        input_shuffled_ds_num_samples_per_chunk,
        output_batch_size,
        output_max_file_size_MB,
        seed,
        num_windowed_examples_per_device,
        num_val_examples_per_device,
        num_test_examples_per_device,
        output_window_size,
        distances_to_filter_on,
        serials_to_filter_on,
        working_dir,
        output_format_str,
        stride_length,
        # output_format_str="shuffled_batchSize-{batch_size}_part-{part}.tfrecord_ds",
    ) -> None:
        self.serial_ids_to_filter_on                 = [serial_number_to_id(serial) for serial in  serials_to_filter_on]
        self.num_windowed_examples_per_device        = num_windowed_examples_per_device
        self.num_val_examples_per_device             = num_val_examples_per_device
        self.num_test_examples_per_device            = num_test_examples_per_device
        self.input_shuffled_ds_num_samples_per_chunk = input_shuffled_ds_num_samples_per_chunk
        self.output_batch_size                       = output_batch_size
        self.output_max_file_size_MB                 = output_max_file_size_MB
        self.seed                                    = seed
        self.output_window_size                      = output_window_size
        self.stride_length                           = stride_length

        self.window_pile_dir                         = os.path.join(working_dir, "pile_train")
        self.window_output_dir                       = os.path.join(working_dir, "train")
        self.val_pile_dir                            = os.path.join(working_dir, "pile_val")
        self.val_output_dir                          = os.path.join(working_dir, "val")
        self.test_pile_dir                           = os.path.join(working_dir, "pile_test")
        self.test_output_dir                         = os.path.join(working_dir, "test")

        # If necessary we can customize these
        self.window_output_format_str                = output_format_str
        self.val_output_format_str                   = output_format_str
        self.test_output_format_str                  = output_format_str

        self.num_devices = len(self.serial_ids_to_filter_on)
        
        # Yeah it's pretty hacky since we don't really need to split the dataset into test and val, but
        # it's already written and tested
        datasets = Shuffled_Dataset_Factory(
            input_shuffled_ds_dir, train_val_test_splits=(0.6, 0.2, 0.2), reshuffle_train_each_iteration=False
        )

        self.train_ds = datasets["train_ds"].unbatch()
        self.val_ds = datasets["val_ds"].unbatch()
        self.test_ds = datasets["test_ds"].unbatch()

        self.og_datasets = {
            "train_ds": self.train_ds, 
            "val_ds": self.val_ds,
            "test_ds": self.test_ds
        }

        # Since we are windowing, the number of examples we take from the original dataset is smaller
        # than the actual number of windows we want to generate
        replication_factor = math.floor((input_shuffled_ds_num_samples_per_chunk - output_window_size)/stride_length + 1)
        num_train_examples_to_get_per_device = math.ceil(num_windowed_examples_per_device/replication_factor)

        # These are a little different. Since we are basically unchunking by striding as big as our output window
        num_val_examples_per_device = math.ceil(num_val_examples_per_device / (input_shuffled_ds_num_samples_per_chunk/output_window_size))
        num_test_examples_per_device = math.ceil(num_test_examples_per_device / (input_shuffled_ds_num_samples_per_chunk/output_window_size))

        # print("Fetching {} Train Chunks Per Device".format(num_train_examples_to_get_per_device))
        # print("Fetching {} Val Chunks Per Device".format(num_val_examples_per_device))
        # print("Fetching {} Tess Chunks Per Device".format(num_test_examples_per_device))
        # print("Replication Factor:", replication_factor)

        self.train_ds = Windowed_Dataset_Shuffler.build_per_device_filtered_dataset(
            distances_to_filter_on=distances_to_filter_on,
            serial_ids_to_filter_on=self.serial_ids_to_filter_on,
            num_examples_per_serial_id=num_train_examples_to_get_per_device,
            ds=self.train_ds,
        )
        self.val_ds = Windowed_Dataset_Shuffler.build_per_device_filtered_dataset(
            distances_to_filter_on=distances_to_filter_on,
            serial_ids_to_filter_on=self.serial_ids_to_filter_on,
            num_examples_per_serial_id=num_val_examples_per_device,
            ds=self.val_ds,
        )
        self.test_ds = Windowed_Dataset_Shuffler.build_per_device_filtered_dataset(
            distances_to_filter_on=distances_to_filter_on,
            serial_ids_to_filter_on=self.serial_ids_to_filter_on,
            num_examples_per_serial_id=num_test_examples_per_device,
            ds=self.test_ds,
        )

        # print("Train Length Before Windowing:", utils.get_iterator_cardinality(self.train_ds))
        # print("Val Length Before Windowing:", utils.get_iterator_cardinality(self.val_ds))
        # print("Test Length Before Windowing:", utils.get_iterator_cardinality(self.test_ds))

        self.train_ds = self.window_ds(self.train_ds, self.stride_length)
        self.val_ds   = self.window_ds(self.val_ds, output_window_size)
        self.test_ds   = self.window_ds(self.test_ds, output_window_size)

        # self.train_ds = self.train_ds.batch(1000)
        # self.val_ds = self.val_ds.batch(1000)
        # self.test_ds = self.test_ds.batch(1000)

        # print("Train Length:", utils.get_iterator_cardinality(self.train_ds))
        # print("Val Length:", utils.get_iterator_cardinality(self.val_ds))
        # print("Test Length:", utils.get_iterator_cardinality(self.test_ds))


        # raise Exception("Done")

        # This is another straight up hack. The val and test aren't really shuffled, we're just using this to write the DS to file
        self.train_shuffler = self.make_train_shuffler()
        self.val_shuffler   = self.make_val_shuffler()
        self.test_shuffler  = self.make_test_shuffler()



    @staticmethod
    def build_per_device_filtered_dataset(
        distances_to_filter_on,
        serial_ids_to_filter_on,
        num_examples_per_serial_id,
        ds,
    ):
        """Filters and takes the appropriate number of examples from each device. Does not do any mapping/windowing"""

        distances_to_get = tf.constant(distances_to_filter_on, dtype=tf.dtypes.uint8)

        ds = ds.filter(lambda x: 
            tf.math.count_nonzero(
                tf.math.equal(
                    x["distance_feet"], distances_to_get
                )
            ) == 1
        )

        datasets = []

        for serial in serial_ids_to_filter_on:
            datasets.append(ds.filter(lambda x: x["serial_number_id"] == serial).take(num_examples_per_serial_id))

        return reduce(lambda a,b: a.concatenate(b), datasets)

    def window_ds(self, ds, stride_length):
        """Applies our window function across the already shuffled dataset"""        

        num_repeats= math.floor((self.input_shuffled_ds_num_samples_per_chunk - self.output_window_size)/stride_length) + 1

        ds = ds.map(
            lambda x: {
                "IQ": tf.transpose(
                    tf.signal.frame(x["IQ"], self.output_window_size, stride_length),
                    [1,0,2]
                ),
                # "index_in_file": tf.repeat(tf.reshape(x["index_in_file"], (1, x["index_in_file"].shape[0])), repeats=num_repeats, axis=0),
                # "index_in_file": tf.repeat(tf.reshape(x["index_in_file"], (1,1)), repeats=num_repeats, axis=0),
                # "serial_number_id": tf.repeat(tf.reshape(x["serial_number_id"], (1,1)), repeats=num_repeats, axis=0),
                # "distance_feet": tf.repeat(tf.reshape(x["distance_feet"], (1,1)), repeats=num_repeats, axis=0),
                # "run": tf.repeat(tf.reshape(x["run"], (1,1)), repeats=num_repeats, axis=0),
                "index_in_file": tf.repeat(x["index_in_file"], repeats=num_repeats, axis=0),
                "serial_number_id": tf.repeat(x["serial_number_id"], repeats=num_repeats, axis=0),
                "distance_feet": tf.repeat(x["distance_feet"], repeats=num_repeats, axis=0),
                "run": tf.repeat(x["run"], repeats=num_repeats, axis=0),
            },
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True
        )

        ds = ds.unbatch()

        return ds

    def make_train_shuffler(self) -> steves_utils.dataset_shuffler.Dataset_Shuffler:
        ds_size_GB =  self.num_windowed_examples_per_device * self.num_devices * self.output_window_size * 8 * 2 / 1024 / 1024 / 1024
        num_piles = int(math.ceil(ds_size_GB))

        self.expected_num_parts = ds_size_GB * 1024 / self.output_max_file_size_MB
        if self.expected_num_parts < 15:
            raise Exception("Expected number of output parts is {}, need a minimum of 15".format(self.expected_num_parts))


        return steves_utils.dataset_shuffler.Dataset_Shuffler(
            input_ds=self.train_ds,
            one_example_to_tf_record_func=oracle_serialization.example_to_tf_record,
            one_example_from_serialized_tf_record_func=oracle_serialization.serialized_tf_record_to_example,
            batch_example_to_tf_record_func=oracle_serialization.example_to_tf_record,
            output_batch_size=self.output_batch_size,
            num_piles=num_piles,
            output_format_str=self.window_output_format_str,
            output_max_file_size_MB=self.output_max_file_size_MB,
            pile_dir=self.window_pile_dir,
            output_dir=self.window_output_dir,
            seed=self.seed
        )


    def make_val_shuffler(self) -> steves_utils.dataset_shuffler.Dataset_Shuffler:
        return steves_utils.dataset_shuffler.Dataset_Shuffler(
            input_ds=self.val_ds,
            one_example_to_tf_record_func=oracle_serialization.example_to_tf_record,
            one_example_from_serialized_tf_record_func=oracle_serialization.serialized_tf_record_to_example,
            batch_example_to_tf_record_func=oracle_serialization.example_to_tf_record,
            output_batch_size=self.output_batch_size,
            num_piles=1,
            output_format_str=self.val_output_format_str,
            output_max_file_size_MB=100*1024,
            pile_dir=self.val_pile_dir,
            output_dir=self.val_output_dir,
            seed=self.seed,
        )


    def make_test_shuffler(self) -> steves_utils.dataset_shuffler.Dataset_Shuffler:
        return steves_utils.dataset_shuffler.Dataset_Shuffler(
            input_ds=self.test_ds,
            one_example_to_tf_record_func=oracle_serialization.example_to_tf_record,
            one_example_from_serialized_tf_record_func=oracle_serialization.serialized_tf_record_to_example,
            batch_example_to_tf_record_func=oracle_serialization.example_to_tf_record,
            output_batch_size=self.output_batch_size,
            num_piles=1,
            output_format_str=self.test_output_format_str,
            output_max_file_size_MB=100*1024,
            pile_dir=self.test_pile_dir,
            output_dir=self.test_output_dir,
            seed=self.seed,
        )

    def get_total_ds_size_GB(self):
        return self.total_ds_size_GB
    def get_expected_pile_size_GB(self):
        return self.expected_pile_size_GB
    def get_expected_num_parts(self):
        return self.expected_num_parts

    def create_and_check_dirs(self):
        for s in (self.train_shuffler, self.val_shuffler,self.test_shuffler):
            s.create_and_check_dirs()

    def shuffle_piles(self, reuse_piles=False):
        print("Shuffling Train")
        self.train_shuffler.shuffle_piles(reuse_piles)

        print("Shuffling Val")
        self.val_shuffler.shuffle_piles(reuse_piles)

        print("Shuffling Test")
        self.test_shuffler.shuffle_piles(reuse_piles)

    def write_piles(self):
        print("Write Train Piles")
        self.train_shuffler.write_piles()

        print("Write Val Piles")
        self.val_shuffler.write_piles()
        
        print("Write Test Piles")
        self.test_shuffler.write_piles()
    
    def get_num_piles(self):
        return self.num_piles

    def get_datasets(self):
        return {
            "train_ds": self.train_ds,
            "val_ds": self.val_ds,
            "test_ds": self.test_ds
        }

    def get_og_datasets(self):
        return self.og_datasets


if __name__ == "__main__":
    from steves_utils.ORACLE.utils import ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ALL_SERIAL_NUMBERS


    if sys.argv[1] == "-":
        print("Reading config from stdin")
        c = json.loads(sys.stdin.read())

        print(c)

        shuffler = Windowed_Dataset_Shuffler(
            input_shuffled_ds_dir=c["input_shuffled_ds_dir"],
            input_shuffled_ds_num_samples_per_chunk=c["input_shuffled_ds_num_samples_per_chunk"],
            output_batch_size=c["output_batch_size"],
            seed=c["seed"],
            num_windowed_examples_per_device=c["num_windowed_examples_per_device"],
            num_val_examples_per_device=c["num_val_examples_per_device"],
            num_test_examples_per_device=c["num_test_examples_per_device"],
            output_max_file_size_MB=c["output_max_file_size_MB"],
            distances_to_filter_on=c["distances_to_filter_on"],
            output_window_size=c["output_window_size"],
            serials_to_filter_on=ALL_SERIAL_NUMBERS,
            working_dir=c["working_dir"],
            output_format_str="batch-{batch_size}_part-{part}.tfrecord_ds",
            stride_length=c["stride_length"],
        )

        # shuffler = Windowed_Dataset_Shuffler(
        #     input_shuffled_ds_dir="/mnt/wd500GB/CSC500/csc500-super-repo/datasets/all_shuffled_chunk-512/output",
        #     input_shuffled_ds_num_samples_per_chunk=4*ORIGINAL_PAPER_SAMPLES_PER_CHUNK,
        #     output_batch_size=100,
        #     seed=1337,
        #     num_windowed_examples_per_device=int(200e3),
        #     num_val_examples_per_device=int(10e3),
        #     num_test_examples_per_device=int(50e3),
        #     output_max_file_size_MB=100,
        #     # output_max_file_size_MB=1,
        #     # num_windowed_examples_per_device=int(3e3),
        #     # num_val_examples_per_device=int(1e3),
        #     # num_test_examples_per_device=int(2e3),
        #     distances_to_filter_on=ALL_DISTANCES_FEET,
        #     output_window_size=ORIGINAL_PAPER_SAMPLES_PER_CHUNK, 
        #     serials_to_filter_on=ALL_SERIAL_NUMBERS,
        #     working_dir="/mnt/wd500GB/CSC500/csc500-super-repo/datasets/windowed_200k-each-devices_batch-100/",
        #     output_format_str="batch-{batch_size}_part-{part}.tfrecord_ds",
        #     stride_length=1
        # )

    # else:
    #     shuffler = Windowed_Dataset_Shuffler(
    #         input_shuffled_ds_dir="/mnt/wd500GB/CSC500/csc500-super-repo/datasets/all_shuffled_chunk-512/output",
    #         input_shuffled_ds_num_samples_per_chunk=4*ORIGINAL_PAPER_SAMPLES_PER_CHUNK,
    #         output_batch_size=100,
    #         seed=1337,
    #         num_windowed_examples_per_device=int(200e3),
    #         num_val_examples_per_device=int(10e3),
    #         num_test_examples_per_device=int(50e3),
    #         output_max_file_size_MB=100,
    #         # output_max_file_size_MB=1,
    #         # num_windowed_examples_per_device=int(3e3),
    #         # num_val_examples_per_device=int(1e3),
    #         # num_test_examples_per_device=int(2e3),
    #         distances_to_filter_on=ALL_DISTANCES_FEET,
    #         output_window_size=ORIGINAL_PAPER_SAMPLES_PER_CHUNK, 
    #         serials_to_filter_on=ALL_SERIAL_NUMBERS,
    #         working_dir="/mnt/wd500GB/CSC500/csc500-super-repo/datasets/windowed_200k-each-devices_batch-100/",
    #         output_format_str="batch-{batch_size}_part-{part}.tfrecord_ds",
    #         stride_length=1
    #     )

    shuffler.create_and_check_dirs()
    print("Write piles")
    shuffler.write_piles()
    print("shuffle")
    shuffler.shuffle_piles()