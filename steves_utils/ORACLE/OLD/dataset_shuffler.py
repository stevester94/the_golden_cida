#! /usr/bin/env python3

import steves_utils.ORACLE.serialization as oracle_serialization
from steves_utils.ORACLE.simple_oracle_dataset_factory import Simple_ORACLE_Dataset_Factory
from steves_utils.ORACLE.utils import ALL_DISTANCES_FEET, ALL_RUNS, ALL_SERIAL_NUMBERS
import steves_utils.dataset_shuffler
import tensorflow as tf

from typing import List
import math

import os

class Dataset_Shuffler:
    """Shuffles tensorflow datasets on disk
    Workflow (See method docstring for details):
            shuffler.create_and_check_dirs()
                This is optional. Creates the directory structure (pile and output dir)
            shuffler.write_piles()
                Write the piles to the output dir
            shuffler.shuffle_piles()
                Shuffle the piles in memory, concatenate them to the output dir.

    Basics:
        This class uses TFRecords to facilitate shuffling large datasets on disk. This is
        done by dropping the input examples into random piles which are small enough to fit
        in memory. These piles are then shuffled, and the output is concatenated to split 
        output files.
    
    NOTE:
        Care should be taken that num_piles results in piles which are small enough to fit in memory,
        but also large enough that reading from them is efficient (several GB is appropriate).

    NOTE:
        partial batches CAN be generated
    """

    def __init__(
        self,
        output_batch_size,
        # num_piles,
        output_max_file_size_MB,
        pile_dir,
        output_dir,
        seed,
        num_samples_per_chunk,
        distances_to_get: List[int] = ALL_DISTANCES_FEET, 
        serial_numbers_to_get: List[str] =ALL_SERIAL_NUMBERS,
        runs_to_get: List[int] = ALL_RUNS,
        output_format_str="shuffled_batchSize-{batch_size}_part-{part}.tfrecord_ds",
        fail_on_too_few_output_parts=True,
    ) -> None:
        self.output_batch_size       = output_batch_size
        # self.num_piles               = num_piles
        self.output_max_file_size_MB = output_max_file_size_MB
        self.pile_dir                = pile_dir
        self.output_dir              = output_dir
        self.seed                    = seed
        self.num_samples_per_chunk   = num_samples_per_chunk
        self.distances_to_get        = distances_to_get
        self.serial_numbers_to_get   = serial_numbers_to_get
        self.runs_to_get             = runs_to_get
        self.output_format_str       = output_format_str

        

        self.ds, self.cardinality = Simple_ORACLE_Dataset_Factory(
            num_samples_per_chunk, 
            runs_to_get=runs_to_get,
            distances_to_get=distances_to_get,
            serial_numbers_to_get=serial_numbers_to_get
        )

        self.total_ds_size_GB = self.cardinality * self.num_samples_per_chunk * 8 * 2 / 1024 / 1024 / 1024
        self.num_piles = int(math.ceil(self.total_ds_size_GB))
        # self.expected_pile_size_GB = self.total_ds_size_GB / self.num_piles
        # if self.expected_pile_size_GB > 5:
        #     raise Exception("Expected pile size is too big: {}GB. Increase your num_piles".format(self.expected_pile_size_GB))

        self.expected_num_parts = self.total_ds_size_GB * 1024 / output_max_file_size_MB
        if self.expected_num_parts < 15:
            if fail_on_too_few_output_parts:
                raise Exception("Expected number of output parts is {}, need a minimum of 15".format(self.expected_num_parts))
            else:
                print("Expected number of output parts is {}, need a minimum of 15".format(self.expected_num_parts))


        self.shuffler = steves_utils.dataset_shuffler.Dataset_Shuffler(
            input_ds=self.ds,
            one_example_to_tf_record_func=oracle_serialization.example_to_tf_record,
            one_example_from_serialized_tf_record_func=oracle_serialization.serialized_tf_record_to_example,
            batch_example_to_tf_record_func=oracle_serialization.example_to_tf_record,
            output_batch_size=output_batch_size,
            num_piles=self.num_piles,
            output_format_str=output_format_str,
            output_max_file_size_MB=output_max_file_size_MB,
            pile_dir=pile_dir,
            output_dir=output_dir,
            seed=seed
        )

    def get_total_ds_size_GB(self):
        return self.total_ds_size_GB
    def get_expected_pile_size_GB(self):
        return self.expected_pile_size_GB
    def get_expected_num_parts(self):
        return self.expected_num_parts

    def create_and_check_dirs(self):
        self.shuffler.create_and_check_dirs()

    def shuffle_piles(self, reuse_piles=False):
        self.shuffler.shuffle_piles(reuse_piles)

    def write_piles(self):
        self.shuffler.write_piles()
    
    def get_num_piles(self):
        return self.num_piles


if __name__ == "__main__":
    from steves_utils.ORACLE.utils import ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ALL_SERIAL_NUMBERS
    shuffler = Dataset_Shuffler(
        # num_piles=5,
        # output_max_file_size_MB=1,
        pile_dir="/mnt/wd500GB/CSC500/csc500-super-repo/datasets/all_shuffled_chunk-512/pile",
        output_dir="/mnt/wd500GB/CSC500/csc500-super-repo/datasets/all_shuffled_chunk-512/output",
        output_format_str="shuffled_chunk-512_batchSize-{batch_size}_part-{part}.tfrecord_ds",
        num_samples_per_chunk=4*ORIGINAL_PAPER_SAMPLES_PER_CHUNK,
        output_batch_size=100,
        output_max_file_size_MB=500,
        seed=1337,
        # runs_to_get=[1],
        # distances_to_get=ALL_DISTANCES_FEET[:1],
        # serial_numbers_to_get=ALL_SERIAL_NUMBERS[:6]
    )


    shuffler.create_and_check_dirs()

    print("Num piles:", shuffler.get_num_piles())
    input("Press enter to continue")
    print("Write piles")
    shuffler.write_piles()
    print("shuffle")
    shuffler.shuffle_piles()