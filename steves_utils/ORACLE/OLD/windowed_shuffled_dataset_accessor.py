#! /usr/bin/env python3

import steves_utils.dataset_shuffler 
import steves_utils.ORACLE.serialization as oracle_serialization
import tensorflow as tf
from typing import Tuple
import os
import steves_utils.utils
from steves_utils.ORACLE.serialization import serialized_tf_record_to_example

def Windowed_Shuffled_Dataset_Factory(
    top_path: str,
    num_parallel_calls=tf.data.AUTOTUNE,
    reshuffle_train_each_iteration=True
):
    train_path = os.path.join(top_path, "train")
    val_path   = os.path.join(top_path, "val")
    test_path  = os.path.join(top_path, "test")

    train_ds = steves_utils.dataset_shuffler.Monolothic_Shuffled_Dataset_Factory(
        path=train_path, 
        reshuffle_each_iteration=reshuffle_train_each_iteration,
    )
    val_ds = steves_utils.dataset_shuffler.Monolothic_Shuffled_Dataset_Factory(
        path=val_path, 
        reshuffle_each_iteration=False,
    )
    test_ds = steves_utils.dataset_shuffler.Monolothic_Shuffled_Dataset_Factory(
        path=test_path, 
        reshuffle_each_iteration=False,
    )

    train_ds = train_ds.map(
        serialized_tf_record_to_example,
        num_parallel_calls=num_parallel_calls
    )
    val_ds   = val_ds.map(
        serialized_tf_record_to_example,
        num_parallel_calls=num_parallel_calls
    )
    test_ds  = test_ds.map(
        serialized_tf_record_to_example,
        num_parallel_calls=num_parallel_calls
    )


    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
    }

if __name__ == "__main__":
    datasets = Shuffled_Dataset_Factory("/mnt/wd500GB/derp/output", train_val_test_splits=(0.6, 0.2, 0.2))

    train_ds = datasets["train_ds"]
    val_ds = datasets["val_ds"]
    test_ds = datasets["test_ds"]

    count = 0
    for e in train_ds:
        count += e["IQ"].shape[0]
        print(count)