#! /usr/bin/env python3

import steves_utils.dataset_shuffler 
import steves_utils.ORACLE.serialization as oracle_serialization
import tensorflow as tf
from typing import Tuple


def Shuffled_Dataset_Factory(
    path: str,
    train_val_test_splits: Tuple[float,float,float],
    num_parallel_calls=tf.data.AUTOTUNE,
    reshuffle_train_each_iteration=True
):
    datasets = steves_utils.dataset_shuffler.Shuffled_Dataset_Factory(path, train_val_test_splits, reshuffle_train_each_iteration)

    train_ds = datasets["train_ds"]
    val_ds = datasets["val_ds"]
    test_ds = datasets["test_ds"]

    train_ds = train_ds.map(oracle_serialization.serialized_tf_record_to_example, num_parallel_calls=num_parallel_calls, deterministic=True)
    val_ds = val_ds.map(oracle_serialization.serialized_tf_record_to_example, num_parallel_calls=num_parallel_calls, deterministic=True)
    test_ds = test_ds.map(oracle_serialization.serialized_tf_record_to_example, num_parallel_calls=num_parallel_calls, deterministic=True)

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