#! /usr/bin/env python3

from tensorflow._api.v2 import dtypes
from tensorflow.python.ops.gen_parsing_ops import serialize_tensor
from steves_utils import utils
import tensorflow as tf
import numpy as np
import tensorflow as tf
import unittest


"""
A brief explanation of this madness:
We are taking an example (either batched or unbatched) from a TF dataset, and converting it into a protobuf.
TF's handling of non-scalars in a protobuf looks bad, so I simply serialize the tensors, then stick them in
byte features.

The flow can be seen in the unit tests.

The importan functions here are
example_to_tf_record, 
"""

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def example_to_tf_record(example):
    serialized_IQ =                  tf.io.serialize_tensor(example["IQ"]).numpy()
    serialized_index_in_file =       tf.io.serialize_tensor(example["index_in_file"]).numpy()
    serialized_serial_number_id =    tf.io.serialize_tensor(example["serial_number_id"]).numpy()
    serialized_distance_feet =       tf.io.serialize_tensor(example["distance_feet"]).numpy()
    serialized_run =                 tf.io.serialize_tensor(example["run"]).numpy()

    feature = {
        "IQ":                _bytes_feature([serialized_IQ]),
        "index_in_file":     _bytes_feature([serialized_index_in_file]),
        "serial_number_id":  _bytes_feature([serialized_serial_number_id]),
        "distance_feet":     _bytes_feature([serialized_distance_feet]),
        "run":               _bytes_feature([serialized_run]),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

_tf_record_description = {
    'IQ':                 tf.io.FixedLenFeature([], tf.string, default_value=''),
    'index_in_file':      tf.io.FixedLenFeature([], tf.string, default_value=''),
    'serial_number_id':   tf.io.FixedLenFeature([], tf.string, default_value=''),
    'distance_feet':      tf.io.FixedLenFeature([], tf.string, default_value=''),
    'run':                tf.io.FixedLenFeature([], tf.string, default_value=''),
}
def serialized_tf_record_to_example(serialized_example):
    parsed_example = tf.io.parse_single_example(serialized_example, _tf_record_description)

    parsed_example["IQ"] = tf.io.parse_tensor(parsed_example["IQ"], tf.float64)
    parsed_example["index_in_file"] = tf.io.parse_tensor(parsed_example["index_in_file"], tf.int64)
    parsed_example["serial_number_id"] = tf.io.parse_tensor(parsed_example["serial_number_id"], tf.uint8)
    parsed_example["distance_feet"] = tf.io.parse_tensor(parsed_example["distance_feet"], tf.uint8)
    parsed_example["run"] = tf.io.parse_tensor(parsed_example["run"], tf.uint8)

    return parsed_example

class Test_serialization(unittest.TestCase):
    def test_serialize_deserialize_no_batching(self):
        from steves_utils.ORACLE.simple_oracle_dataset_factory import Simple_ORACLE_Dataset_Factory
        from steves_utils.ORACLE.utils import ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ALL_SERIAL_NUMBERS

        ds, cardinality = Simple_ORACLE_Dataset_Factory(
            ORIGINAL_PAPER_SAMPLES_PER_CHUNK, 
            runs_to_get=[1],
            distances_to_get=[8],
            serial_numbers_to_get=ALL_SERIAL_NUMBERS[:3]
        )

        for e in ds.take(10000):
            record = example_to_tf_record(e)
            serialized = record.SerializeToString()
            deserialized = serialized_tf_record_to_example(serialized)

            self.assertTrue(
                np.array_equal(
                    e["IQ"].numpy(),
                    deserialized["IQ"].numpy()
                )
            )

            # Just a dumb sanity check
            self.assertFalse(
                np.array_equal(
                    e["IQ"].numpy(),
                    deserialized["IQ"].numpy()[0]
                )
            )

    def test_serialize_deserialize_with_batching(self):
        from steves_utils.ORACLE.simple_oracle_dataset_factory import Simple_ORACLE_Dataset_Factory
        from steves_utils.ORACLE.utils import ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ALL_SERIAL_NUMBERS

        ds, cardinality = Simple_ORACLE_Dataset_Factory(
            ORIGINAL_PAPER_SAMPLES_PER_CHUNK, 
            runs_to_get=[1],
            distances_to_get=[8],
            serial_numbers_to_get=ALL_SERIAL_NUMBERS[:3]
        )

        for e in ds.batch(1000):
            record = example_to_tf_record(e)
            serialized = record.SerializeToString()
            deserialized = serialized_tf_record_to_example(serialized)

            self.assertTrue(
                np.array_equal(
                    e["IQ"].numpy(),
                    deserialized["IQ"].numpy()
                )
            )

            # Just a dumb sanity check
            self.assertFalse(
                np.array_equal(
                    e["IQ"].numpy(),
                    deserialized["IQ"].numpy()[0]
                )
            )

class Test_basic_serialization(unittest.TestCase):
    def test_basic_serialization(self):
        c = tf.constant(1337, dtype=tf.uint8)

        serialized_c = tf.io.serialize_tensor(c).numpy()


        # if example["index_in_file"].shape != ():
        # if True:
        feature = {
            "c":                _bytes_feature([serialized_c]),
        }

        example =  tf.train.Example(features=tf.train.Features(feature=feature))

        serialized_example = example.SerializeToString()

        _tf_record_description = {
            'c':                 tf.io.FixedLenFeature([], tf.string, default_value=''),
        }

        parsed_example = tf.io.parse_single_example(serialized_example, _tf_record_description)
        parsed_example["c"] = tf.io.parse_tensor(parsed_example["c"], tf.uint8)

        self.assertTrue(
            np.array_equal(
                parsed_example["c"],
                c
            )
        )


if __name__ == "__main__":
    unittest.main()

    # from steves_utils.ORACLE.simple_oracle_dataset_factory import Simple_ORACLE_Dataset_Factory
    # from steves_utils.ORACLE.utils import ORIGINAL_PAPER_SAMPLES_PER_CHUNK, ALL_SERIAL_NUMBERS

    # ds, cardinality = Simple_ORACLE_Dataset_Factory(
    #     ORIGINAL_PAPER_SAMPLES_PER_CHUNK, 
    #     runs_to_get=[1],
    #     distances_to_get=[8],
    #     serial_numbers_to_get=ALL_SERIAL_NUMBERS[:3]
    # )

    # for e in ds.batch(2).take(1):

    #     record = example_to_tf_record(e)
    #     serialized = record.SerializeToString()
    #     deserialized = serialized_tf_record_to_example(serialized)

    #     print(deserialized)