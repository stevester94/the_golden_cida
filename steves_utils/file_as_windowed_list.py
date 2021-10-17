#! /usr/bin/env python3

import numpy as np
from math import floor

from numpy.testing._private.utils import assert_equal

class File_As_Windowed_Sequence:
    """
    Creates an indexable sequence out of a file given the window size and the stride size.

    """
    def __init__(
        self,
        path:str,
        window_length:int,
        stride:int,
        numpy_dtype:np.dtype,
        return_as_tuple_with_offset:bool=False) -> None:

        if stride < 1:
            raise Exception("Stride must be > 0")
        
        if window_length < 1:
            raise Exception("Window length must be > 0")

        
        self.f = open(path, "r+")

        self.memmap = np.memmap(self.f, numpy_dtype)

        self.window_length = window_length
        self.stride = stride

        self.len = floor((len(self.memmap) - self.window_length) / self.stride) + 1

        self.return_as_tuple_with_offset = return_as_tuple_with_offset

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        if index >= self.len or index < 0:
            raise IndexError
        
        if self.return_as_tuple_with_offset:
            return (
                index*self.stride,
                np.array(self.memmap[index*self.stride : index*self.stride+self.window_length])
            )
        else:
            return np.array(self.memmap[index*self.stride : index*self.stride+self.window_length])
        
    def __iter__(self):
        self.iter_idx = 0
        return self

    def __next__(self):
        if self.iter_idx >= len(self):
            raise StopIteration
        else:
            result = self[self.iter_idx]
            self.iter_idx += 1
            return result

    


if __name__ == "__main__":
    import unittest
    import tempfile

    TEST_DTYPE = np.single
    TEST_BUFFER_NUM_ITEMS = 10000
    STRIDES_TO_TEST = [1,2,3,5,10,50,90,100] # The tests break if bigger than the window (This is due to the tests not the faws)
    class test_File_As_Windowed_Sequence(unittest.TestCase):
        @classmethod
        def setUpClass(self) -> None:
            self.dtype = TEST_DTYPE
            self.num_items = TEST_BUFFER_NUM_ITEMS



            self.f = tempfile.NamedTemporaryFile("w+b")
            self.source = np.random.default_rng().integers(0, 100, self.num_items).astype(self.dtype)

            self.f.write(self.source.tobytes())
            self.f.seek(0)


        # @classmethod
        # def tearDownClass(self) -> None:
        #     pass

        def test_len(self):
            buf = np.frombuffer(self.f.read(), dtype=self.dtype)

            self.assertEqual(len(buf), self.num_items)
            self.assertEqual(len(buf), len(self.source))
        
        def test_big_window(self):
            faws = File_As_Windowed_Sequence(self.f.name, window_length=self.num_items, stride=30, numpy_dtype=self.dtype)

            self.assertTrue(np.array_equal(faws[0], self.source))
            
        def test_window_beginnings_are_true_to_source(self):

            for stride in STRIDES_TO_TEST:
                faws = File_As_Windowed_Sequence(self.f.name, window_length=100, stride=stride, numpy_dtype=self.dtype)

                built_up = []
                for i in faws:
                    built_up.extend(i[:stride])

                built_up = np.array(built_up, dtype=self.dtype)          

                self.assertTrue(np.array_equal(built_up, self.source[:len(built_up)]))

        def test_edge_case_stride_sizes(self):
            # Expect two windows from a source of length 10k
            stride = 5000
            faws = File_As_Windowed_Sequence(self.f.name, window_length=100, stride=stride, numpy_dtype=self.dtype)
            self.assertTrue(len(faws), 2)

            stride = 5001
            faws = File_As_Windowed_Sequence(self.f.name, window_length=100, stride=stride, numpy_dtype=self.dtype)
            self.assertTrue(len(faws), 1)

            stride = 9000
            faws = File_As_Windowed_Sequence(self.f.name, window_length=100, stride=stride, numpy_dtype=self.dtype)
            self.assertTrue(len(faws), 1)

            stride = 9000
            faws = File_As_Windowed_Sequence(self.f.name, window_length=stride, stride=stride, numpy_dtype=self.dtype)
            built_up = []
            for i in faws:
                built_up.extend(i[:stride])
            self.assertEqual(len(built_up), 9000)


            stride = 9000
            faws = File_As_Windowed_Sequence(self.f.name, window_length=stride, stride=stride, numpy_dtype=self.dtype)
            built_up = []
            for i in faws:
                built_up.extend(i[:stride])
            self.assertEqual(len(built_up), 9000)

            stride = 2
            faws = File_As_Windowed_Sequence(self.f.name, window_length=stride, stride=stride, numpy_dtype=self.dtype)
            built_up = []
            for i in faws:
                built_up.extend(i[:stride])
            self.assertEqual(len(built_up), len(self.source))

            stride = 1
            faws = File_As_Windowed_Sequence(self.f.name, window_length=stride, stride=stride, numpy_dtype=self.dtype)
            built_up = []
            for i in faws:
                built_up.extend(i[:stride])
            self.assertEqual(len(built_up), len(self.source))
            
        @unittest.expectedFailure
        def test_bad_path(self):
            File_As_Windowed_Sequence("non-existent", window_length=100, stride=1, numpy_dtype=self.dtype)

        @unittest.expectedFailure
        def test_bad_stride_1(self):
            File_As_Windowed_Sequence(self.f.name, window_length=100, stride=0, numpy_dtype=self.dtype)

        @unittest.expectedFailure
        def test_bad_stride_2(self):
            File_As_Windowed_Sequence(self.f.name, window_length=100, stride=-1, numpy_dtype=self.dtype)

        @unittest.expectedFailure
        def test_bad_window_1(self):
            File_As_Windowed_Sequence(self.f.name, window_length=0, stride=1, numpy_dtype=self.dtype)

        @unittest.expectedFailure
        def test_bad_window_2(self):
            File_As_Windowed_Sequence(self.f.name, window_length=-1, stride=1, numpy_dtype=self.dtype)

    unittest.main()