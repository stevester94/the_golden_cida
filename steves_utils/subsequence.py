#! /usr/bin/env python3

import numpy as np

class Subsequence:
    """
    Exposes a contiguous subsequence of an original sequence
    """
    def __init__(self, sequence, start_index_inclusive, end_index_exclusive) -> None:
        self.sequence = sequence
        self.start_index = start_index_inclusive
        self.stop_index  = end_index_exclusive

        assert len(self.sequence) > 0
        assert self.start_index < self.stop_index
        assert self.stop_index > 0
        assert self.stop_index <= len(sequence)

        self.length = self.stop_index - self.start_index
    
    def __getitem__(self, idx):
        if idx > self.length:
            raise IndexError

        return self.sequence[idx+self.start_index]

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.length:
            raise StopIteration

        x = self[self.idx]
        self.idx += 1
        return x

    def __len__(self):
        return self.length
    



if __name__ == "__main__":
    import unittest

    SEQUENCE_LENGTH   = 1000
    SUBSEQUENCE_START = 20
    SUBSEQUENCE_END   = 970

    class test_Subsequence(unittest.TestCase):
        def test_length(self):
            sub = Subsequence(list(range(SEQUENCE_LENGTH)), SUBSEQUENCE_START, SUBSEQUENCE_END)

            self.assertEqual(len(sub), SUBSEQUENCE_END-SUBSEQUENCE_START)

        @unittest.expectedFailure
        def test_length_fail(self):
            sub = Subsequence(list(range(SEQUENCE_LENGTH)), SUBSEQUENCE_START, SUBSEQUENCE_END)

            self.assertEqual(len(sub), SUBSEQUENCE_END-SUBSEQUENCE_START+1)
        
        def test_indexing(self):
            l = list(range(SEQUENCE_LENGTH))[SUBSEQUENCE_START:SUBSEQUENCE_END]
            sub = Subsequence(list(range(SEQUENCE_LENGTH)), SUBSEQUENCE_START, SUBSEQUENCE_END)

            for idx, x in enumerate(l):
                self.assertEqual(x, sub[idx])
        
        def test_iteration(self):
            l = list(range(SEQUENCE_LENGTH))
            sub = Subsequence(list(range(SEQUENCE_LENGTH)), SUBSEQUENCE_START, SUBSEQUENCE_END)

            for idx, x in enumerate(sub):
                self.assertEqual(l[SUBSEQUENCE_START+idx], x)



    unittest.main()
