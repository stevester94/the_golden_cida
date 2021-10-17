#! /usr/bin/env python3

import numpy as np

class Sequence_Mask:
    """
    Iterate/index a sequence based on a mask of indices provided. The mask need not have unique indices
    """
    def __init__(self, sequence, mask) -> None:
        self.sequence = sequence
        self.mask  = mask

        assert(len(self.mask) > 0)

    
    def __getitem__(self, idx):
        return self.sequence[self.mask[idx]]

    def __iter__(self):
        self.iter_idx = -1
        return self

    def __next__(self):
        self.iter_idx += 1 

        if self.iter_idx >= len(self.mask):
            raise StopIteration

        return self[self.iter_idx]

    def __len__(self):
        return len(self.mask)
    
if __name__ == "__main__":
    import unittest
    import random

    LEN_SEQUENCE = 100000
    LEN_MASK=1000
    MAX_CACHE_SIZE = 1000

    class test_File_As_Windowed_Sequence(unittest.TestCase):
        @classmethod
        def setUpClass(self) -> None:
            self.mask = np.random.default_rng().integers(low=0, high=LEN_MASK, size=LEN_MASK)
            self.sequence = list(range(LEN_SEQUENCE))

        def test_length(self):
            sm =  Sequence_Mask(sequence=self.sequence, mask=self.mask)
        
            self.assertEqual(len(sm), len(self.mask))

        def test_iteration(self):
            sm =  Sequence_Mask(sequence=self.sequence, mask=self.mask)

            for i,x in enumerate(sm):
                masked_idx = self.mask[i]

                truth =  self.sequence[masked_idx]

                self.assertEqual(x, truth)

        def test_indexing(self):
            sm =  Sequence_Mask(sequence=self.sequence, mask=self.mask)

            for i in range(LEN_MASK):
                x = sm[i]
                truth = self.sequence[self.mask[i]]

                self.assertEqual(x,truth)
        
    unittest.main()
