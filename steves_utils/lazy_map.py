#! /usr/bin/env python3

import numpy as np

class Lazy_Map:
    """
    Map a sequence, applying a lambda for any index
    """
    def __init__(self, sequence, lam) -> None:
        self.sequence = sequence
        self.lam = lam

    def __getitem__(self, idx):
        return self.lam(self.sequence[idx])

    def __iter__(self):
        self.iter_idx = -1
        return self

    def __next__(self):
        self.iter_idx += 1

        if self.iter_idx >= len(self.sequence):
            raise StopIteration
        
        return self[self.iter_idx]

    def __len__(self):
        return len(self.sequence)
    

if __name__ == "__main__":
    import unittest
    import random

    LEN_SEQUENCE = 100000
    LAM = lambda x: x*x

    class test_File_As_Windowed_Sequence(unittest.TestCase):
        @classmethod
        def setUpClass(self) -> None:
            self.sequence = np.random.default_rng().integers(low=0, high=LEN_SEQUENCE, size=LEN_SEQUENCE)

        def test_length(self):
            lm =  Lazy_Map(sequence=self.sequence, lam=LAM)
        
            self.assertEqual(len(lm), len(self.sequence))

        def test_iteration(self):
            lm =  Lazy_Map(sequence=self.sequence, lam=LAM)

            for i,x in enumerate(lm):
                self.assertEqual(x, LAM(self.sequence[i]))
        
        def test_random_indexing(self):
            lm =  Lazy_Map(sequence=self.sequence, lam=LAM)

            random_indices = list(range(len(self.sequence)))
            random.shuffle(random_indices)
            
            # Just make sure our random indices cover the entirety of our sequence
            self.assertEqual(len(random_indices), len(self.sequence))

            for i in random_indices:
                x = lm[i]
                truth = LAM(self.sequence[i])

                self.assertEqual(LAM(self.sequence[i]), lm[i])

    unittest.main()
