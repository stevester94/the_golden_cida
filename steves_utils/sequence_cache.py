#! /usr/bin/env python3

import numpy as np

class Sequence_Cache:
    """
    cache up to max_items from an underlying indexable sequence
    """
    def __init__(self, sequence, max_items) -> None:
        self.sequence = sequence
        self.max_items = max_items
        self.cache = {}
        self.cache_misses = 0
        self.cache_hits = 0

        assert(self.max_items >= 0)

    
    def __getitem__(self, idx):

        if idx in self.cache.keys():
            self.cache_hits += 1
            return self.cache[idx]
        elif len(self.cache) < self.max_items:
            """
            Add the iter_idx element to the cache if it doesn't already exist
            """
            self.cache_misses += 1
            self.cache[idx] = self.sequence[idx]
            return self.cache[idx]
        else:
            """
            We've maxed out our cache, must fetch it from the underlying sequence
            """
            self.cache_misses += 1
            return self.sequence[idx]

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
    
    def get_len_cache(self):
        return len(self.cache)

    def get_cache_misses(self):
        return self.cache_misses

    def get_cache_hits(self):
        return self.cache_hits
    



if __name__ == "__main__":
    import unittest
    import random

    LEN_SEQUENCE = 100000
    MAX_CACHE_SIZE = 1000

    class test_File_As_Windowed_Sequence(unittest.TestCase):
        @classmethod
        def setUpClass(self) -> None:
            self.sequence = np.random.default_rng().integers(low=0, high=LEN_SEQUENCE, size=LEN_SEQUENCE)

        def test_length(self):
            sc =  Sequence_Cache(sequence=self.sequence, max_items=MAX_CACHE_SIZE)
        
            self.assertEqual(len(sc), len(self.sequence))

        def test_iteration(self):
            sc =  Sequence_Cache(sequence=self.sequence, max_items=MAX_CACHE_SIZE)

            for i,x in enumerate(sc):
                self.assertEqual(x, self.sequence[i])
        
        def test_random_indexing(self):
            sc =  Sequence_Cache(sequence=self.sequence, max_items=MAX_CACHE_SIZE)

            random_indices = list(range(len(self.sequence)))
            random.shuffle(random_indices)
            
            # Just make sure our random indices cover the entirety of our sequence
            self.assertEqual(len(random_indices), len(self.sequence))

            for i in random_indices:
                self.assertEqual(self.sequence[i], sc[i])

            for i in random_indices:
                self.assertEqual(self.sequence[i], sc[i])

        def test_cache_size(self):
            sc =  Sequence_Cache(sequence=self.sequence, max_items=MAX_CACHE_SIZE)

            random_indices = list(range(len(self.sequence)))
            random.shuffle(random_indices)
            
            # Just make sure our random indices cover the entirety of our sequence
            self.assertEqual(len(random_indices), len(self.sequence))

            for i in random_indices:
                self.assertEqual(self.sequence[i], sc[i])
        
                self.assertTrue(sc.get_len_cache() <= MAX_CACHE_SIZE)

            self.assertEqual(sc.get_len_cache(), MAX_CACHE_SIZE)
        
        def test_actual_caching(self):
            class Dangerous_Sequence:
                def __init__(self) -> None:
                    self.destruct = False
                
                def __getitem__(self, idx):
                    if self.destruct:
                        raise Exception("SELF DESTRUCTED")
                    else:
                        return 1
            
            max = 100
            ds = Dangerous_Sequence()
            sc =  Sequence_Cache(sequence=ds, max_items=max)

            for i in range(max):
                x = sc[i]

            # The dangerous sequence will now throw when accessed, so if caching is working, we will not actually hit the sequence
            ds.destruct = True
            for i in range(max):
                x = sc[i]
            
            # We should destruct now
            try:
                for i in range(max*2):
                    x = sc[i]
                self.fail("Should have destructed")
            except:
                # self.fail("heh")
                pass

    unittest.main()
