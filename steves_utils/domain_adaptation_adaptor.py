#! /usr/bin/env python3

import numpy as np

from steves_utils.lazy_map import Lazy_Map
from steves_utils.sequence_aggregator import Sequence_Aggregator

class Domain_Adaptation_Adaptor:
    """
    Takes source and target sequences, and appends a 1 or 0 respectively to their tuples
    """
    def __init__(self, source_seq, target_seq) -> None:
        source_seq = Lazy_Map(source_seq, lambda x: x + (1,))
        target_seq = Lazy_Map(target_seq, lambda x: x + (0,))

        self.seq = Sequence_Aggregator((source_seq, target_seq))


        
    
    def __getitem__(self, idx):
        return self.seq[idx]

    def __iter__(self):
        return iter(self.seq)

    def __len__(self):
        return len(self.seq)
        


if __name__ == "__main__":
    from oshea_RML2016_ds import OShea_RML2016_DS
    import unittest

    def check_equal(t1, t2):
        return np.array_equal(t1[0], t2[0]) and t1[1:] == t2[1:]

    class test_File_As_Windowed_Sequence(unittest.TestCase):
        @classmethod
        def setUpClass(self) -> None:
            self.source_ds = OShea_RML2016_DS(
                snrs_to_get=[18]
            )

            self.target_ds = OShea_RML2016_DS(
                snrs_to_get=[6]
            )

        def test_len(self):
            daa = Domain_Adaptation_Adaptor(self.source_ds, self.target_ds)

            self.assertEqual(len(self.source_ds), len(self.target_ds))
            self.assertEqual(len(self.source_ds)+len(self.target_ds), len(daa))
            
        def test_items(self):
            daa = Domain_Adaptation_Adaptor(self.source_ds, self.target_ds)



            for i in range(len(self.source_ds)):
                self.assertTrue(
                    check_equal(daa[i], self.source_ds[i]+(1,))
                )
            for i in range(len(self.source_ds), len(self.source_ds)+len(self.target_ds)):
                self.assertTrue(
                    check_equal(daa[i], self.target_ds[i-len(self.source_ds)]+(0,))
                )

        def test_full_length_indexing(self):
            daa = Domain_Adaptation_Adaptor(self.source_ds, self.target_ds)


            for i in range(len(daa)):
                _ = daa[i]


        def test_iteration(self):
            daa = Domain_Adaptation_Adaptor(self.source_ds, self.target_ds)
            daa = Domain_Adaptation_Adaptor(self.source_ds, [(1,)])

            it = iter(daa)

            for i in range(len(self.source_ds)):
                self.assertTrue(check_equal(daa[i], next(it)))

            x = next(it)

            # i = iter(daa)

            # for x in self.source_ds:
            #     self.assertTrue(
            #         check_equal(
            #             x + (1,),
            #             next(i)
            #         )
            #     )

            # for x in self.target_ds:
            #     self.assertTrue(
            #         check_equal(
            #             x + (0,),
            #             next(i)
            #         )
            #     )

            # for i,x in enumerate(daa[:int(len(daa)/2)]):
                
            # for i,x in daa[:int(len(daa)/2)]:
            #     self.assertTrue(
            #         x == self.source_ds[i]+(1,)
            #     )
            

    unittest.main()