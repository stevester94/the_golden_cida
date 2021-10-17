#! /usr/bin/env python3

import numpy as np

class Sequence_Aggregator:
    """
    Combine input sequences into one large sequence.
    Indices are only preserved for the first sequence, the rest are the sum of the previous sequence.
    Order matters for sequence.
    Each list in sequence must have length > zero
    """
    def __init__(self, sequences) -> None:
        self.sequences = sequences

        for l in self.sequences:
            if len(l) <= 0:
                raise "List received in List_Aggregator of 0 or negative length"
        
        self.lengths = [len(l) for l in self.sequences]

        self.idx_list = [0]
        for l in self.lengths:
            self.idx_list.append(self.idx_list[-1] + l)

        self.length = sum(self.lengths)

    
    def __getitem__(self, idx):
        which_seq = np.searchsorted(self.idx_list, idx, side="right") - 1
        idx_in_seq = idx - self.idx_list[which_seq]
        
        # print(which_seq)
        # print(idx_in_seq)

        return self.sequences[which_seq][idx_in_seq]

    def __iter__(self):
        self.sequences_iter = iter(self.sequences)
        self.sub_sequence_iter = iter(next(self.sequences_iter))

        return self

    def __next__(self):
        try:
            x = next(self.sub_sequence_iter)
        except StopIteration:
            self.sub_sequence_iter = iter(next(self.sequences_iter))

            x = next(self.sub_sequence_iter)
        return x

    def __len__(self):
        return self.length
    



if __name__ == "__main__":
    import unittest
    import tempfile

    NUM_SUBSEQUENCES = 100
    MIN_SUBSEQ_LENGTH = 1
    MAX_SUBSEQ_LENGTH = 100

    class test_File_As_Windowed_Sequence(unittest.TestCase):
        @classmethod
        def setUpClass(self) -> None:
            self.sequences = []

            for i in range(NUM_SUBSEQUENCES):
                length = np.random.default_rng().integers(MIN_SUBSEQ_LENGTH+1, MAX_SUBSEQ_LENGTH, 1)[0]

                if len(self.sequences) > 0:
                    last_seq_end = self.sequences[-1][-1]
                else:
                    last_seq_end = 0

                self.sequences.append(list(range(last_seq_end+1, last_seq_end+1+length)))
            # print(self.sequences)
        # @classmethod
        # def tearDownClass(self) -> None:
        #     pass

        # Verify our test bed is good
        def test_testbed_is_kosher(self):
            self.assertEqual(len(self.sequences), NUM_SUBSEQUENCES)

            for i in self.sequences:
                self.assertTrue(len(i) < MAX_SUBSEQ_LENGTH)
                self.assertTrue(len(i) > MIN_SUBSEQ_LENGTH)

            consolidated = []
            for i in self.sequences:
                consolidated.extend(i)

            self.assertEqual(list(range(1,len(consolidated)+1)), consolidated)

        def test_length(self):
            sa =  Sequence_Aggregator(self.sequences)

            consolidated = []
            for i in self.sequences:
                consolidated.extend(i)

            self.assertEqual(len(consolidated), len(sa))


        def test_simple_aggregation(self):
            sa =  Sequence_Aggregator(self.sequences)

            target = range(1, len(sa) + 1)

            for i in range(len(sa)):
                self.assertEqual(sa[i], target[i])

        def test_full_length_indexing(self):
            sa =  Sequence_Aggregator(self.sequences)

            for i in range(len(sa)):
                _ = sa[i]

        def test_iteration(self):
            sa =  Sequence_Aggregator(self.sequences)

            target = range(1, len(sa) + 1)

            for i,x in enumerate(sa):
                self.assertEqual(x, target[i])


        def test_iter(self):
            sa =  Sequence_Aggregator(self.sequences)
            it = iter(sa)

            for i in range(len(sa)):
                _ = next(it)
            

    unittest.main()
