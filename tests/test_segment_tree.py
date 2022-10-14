import numpy as np

from minerl3161.utils.segment_tree import SumSegmentTree, MinSegmentTree

def test_sum_segment_tree():
    n_init = 16
    n_add = 5

    sst = SumSegmentTree(n_init)

    for i in range(n_add):
        sample = np.random.rand()
        sst[i] = sample
        assert sst[i] == sample
    
    sst.retrieve(0.5)
    

def test_min_segment_tree():
    n_init = 16
    n_add = 5

    mst = MinSegmentTree(n_init)

    for i in range(n_add):
        sample = np.random.rand()
        mst[i] = sample
        assert mst[i] == sample
    
    mst.min(0, 2)
