"""
This functions tests the weights and direction parameters defined in a lattice classes.

The test is very simple:
    1. sum of all weights should be equal to 1. Since weights are floating point numbers,
       approx comparison is performed with tolerance of 1e-6
    2. sum of all directions should be equal to 0. Here exact comparison is performed.

The test is performed for all three lattice classes D2Q9, D3Q19, D3Q27 and for all three precisions f16, f32, f64.
"""
from src.lattice import D2Q9, D3Q19, D3Q27
import os
import numpy as np
import pytest

def test_weights():
    """
    Test the weights of the lattice. Their sum should be equal to 1.0

    Arguments:
        None
    """
    for precision in ["f16", "f32", "f64"]:
        Obj = [D2Q9(precision), D3Q19(precision), D3Q27(precision)]
        for obj in Obj:
            w = np.array(obj.w)
            assert pytest.approx(np.sum(w)) == 1.0

def test_velocity_directions():
    """
    Test the velocity directions of the lattice. Their sum should be equal to 0 exactly for all coordinates.

    Arguments:
        None
    """
    for precision in ["f16", "f32", "f64"]:
        Obj = [D2Q9(precision), D3Q19(precision), D3Q27(precision)]
        for obj in Obj:
            e = np.array(obj.e)
            for i in range(obj.d):
                assert np.sum(e, axis=1)[i] == 0
    os.system("rm -rf ./test/__pycache__")
