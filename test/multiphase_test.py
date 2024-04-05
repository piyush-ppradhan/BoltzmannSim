"""
Testing the multiphase initialization
"""
import os
import numpy as np
from src.lattice import D2Q9
from src.multiphase import Multiphase

def test_multiphase_initialization():
    precision = "f32"
    lattice = D2Q9(precision)

    nx = 200
    ny = 200

    Re = 200.0
    prescribed_vel = 0.1
    clength = nx - 1

    checkpoint_rate = 1000
    checkpoint_dir = os.path.abspath("./checkpoints")

    visc = prescribed_vel * clength / Re
    omega = 1.0 / (3.0 * visc + 0.5)

    kwargs = {
        'lattice': lattice,
        'omega': [omega],
        'n_components': 1,
        'g_kkprime': np.ones((1, 1)),
        'g_kkprime_inv': [1.0],
        'nx': nx,
        'ny': ny,
        'nz': 0,
        'total_timesteps': 5000,
        'write_precision': precision,
        'write_start': 100,
        'write_control': 500,
        'output_dir': "output",
        'print_info_rate': 100,
        'checkpoint_rate': checkpoint_rate,
        'checkpoint_dir': checkpoint_dir,
        'restore_checkpoint': False,
    }

    sim = Multiphase(**kwargs)
