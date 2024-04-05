"""
This file tests the checkpointing functionality of jax. 

The test works as follows:
    1. Generate a random array and save it to a file. Keep a numpy copy of the array with the same precision.
    2. Load the array from the file and compare it with the numpy copy.

The test is performed for all three precisions: f16, f32, f64.
"""
import os

import jax
import numpy as np
import orbax.checkpoint as orb


def np_precision(precision):
    """
    Return the numpy precision type based on the input string.

    Arguments:
        precision: str = "f16", "f32", "f64"
    """
    match precision:
        case "f16":
            return np.float16
        case "f32":
            return np.float32
        case "f64":
            return np.float64

def jax_precision(precision):
    """
    Return the jax precision type based on the input string.

    Arguments:
        precision: str = "f16", "f32", "f64"
    """
    match precision:
        case "f16":
            return jax.numpy.float16
        case "f32":
            return jax.numpy.float32
        case "f64":
            return jax.numpy.float64

def test_checkpoint():
    """
    Test the checkpointing functionality of jax.

    Arguments:
        None
    """
    checkpoint_rate = 1
    mngr_options = orb.CheckpointManagerOptions(save_interval_steps=checkpoint_rate, max_to_keep=1)

    np.random.seed(0)
    for d in [2, 3]:
        for n in [10, 50, 100]:
            for precision in ["f16", "f32", "f64"]:
                if precision == "f64":
                    jax.config.update("jax_enable_x64", True)

                if d == 2:
                    shape = (n, n)
                    x = np.random.randn(n, n).astype(np_precision(precision))
                else:
                    shape = (n, n, n)
                    x = np.random.randn(n, n, n).astype(np_precision(precision))

                # Save the array to file
                checkpoint_dir = os.path.abspath("./checkpoints_" + precision + "_" + str(n) + "_" + str(d))
                mngr = orb.CheckpointManager(checkpoint_dir, orb.PyTreeCheckpointer(), options=mngr_options)
                x_jax = jax.numpy.array(x, dtype=jax_precision(precision))
                state = {'x': x_jax}
                mngr.save(0, state)

                # Load the array from the file
                y = jax.numpy.zeros(shape, dtype=jax_precision(precision))
                state = {'x': y}
                shardings = jax.tree_map(lambda x: x.sharding, state)
                restore_args = orb.checkpoint_utils.construct_restore_args(state, shardings)
                y = mngr.restore(mngr.latest_step(), restore_kwargs={'restore_args': restore_args})['x']

                assert jax.numpy.allclose(x, y, atol=1e-6)
    # Clean up
    os.system("rm -rf ./checkpoints*")
    os.system("rm -rf ./test/__pycache__")
