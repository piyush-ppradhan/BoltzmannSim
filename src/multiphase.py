"""
Definition of Multiphase class for defining and running a multiphase problem
"""

# Standard libraries
import time

# Third-party imports
from functools import partial
import numpy as np
from termcolor import colored
import jax
import jmp
import orbax.checkpoint as orb
import jmp
 
# JAX-specific imports
import jax.numpy as jnp
from jax import jit, lax, vmap, config
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding, NamedSharding, PartitionSpec, Mesh
from jax.experimental.shard_map import shard_map
from jax.experimental.multihost_utils import process_allgather

# Locally defined functions import
from base import LBMBase
from lattice import *
from utilities import write_vtk, downsample_field

class Multiphase(LBMBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    