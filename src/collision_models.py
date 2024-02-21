import jax
import jax.numpy as jnp
import numpy as np
from src.base import *

class BGK(LBMBase):
    """
    Bhatnagar-Gross-Krook (BGK) approximation for collision step of the Lattice Boltzmann Method.

    Attributes:
        tau: float
            Relaxation parameter (non-dimensional)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, fin):
        fin = self.precision_policy.cast_to_compute(fin)
        rho, u = self.compute_macroscopic_variables(fin)
        feq = self.equilibrium(rho, u, cast_output=False)
        fneq = feq - fin
        fout =  fin + self.omega*fneq
        if self.force is not None:
            fout = self.apply_force(fout, feq, rho, u)
        return self.precision_policy.cast_to_output(fout)

#TODO
class MRT(LBMBase):
    """
    Multi-Relaxation Time (MRT) model for the collision step of the Lattice Botlzmann Method
    Coveney et.al “Multiple-Relaxation-Time Lattice Boltzmann Models in Three Dimensions.” 
    https://doi.org/10.1098/rsta.2001.0955.

    Attributes:
        tau: float
            Relaxation parameter (non-dimensional)
    """
    def __init__(self,tau):
        self.tau = tau
