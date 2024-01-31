import jax.numpy as jnp
import numpy as np
from base import *

class BGK(LBMBase):
    """
        Bhatnagar-Gross-Krook (BGK) approximation for collision step of the Lattice Boltzmann Method.

        Attributes:
            tau: float
                Relaxation parameter (non-dimensional)
    """
    def __init__(self,tau):
        self.tau = tau
        self.omega = 1.0 / self.tau

    @partial(jit, static_argnums=(0,3), donate_argnums=(1,))
    def collision(self,f,rho,u):
        feq = self.equillibrium(rho,u)
        f = f.at[:].set(f + self.omega*(feq - f))
        return f

#TODO
class MRT(LBMBase):
    """
        Multi-Relaxation Time (MRT) model for the collision step of the Lattice Botlzmann Method
        Coveney et.al “Multiple–Relaxation–Time Lattice Boltzmann Models in Three Dimensions.” 
        https://doi.org/10.1098/rsta.2001.0955.

        Attributes:
            tau: float
                Relaxation parameter (non-dimensional)
    """
    def __init__(self,tau):
        self.tau = tau
       
