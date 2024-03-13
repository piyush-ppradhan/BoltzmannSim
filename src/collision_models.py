import numpy as np
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

# Write the matrix for MRT collision model for D2Q9 lattice
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

    def generate_matrix(self):
        """
        Generate the transformation matrix for the MRT collision model.

        Arguments:
            None
        """
        match self.lattice.name:
            case "D2Q9":
                M = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                               [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                               [4, -2, -2, -2, -2, 1, 1, 1, 1],
                               [0, 1, 0, -1, 0, 1, -1, -1, 1],
                               [0, -2, 0, 2, 0, 1, -1, -1, 1],
                               [0, 0, 1, 0, -1, 1, 1, -1, -1],
                               [0, 0, -2, 0, 2, 1, 1, -1, -1],
                               [0, 1, -1, 1, -1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 1, -1, 1, -1]])
                invM = np.linalg.inv(M)
                S = np.diag([])
            case "D3Q19":
                # Write the transformation matrix for D3Q19 lattice
                M =  np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                               [-1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                               [1, -2, -2, -2, -2, -2, -2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                               [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0], 
                               [0, -2, 2, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0], 
                               [0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1], 
                               [0, 0, 0, -2, 2, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1], 
                               [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1], 
                               [0, 0, 0, 0, 0, -2, 2, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1], 
                               [0, 2, 2, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -2, -2, -2, -2], 
                               [0, -2, -2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -2, -2, -2, -2], 
                               [0, 0, 0, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0], 
                               [0, 0, 0, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0], 
                               [0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0], 
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1], 
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0], 
                               [0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, -1, 1, -1, 1, 0, 0, 0, 0], 
                               [0, 0, 0, 0, 0, 0, 0, -1, 1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1], 
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, -1, 1, 1, -1]])                            
                invM = np.linalg.inv(M)
                s1 = self.omega
                s2 = 8.0*(2.0 - s1)/(8.0 - s1)
                S = np.diag([0, s1, s1, 0, s2, 0, s2, 0, s2, s1, s1, s1, s1, s1, s1, s1, s2, s2, s2])
            case "D3Q27":
                M = np.array([[]])
                invM = np.linalg.inv(M)
            case _:
                raise ValueError("Lattice not supported")

        return M, invM, S
