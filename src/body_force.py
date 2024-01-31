import jax.numpy as jnp
from jax import jit
from functools import partial

class BodyForce(object):
    """
        Base class to define the volumetric forces that are applied during the LBM simulation

        Attributes:
            F: array[float]
                Volumetric body forces defined in SI units. To convert the body force to Lattice Units, CoversionParameters are used internally
    """
    def __init__(self,F):
        self.F = F

class NoBodyForce(BodyForce):
    def __init__(self):
        super().__init__([0.0,0.0,0.0]) # Actual dimension of the force array does not matter, it is simply ignored

class ShanChenForce(BodyForce):
    """
        Implementation of body force using Shan-Chen method, as described in “Lattice Boltzmann Model for Simulating Flows with Multiple Phases and Components.” 
        Physical Review E 47, no. 3 (March 1, 1993): 1815–19. https://doi.org/10.1103/PhysRevE.47.1815.

        The body force F is applied by modifying the velocity, after calculating it using the distribution:
        u_new = u + F*dt_star / rho
    """
    def __init__(self,F):
        super().__init__(F)

    @partial(jit, static_argnums=(0,2), donate_argnums=(2,))
    def apply(self,rho,u,precision=jnp.float32):
        """
            Apply the body force to the velocity as per the Shan-Chen method
            
            Arguments:
                 u: array[float]
                    jax.numpy array storing the velocity at all grid points
        """
        F = jnp.array(self.F,dtype=precision)
        u = u + F / rho
        return u

#TODO
class GuoBodyForce(BodyForce):
    """
        Implementation of body force using Guo et. al's method, as described in“ Lattice Boltzmann Model for Incompressible Flows through Porous Media.” 
        Physical Review. E, Statistical, Nonlinear, and Soft Matter Physics 66 (October 1, 2002): 036304. https://doi.org/10.1103/PhysRevE.66.036304.

        The body force is applied by adding deltaf_i to the distribution functions, where:
        
        deltaf_i = w_i * (1 + (B*e_i / c_s^2)  + (C / (2*c_s^4))*(e_i*e_i - c_s^2 * I)) * deltat

        Attributes
            A: float
                Default: 0.0
            B: float
                Default: F
            C: float
                Typical value: (1 - 1/(2*tau_star)) * F
    """
    def __init__(self,F,**kwargs):
        super().__init__(F)
        self.A = kwargs.get("A",0.0)
        self.B = kwargs.get("B",F)
        self.C = kwargs.get("C")

    @partial(jit, static_argnums=(0,1), donate_argnums=(1,))
    def apply(self,f):
        pass
