import jax.numpy as jnp
from jax import jit
from functools import partial
from conversion_parameters import *

class BodyForce(object):
    """
        Base class to define the volumetric forces that are applied during the LBM simulation

        Attributes:
            F: array[float]
                Volumetric body forces defined in SI units. To convert the body force to Lattice Units, CoversionParameters are used internally
            implementation_step: str
                Defines the step where the boundary condition is applied. Possible values: "none", "distribution", "macroscopic". Value is set in sub-class.
                "none": No changes to the distribution and macroscopic flow variable.
                "distribution": Changes to the distribution values. (For Guo force model).
                "macroscopic": Changes the macroscopic values for incorporating the body force. (For Shan-Chen force model)
    """
    def __init__(self,F,implementation_step):
        self.F = F
        self.implementation_step = implementation_step

    def convert_to_lattice_units(self,conv_param):
        """
            Convert the force in SI units to lattice units 
        """
        self.F = F / (self.C_rho * (self.C_l**4) * (self.C_t**0.5))
    
    def apply(self,f,rho,u,step,precision):
        """
            Apply the body force to the respective data depending on the force model used. Defined in sub-class.
            
            Arguments:
                f: array[float]
                    Distribution values at all grid points.
                rho: array[float]
                    Density at all grid points.
                u: array[float]
                    Velocity at all grid points.
                step: str
                    The current step of applying body force. It is compared with implementation_step of the force model to accordingly modify the velocity or the distribution
        """
        pass

class NoBodyForce(BodyForce):
    def __init__(self):
        super().__init__([0.0,0.0,0.0],"none") # Actual dimension of the force array does not matter, it is simply ignored
    
    @partial(jit, static_argnums=(0,3))
    def apply(self,f,rho,u):
        return f, rho, u

class ShanChenBodyForce(BodyForce):
    """
        Implementation of body force using Shan-Chen method, as described in “Lattice Boltzmann Model for Simulating Flows with Multiple Phases and Components.” 
        Physical Review E 47, no. 3 (March 1, 1993): 1815–19. https://doi.org/10.1103/PhysRevE.47.1815.

        The body force F is applied by modifying the velocity, after calculating it using the distribution:
        u_new = u + F*dt_star / rho
    """
    def __init__(self,F):
        super().__init__(F,"macroscopic")

    @partial(jit, static_argnums=(0,2), donate_argnums=(3,))
    def apply(self,f,rho,u,step,precision=jnp.float32):
        """
            Apply the body force to the velocity as per the Shan-Chen method.
            
            Arguments:
                f: array[float] 
                    Distribution values at all grid points.
                rho: array[float]
                    Density values at all grid points.
                u: array[float]
                    Velocity at all grid points.
                step: str
                    The current step of applying body force. It is compared with implementation_step of the force model to accordingly modify the velocity or the distribution.
                precision: type
                    Precision used for computation.
        """
        if self.step == self.implementation_step:
            F = jnp.array(self.F,dtype=precision)
            u = u + F / rho

        return f, rho, u

#TODO
class GuoBodyForce(BodyForce):
    """
        Implementation of body force using Guo et. al's method, as described in "Lattice Boltzmann Model for Incompressible Flows through Porous Media."
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
    def __init__(self,F,implementation_step,**kwargs):
        super().__init__(F,"distribution")
        self.A = kwargs.get("A",0.0)
        self.B = kwargs.get("B",F)
        self.C = kwargs.get("C")

    @partial(jit, static_argnums=(0,4), donate_argnums=(1,))
    def apply(self,f,rho,u,precision):
        """
            Apply the body force to the velocity as per the Guo method.
            
            Arguments:
                f: array[float]
                    Distribution values at all grid points.
                rho: array[float]
                    Density values at all grid points.
                u: array[float]
                    Velocity at all grid points.
        """
        pass

