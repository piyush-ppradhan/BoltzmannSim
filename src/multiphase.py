"""
Definition of Multiphase class for defining and running a multiphase problem
"""

# System libraries
from functools import partial
import inspect

# Third-party libraries
from jax import jit
import jax.numpy as jnp

# User-defined libraries
from src.base import LBMBase
from src.conversion_parameters import *

class SCMP(LBMBase):
    """
    Single Component Multiphase (SCMP) model, based on the Shan-Chen method. To model the fluid, an equation of state (EOS) is defined by the user.
    EOS is then used to compute the pressure, and ultimately, the interaction potential (phi).
    
    Reference:
        1. Shan, Xiaowen, and Hudong Chen. “Lattice Boltzmann Model for Simulating Flows with Multiple Phases and Components.” 
           Physical Review E 47, no. 3 (March 1, 1993): 1815-19. https://doi.org/10.1103/PhysRevE.47.1815.

        2. Yuan, Peng, and Laura Schaefer. “Equations of State in a Lattice Boltzmann Model.” 
           Physics of Fluids 18, no. 4 (April 3, 2006): 042101. https://doi.org/10.1063/1.2187070.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.R = self.conv_param.to_lattice_units(kwargs.get("gas_constant"))
        self.T = self.conv_param.to_lattice_units(kwargs.get("temperature"))
        self.G = kwargs.get("G")

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, value):
        caller_frame = inspect.stack()[1]
        caller_class = caller_frame[0].f_locals.get('self', None).__class__
        if caller_class != "ShanChen_SCMP":
            if value is None:
                raise ValueError("Gas constant value must be provided")
        self._R = value

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        caller_frame = inspect.stack()[1]
        caller_class = caller_frame[0].f_locals.get('self', None).__class__
        if caller_class != "ShanChen_SCMP":
            if value is None:
                raise ValueError("Temperature value must be provided")
        self._T = value

    @property
    def G(self):
        return self._G

    @G.setter
    def G(self, value):
        if value is None:
            raise ValueError("G value must be provided")
        self._G = value

    @partial(jit, static_argnums=(0,))
    def EOS(self, rho):
        """
        Define the equation of state for the problem. Defined in sub-classes.

        Arguments:
            rho: jax.numpy.ndarray
                Density values

        Returns:
            p: jax.numpy.ndarray
        """
        pass

    @partial(jit, static_argnums=(0,))
    def compute_psi(self, rho):
        """
        Compute psi, the effective mass which is used for modelling the interaction of forces.
        This function uses the value of pressure obtained from EOS and density to compute psi.

        Arguments:
            rho: jax.numpy.ndarray
                Density value at all grid points.

        Returns:
            psi: jax.numpy.ndarray
        """
        rho = self.precision_policy.cast_to_compute(rho)
        p = self.EOS(rho)
        psi = (2 * (p - self.lattice.c_s2 * rho) / (self.lattice.c0 * self.G)) ** 0.5
        return psi

    @partial(jit, static_argnums=(0,))
    def streamed_psi(self, psi):
        """
        This function computes the streamed effective mass (psi) used in the calculation of force.
        psi_s = psi(x + e*dt), e is the lattice velocity directions

        Arguments:
            psi: jax.numpy.ndarray
                Effective mass array

        Returns:
            psi_s: jax.numpy.ndarray
                Streamed effective mass
        """
        psi_s = self.streaming(psi, self.e.T)
        return psi_s

    @partial(jit, static_argnums=(0,))
    def compute_force(self, psi):
        psi_s = self.streamed_psi(psi)
        f_int = self.G * psi * jnp.sum(jnp.dot(psi_s, self.e.T), axis=-1, keepdims=True)
        return f_int


class ShanChen_SCMP(SCMP):
    """
    Define the SCMP model using the original Shan-Chen EOS. For this class compute_psi is redefined.
    For this case, there is no need to define R and T as they are not used in the EOS.

    Attributes:
        rho_0: float
            rho_0 used for computing the effective mass (psi)

    Reference:
        1. Shan, Xiaowen, and Hudong Chen. “Lattice Boltzmann Model for Simulating Flows with Multiple Phases and Components.” 
           Physical Review E 47, no. 3 (March 1, 1993): 1815-19. https://doi.org/10.1103/PhysRevE.47.1815.

    Notes:
        The expression for psi in this case is:
        psi = rho_0 * (1 - exp(-rho / rho_0))
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rho_0 = kwargs.get("rho_0")

    @property
    def rho_0(self):
        return self._rho_0

    @rho_0.setter
    def rho_0(self, value):
        if value is None:
            raise ValueError("rho_0 value must be provided Shan-Chen EOS")
        self._rho_0 = value
   
    @partial(jit, static_argnums=(0,))
    def compute_psi(self, rho):
        psi = self.rho_0 * (1.0 - jnp.exp(-rho / self.rho_0))
        return psi

    @partial(jit, static_argnums=(0,))
    def EOS(self, rho):
        psi = self.compute_psi(rho)
        p = self.lattice.c_s2 * rho + (0.5 * self.lattice.c0 * self.G * psi**2)
        return p

class Redlich_Kwong_SCMP(SCMP):
    """
    Define SCMP model using the Redlich-Kwong EOS.

    Attributes:

    Reference:
        1. Redlich O., Kwong JN., "On the thermodynamics of solutions; an equation of state; fugacities of gaseous solutions."
        Chem Rev. 1949 Feb;44(1):233-44. https://doi.org/10.1021/cr60137a013. PMID: 18125401.

    Notes:
        EOS is given by:
            p = (rho*R*T)/(1 - b*rho) - (a*rho^2)/(sqrt(T) * (1 + b*rho))
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value is None:
            raise ValueError("a value must be provided for using Redlich-Kwong EOS")
        self._a = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if value is None:
            raise ValueError("b value must be provided for using Redlich-Kwong EOS")
        self._b = value
   
    @partial(jit, static_argnums=(0,))
    def EOS(self, rho):
        rho = self.precision_policy.cast_to_compute(rho)
        p = (rho * self.R * self.T)/(1.0 - self.b*rho) - (self.a * rho**2)/(self.T**0.5 * (1.0 + self.b*rho))
        return p

class Redlich_Kwong_Soave_SCMP(SCMP):
    """
    Define SCMP model using the Redlich-Kwong-Soave EOS.

    Attributes:

    Reference:
        1. Giorgio Soave, "Equilibrium constants from a modified Redlich-Kwong equation of state",
        Chemical Engineering Science 27, no. 6(1972), 1197-1203, https://doi.org/10.1016/0009-2509(72)80096-4.

    Notes:
        EOS is given by:
            p = (rho*R*T)/(1 - b*rho) - (a*alpha*rho^2)/(1 + b*rho)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")
        self.alpha = kwargs.get("alpha")

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value is None:
            raise ValueError("a value must be provided for using Redlich-Kwong-Soave EOS")
        self._a = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if value is None:
            raise ValueError("b value must be provided for using Redlich-Kwong-Soave EOS")
        self._b = value

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value is None:
            raise ValueError("alpha value must be provided for using Redlich-Kwong EOS")
        self._alpha = value
   
    @partial(jit, static_argnums=(0,))
    def EOS(self, rho):
        rho = self.precision_policy.cast_to_compute(rho)
        p = (rho * self.R * self.T)/(1.0 - self.b*rho) - (self.a * self.alpha * rho**2)/(1.0 + self.b*rho)
        return p

class Peng_Robinson_SCMP(SCMP):
    """
    Define SCMP model using the Peng-Robinson EOS.

    Attributes:

    Reference:
        1. Peng, Ding-Yu, and Donald B. Robinson. "A new two-constant equation of state." 
        Industrial & Engineering Chemistry Fundamentals 15, no. 1 (1976): 59-64. https://doi.org/10.1021/i160057a011

    Notes:
        EOS is given by:
            p = (rho*R*T)/(1 - b*rho) - (a*alpha*rho^2)/(1 + 2*b*rho - (b*rho)**2)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")
        self.alpha = kwargs.get("alpha")

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value is None:
            raise ValueError("a value must be provided for using Peng-Robinson EOS")
        self._a = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if value is None:
            raise ValueError("b value must be provided for using Peng-Robinson EOS")
        self._b = value

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value is None:
            raise ValueError("alpha value must be provided for using Peng-Robinson EOS")
        self._alpha = value
   
    @partial(jit, static_argnums=(0,))
    def EOS(self, rho):
        rho = self.precision_policy.cast_to_compute(rho)
        p = (rho * self.R * self.T)/(1.0 - self.b*rho) - (self.a * self.alpha * rho**2)/(1.0 + 2*self.b*rho - self.b**2 * rho**2)
        return p

class Carnahan_Starling_SCMP(SCMP):
    """
    Define SCMP model using the Carnahan-Starling EOS.

    Attributes:

    Reference:
        1.  Carnahan, Norman F., and Kenneth E. Starling. "Equation of state for nonattracting rigid spheres." 
        The Journal of chemical physics 51, no. 2 (1969): 635-636. https://doi.org/10.1063/1.1672048

    Notes:
        EOS is given by:
            p = (rho*R*T)/(1 - b*rho) - (a*alpha*rho^2)/(1 + 2*b*rho - (b*rho)**2)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value is None:
            raise ValueError("a value must be provided for using Carnahan-Starling EOS")
        self._a = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if value is None:
            raise ValueError("b value must be provided for using Carnahan-Starling EOS")
        self._b = value

    @partial(jit, static_argnums=(0,))
    def EOS(self, rho):
        rho = self.precision_policy.cast_to_compute(rho)
        p = rho * self.R * self.T * (1.0 + 0.25*self.b*rho + (0.25*self.b*rho)**2 - (0.25*self.b*rho)**3)/(1.0 - 0.25*self.b*rho)**3 - self.a*rho**2
        return p



        
