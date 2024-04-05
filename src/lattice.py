import re

import jax.numpy as jnp
import numpy as np
from jax import config


class Lattice(object):
    """
    Parent class to define the lattice used for simulation.

    Attributes:
        d: int
            Dimension of the problem
        q: int
            Number of discrete directions used in the lattice name
        name: str
            Lattice name used in the simulation. "D2Q9", "D3Q19", "D3Q27"
        precision: str
            Precision for computation and exporting data. "f16", "f32", "f64". Default: "f32"
        w: array[float]
            Weights used for LBM calculation
        e: array[int]
            Discrete velocity directions present in the name
        main_indices: array[int]
            Index of main directions in the discrete velocity set.
        oppsite_indices: array[int]
            Opposite direction for each discrete velocity direction, useful for applying bounce-back boundary conditions
    """
    def __init__(self, name, precision):
        self.name = name
        match(precision):
            case "f16":
                self.precision = jnp.float16
            case "f32":
                self.precision = jnp.float32
            case "f64":
                self.precision = jnp.float64
                config.update("jax_enable_x64", True)
            case _:
                ValueError("Invalid precision type. Supported precision: f16, f32, f64")
                exit()
        dq = re.findall(r"\d+", name)
        self.d = int(dq[0])
        self.q = int(dq[1])
        self.e = jnp.array(self.construct_velocity_directions(),dtype=self.precision)
        self.w = jnp.array(self.construct_lattice_weights(),dtype=self.precision)
        self.main_indices = self.construct_main_velocity_indices()
        self.opposite_indices = self.construct_opposite_directions_indices()
        self.left_indices = jnp.array(self.construct_left_indices(), dtype=jnp.int8)
        self.right_indices = jnp.array(self.construct_right_indices(), dtype=jnp.int8)
        self.ee = jnp.array(self.construct_lattice_moments(), dtype=jnp.int8)
    
    def construct_velocity_directions(self):
        """
        Compute discrete velocity directions for a given lattice name.
        
        Arguments:
            None

        Returns:
            e: array[int]
                Discrete velocity directions for given lattice.
        """
        match(self.name):
            case "D2Q9":
                ex = [0,1,0,-1,0,1,-1,-1,1]
                ey = [0,0,1,0,-1,1,1,-1,-1]
                e = np.array([tuple(zip(ex,ey))]).squeeze()
            case "D3Q19":
                e = np.array([(x,y,z) for x in [0,-1,1] for y in [0,-1,1] for z in [0,-1,1]])
                e = e[np.linalg.norm(e, axis=1) < 1.45]
            case "D3Q27":
                e = np.array([(x,y,z) for x in [0, -1, 1] for y in [0, -1, 1] for z in [0, -1, 1]])
            case _:
                raise ValueError("Invalid lattice name. Supported lattice name: D2Q9, D3Q19, D3Q27")
        return e.T

    def construct_main_velocity_indices(self):
        """
        Compute the index of the main directions in the discrete velocity set. 

        Arguments:
            None
        
        Returns:
            main_indices: array[int]
                Indices of the main directions in the discrete velocity set.
        """
        e = np.transpose(self.e)
        if self.d == 2:
            main_indices = np.nonzero((np.abs(e[:,0]) + np.abs(e[:,1]) == 1))[0]
        else:
            main_indices = np.nonzero((np.abs(e[:,0]) + np.abs(e[:,1]) + np.abs(e[:,2]) == 1))[0]
        return main_indices

    def construct_opposite_directions_indices(self):
        """
        Compute the index of the opposite direction for each direction in the discrete velocity set.

        Arguments:
            None
        
        Returns:
            array[int]
            Indices for the opposite direction for velocity directions
        """
        e = self.e.T
        opposite = np.array([e.tolist().index((-e[i]).tolist()) for i in range(self.q)])
        return opposite 

    def construct_right_indices(self):
        """
        Construct the indices of the velocities that point in the positive x-direction.

        Returns:
            numpy.ndarray
                The indices of the right velocities.
        """
        e = self.e.T
        return np.nonzero(e[:, 0] == 1)[0]
    
    def construct_left_indices(self):
        """
        Construct the indices of the velocities that point in the negative x-direction.

        Returns:
            numpy.ndarray
                The indices of the left velocities.
        """
        e = self.e.T
        return np.nonzero(e[:, 0] == -1)[0]

    def construct_lattice_weights(self):
        """
        Compute weights for a given lattice name.
        
        Arguments:
            None

        Returns:
            w: array[float]
                Lattice weights used for computation.
        """
        e = self.e.T
        w = (1.0 / 36.0) * np.ones(self.q)
        match(self.name):
            case "D2Q9":
                w[np.linalg.norm(e,axis=1) < 1.1] = 1.0 / 9.0
                w[0] = 4.0 / 9.0
            case "D3Q19":
                w[np.linalg.norm(e, axis=1) < 1.1] = 2.0 / 36.0
                w[0] = 1.0 / 3.0
            case "D3Q27":
                el = np.linalg.norm(e, axis=1)
                w[np.isclose(el, 1.0, atol=1e-08)] = 2.0 / 27.0
                w[(el > 1) & (el  <= np.sqrt(2))] = 1.0 / 54.0
                w[(el > np.sqrt(2)) & (el  <= np.sqrt(3))] = 1.0 / 216.0
                w[0] = 8.0 / 27.0
            case _:
                ValueError("Invalid lattice name. Supported lattice name: D2Q9, D3Q19, D3Q27")
                exit()
        return w
    
    def construct_lattice_moments(self):
        """
        Constructs the moments of the lattice i.e., the products of the velocity vectors.
        Used in the computation of the equilibrium distribution and the collision operator in the Lattice Boltzmann Method (LBM).

        Returns:
            ee: numpy.ndarray
                The moments of the lattice.
        """
        e = self.e.T
        # Counter for the loop
        counter = 0

        # nt: number of independent elements of a symmetric tensor
        nt = self.d * (self.d + 1) // 2

        ee = np.zeros((self.q, nt))
        for a in range(0, self.d):
            for b in range(a, self.d):
                ee[:, counter] = e[:, a] * e[:, b]
                counter += 1

        return ee 

class D2Q9(Lattice):
    """
    D2Q9(precision)
    Lattice definition for D2Q9 lattice derived from the Lattice parent class.

    Attributes:
        d: int = 2
        q: int = 9
        name: str = "D2Q9"
        precision:str = "f16", "f32", "f64" Default: "f32"
        w: Array-like
        e: Array-like
    """
    def __init__(self,precision="f32"):
        super().__init__("D2Q9",precision)
        self.compute_constants()

    def compute_constants(self):
        """
        Compute the speed of sound (c_s) and its square (c_s2).

        Arguments:
            None

        Returns:
            None
        """
        self.c_s = 1.0 / (3.0 ** 0.5)
        self.c_s2 = 1.0 / 3.0

class D3Q19(Lattice):
    """
    D3Q19(precision)
    Lattice definition for D3Q19 lattice derived from the Lattice parent class.

    Attributes:
        d: int = 3
        q: int = 19
        name: str = "D3Q19"
        precision:str = "f16", "f32", "f64" Default: "f32"
        w: array[float]
        e: array[int]
    """
    def __init__(self,precision="f32"):
        super().__init__("D3Q19",precision)
        self.compute_constants()

    def compute_constants(self):
        """
        Compute the speed of sound (c_s) and its square (c_s2).

        Arguments:
            None
        """
        self.c_s = 1.0 / (3.0 ** 0.5)
        self.c_s2 = 1.0 / 3.0

class D3Q27(Lattice):
    """
    D3Q27(precision)
    Lattice definition for D3Q27 lattice derived from the Lattice parent class.

    Attributes:
        d: int = 3
        q: int = 27
        name: str = "D3Q27"
        precision:str = "f16", "f32", "f64" Default: "f32"
        w: array[float]
        e: array[int]
    """
    def __init__(self,precision="f32"):
        super().__init__("D3Q27",precision)
        self.compute_constants()

    def compute_constants(self):
        """
        Compute the speed of sound (c_s) and its square (c_s2).

        Arguments:
            None 
        """
        self.c_s = 1.0 / (3.0 ** 0.5)
        self.c_s2 = 1.0 / 3.0
