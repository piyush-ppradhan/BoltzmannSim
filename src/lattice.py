import jax.numpy as jnp
import numpy as np

class Lattice(object):
    """
        Parent class to define the lattice used for simulation.

        Attributes:
            d: int
                Dimension of the problem
            q: int
                Number of discrete directions used in the lattice stencil
            stencil: str
                Lattice stencil used in the simulation. "D2Q9", "D3Q19", "D3Q27"
            precision: str
                Precision for computation and exporting data. "f16", "f32", "f64". Default: "f32"
            w: array[float]
                Weights used for LBM calculation
            e: array[int]
                Discrete velocity directions present in the stencil
            oppsite_indices: array[int]
                Opposite direction for each discrete velocity direction, useful for applying bounce-back boundary conditions
    """
    def __init__(self,stencil,precision):
        self.d = int(stencil[1])
        self.q = int(stencil[3])
        self.stencil = stencil
        match(precision):
            case "f16":
                self.precision = jnp.float16
            case "f32":
                self.precision = jnp.float32
            case "f64":
                self.precision = jnp.float64
            case _:
                ValueError("Invalid precision type. Supported precision: f16, f32, f64")
                exit()
        self.e = jnp.array(self.get_velocity_directions(),dtype=self.precision)
        self.w = jnp.array(self.get_lattice_weights(),dtype=self.precision)
    
    def get_velocity_directions(self):
        """
            Helper function to compute discrete velocity directions for a given lattice stencil.
            
            Arguments:
                None

            Returns:
                e: array[int]
                    Discrete velocity directions for given lattice.
        """
        match(self.stencil):
            case "D2Q9":
                ex = [0,1,0,-1,0,1,-1,-1,1]
                ey = [0,0,1,0,-1,1,1,-1,-1]
                e = np.array([tuple(zip(ex,ey))])
            case "D3Q19":
                e = np.array([(x,y,z) for x in [0,-1,1] for y in [0,-1,1] for z in [0,-1,1]])
                e = e[np.linalg.norm(e, axis=1) < 1.45]
            case "D3Q19":
                e = np.array([(x,y,z) for x in [-1,0,1] for y in [-1,0,1] for z in [-1,0,1]])
            case _:
                ValueError("Invalid lattice stencil. Supported lattice stencil: D2Q9, D3Q19, D3Q27")
                exit()
        return e

    def get_lattice_weights(self):
        """
            Helper function to compute weights for a given lattice stencil.
            
            Arguments:
                None

            Returns:
                w: array[float]
                    Lattice weights used for computation
        """
        e = self.get_velocity_directions()
        match(self.stencil):
            case "D2Q9":
                w = np.zeros(self.q)
                w[np.linalg.norm(e,axis=1) < 1.1] = 1.0 / 9.0
                w[np.linalg.norm(e,axis=1) > 1.1] = 1.0 / 36.0
                w[0] = 4.0 / 9.0
            case "D3Q19":
                w = (1.0 / 36.0) * np.ones(self.q)
                w[np.isclose(np.linalg.norm(e,axis=1),1,atol=1e-9)] = 1.0 / 18.0
                w[np.linalg.norm(e, axis=1) > 1.1] = 1.0 / 36.0
                w[0] = 1.0 / 3.0
            case "D3Q27":
                w = (1.0 / 216.0)*np.ones(self.q)
                w[np.isclose(np.linalg.norm(e,axis=1), 1.0, atol=1e-9)] = 2.0 / 27.0
                w[1.0 < np.linalg.norm(e,axis=1) < 1.45] = 1.0 / 54.0
                w[0] = (8.0 / 27.0)
            case _:
                ValueError("Invalid lattice stencil. Supported lattice stencil: D2Q9, D3Q19, D3Q27")
                exit()
        return w

class D2Q9(Lattice):
    """
        D2Q9(precision)
        Lattice definition for D2Q9 lattice derived from the Lattice parent class.

        Attributes:
            d: int = 2
            q: int = 9
            stencil: str = "D2Q9"
            precision:str = "f16", "f32", "f64" Default: "f32"
            w: array[float]
            e: array[int]
    """
    def __init__(self,precision="f32"):
        super().__init__("D2Q9",precision)
        self.compute_constants

    def compute_constants(self):
        """
            Helper function to compute the speed of sound (c_s) and its square (c_s2).

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
            stencil: str = "D3Q19"
            precision:str = "f16", "f32", "f64" Default: "f32"
            w: array[float]
            e: array[int]
    """
    def __init__(self,precision="f32"):
        super().__init__("D3Q19",precision)
        self.compute_constants()

    def compute_constants(self):
        """
            Helper function to compute the speed of sound (c_s) and its square (c_s2).

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
            stencil: str = "D3Q27"
            precision:str = "f16", "f32", "f64" Default: "f32"
            w: array[float]
            e: array[int]
    """
    def __init__(self,precision="f32"):
        super().__init__("D3Q27",precision)
        self.compute_constants()

    def compute_constants(self):
        """
            Helper functions to compute the speed of sound (c_s) and its square (c_s2).

            Arguments:
                None 
        """
        self.c_s = 1.0 / (3.0 ** 0.5)
        self.c_s2 = 1.0 / 3.0
