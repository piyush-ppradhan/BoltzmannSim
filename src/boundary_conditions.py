import jax.numpy as jnp
from jax import jit
from functools import partial

class BoundaryCondition(object):
    """
        Definition of boundary conditions for the Lattice Boltzmann Method.

        Implemented Boundary Condition:
            1. DoNothing
            2. Zou-He Pressure and Velocity BC
            3. Halfway-Bounceback BC

        Attributes:
            lattice: Lattice
                lattice used in the simulation. Defined using the Lattice class, as described in lattice.py
            nx: int
                Number of grid points in the x-direction
            ny: int
                Number of grid points in the y-direction
            nz: int
                Number of grid points in the z-direction
            d: int
                Dimension of the problem
            precision: str
                Precision to be used for computation. Possible values: "f16", "f32", "f64". Default: "f32"
            boundary_indices: array[tuple]
                Indices of the boundary nodes. Stored as array of tuple
            is_solid: bool
                Whether the boundary condition is defined for a solid boundary
            implementation_step: str
                Define when the boundary condition is applied. Possible values: "pre_collision", "post_collision". Defined in subclass
    """
    def __init__(self,lattice,boundary_indices,precision,nx,ny,nz=0):
        self.lattice = lattice
        self.boundary_indices = boundary_indices
        self.nx = nx
        self.ny = ny
        self.nz = nz
        match(precision):
            case "f16":
                self.precision = jnp.float16
            case "f32":
                self.precision = jnp.float32
            case "f64":
                self.precision = jnp.float64
            case _:
                ValueError("Invalid precision type. Valid precision type are: \"f16\", \"f32\", \"f64\"")
    
    @partial(jit, static_argnums=(0,))
    def apply(self,fout,fin):
        """
            Apply boundary condition to the distribution f. Defined in subclass.

            Arguments:
                fout: array[float]
                    Output distribution array
                fin: array[float]
                    Input distribution array
        """
        pass

class DoNothing(BoundaryCondition):
    def __init__(self,lattice,indices,precision,nx,ny,nz=0):
        super().__init__(lattice,indices,precision,nx,ny,nz)

    @partial(jit, static_argnums=(0,))
    def apply(self,fout,fin):
        return fin

class HalfwayBounceBack(BoundaryCondition):
    """
        Implement the half-way bounce-back boundary condition to simulate a stationary wall.
        
        Attributes:
            None
    """
    def __init__(self,lattice,indices,precision,nx,ny,nz=0):
        super().__init__(lattice,indices,precision,nx,ny,nz)

    @partial(jit, static_argnums=(0,))
    def apply(self,fout,fin):
        return fin.at[self.boundary_indices].set(fin[self.boundary_indices,self.lattice.opposite_indices])

class ZouHe(BoundaryCondition):
    """
        Implement the Zou-He pressure and velocity boundary condition.
        Zou, Qisu, and Xiaoyi He. “On Pressure and Velocity Boundary Conditions for the Lattice Boltzmann BGK Model.” 
        Physics of Fluids 9, no. 6 (June 1, 1997): 1591–98. https://doi.org/10.1063/1.869307.
    """
    def __init__(self,bc_density,lattice,indices,precision,nx,ny,nz=0):
        super().__init__(lattice,indices,precision,nx,ny,nz)
    
    @partial(jit, static_argnums=(0,))
    def apply(self,fout,fin):
        """
            Apply boundary condition to the distribution f. Defined in subclass.

            Arguments:
                fout: array[float]
                    Output distribution array
                fin: array[float]
                    Input distribution array
        """
        pass
