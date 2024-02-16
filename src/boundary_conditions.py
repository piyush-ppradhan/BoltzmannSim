import numpy as np
import jax.numpy as jnp
from jax import jit, device_count
from functools import partial

class BoundaryCondition(object):
    """
    Definition of boundary conditions for the Lattice Boltzmann Method.

    Implemented Boundary Condition:
        1. DoNothing
        2. Zou-He Pressure and Velocity BC
        3. Halfway Bounce-back BC

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
            Indices of the boundary nodes. Stored as array of tuple.
        is_solid: bool 
            Whether the boundary condition is defined for a solid boundary.
        is_dynamic: bool
            Whether the boundary condition is dynamic i.e., it changes over time. For example, moving wall boundary condition.
        implementation_step: str
            Define when the boundary condition is applied. Possible values: "pre_collision", "post_collision". Set in sub-class
        needs_extra_configuration: bool
            Whether the boundary condition needs extra information (for example, the velocity boundary condition). Set in sub-class
    """
    def __init__(self, lattice, boundary_indices, precision, nx, ny, nz=0):
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
    
    def create_local_mask_and_normal_arrays(self, grid_mask):
        """
        Creates local mask and normal arrays for the boundary condition, based on the grid mask.
        If extra configuration is necessary, the `configure` method is called.

        Arguments: 
            grid_mask : Array-like
                The grid mask for the lattice.

        Returns:
            None
        """
        if self.needs_extra_configuration:
            boundary_mask = self.get_boundary_mask(grid_mask)
            self.configure(boundary_mask)
            self.needs_extra_configuration = False

        boundary_mask = self.get_boundary_mask(grid_mask)
        self.normals = self.get_normals(boundary_mask)
        self.imissing, self.iknown = self.get_missing_indices(boundary_mask)
        self.imissing_mask, self.iknown_mask, self.imiddle_mask = self.get_missing_mask(boundary_mask)

        return

    def get_boundary_mask(self, grid_mask):  
        """
        Add jax.device_count() to the self.indices in x-direction, and 1 to the self.indices other directions
        This is to make sure the boundary condition is applied to the correct nodes as grid_mask is
        expanded by (jax.device_count(), 1, 1)

        Parameters:
            grid_mask : array-like
                The grid mask for the lattice.
        
        Returns:
            boundaryMask : array-like
        """   
        shifted_indices = np.array(self.boundary_indices)
        shifted_indices[0] += device_count() # For single device implementation, there is no change to the grid_mask 
        shifted_indices[1:] += 1

        shifted_indices = tuple(shifted_indices)
        boundary_mask = np.array(grid_mask[shifted_indices])

        return boundary_mask

    def configure(self, boundary_mask):
        """
        Configures the boundary condition.

            Parameters:
                boundary_mask : array-like
                    The grid mask for the boundary voxels.

            Returns:
                None 

            This method should be overridden in subclasses if the boundary condition requires extra configuration.
        """
        pass

    @partial(jit, static_argnums=(0, 3), inline=True)
    def prepare_populations(self, fout, fin, implementation_step):
        """
        Prepares the distribution functions for the boundary condition. Defined in sub-class.

        Parameters:
            fout : jax.numpy.ndarray
                The incoming distribution functions.
            fin : jax.numpy.ndarray
                The outgoing distribution functions.
            implementation_step : str
                The step in the lattice Boltzmann method algorithm at which the preparation is applied.

        Returns:
            fout: jax.numpy.ndarray
                The prepared distribution functions.
        """   
        return fout

    def get_normals(self, boundary_mask):
        """
        Calculates the normal vectors at the boundary nodes.

        Parameters:
            boundary_mask : array-like
                The boundary mask for the lattice.

        Returns:
            normals: array-like
                The normal vectors at the boundary nodes.

        This method calculates the normal vectors by dotting the boundary mask with the main lattice directions.
        """
        main_c = self.lattice.c.T[self.lattice.main_indices]
        m = boundary_mask[..., self.lattice.main_indices]
        normals = -np.dot(m, main_c)
        return normals

    def get_missing_indices(self, boundary_mask):
        """
        Returns two int8 arrays the same shape as boundary_mask. The non-zero entries of these arrays indicate missing
        directions that require BCs (imissing) as well as their corresponding opposite directions (iknown).

        Parameters:
            boundary_mask : array-like
                The boundary mask for the lattice.

        Returns:
            tuple of array-like
                The missing and known indices for the boundary condition.

        This method calculates the missing and known indices based on the boundary mask. The missing indices are the
        non-zero entries of the boundary mask, and the known indices are their corresponding opposite directions.
        """
        # Find imissing, iknown 1-to-1 corresponding indices
        # Note: the "zero" index is used as default value here and won't affect BC computations
        nbd = len(self.boundary_indices[0])
        imissing = np.vstack([np.arange(self.lattice.q, dtype='uint8')] * nbd)
        iknown = np.vstack([self.lattice.opp_indices] * nbd)
        imissing[~boundary_mask] = 0
        iknown[~boundary_mask] = 0
        return imissing, iknown

    def get_missing_mask(self, boundary_mask):
        """
        Returns three boolean arrays the same shape as boundary_mask. Useful for reduction (eg. summation) operators of selected q-directions.

        Parameters:
            boundary_mask : array-like
                The boundary mask for the lattice.

        Returns:
            tuple of array-like
                The missing, known, and middle masks for the boundary condition.

        This method calculates the missing, known, and middle masks based on the boundary mask. The missing mask is the boundary mask, the known mask is the opposite directions of the missing mask, 
        and the middle mask is the directions that are neither missing nor known.
        """
        imissing_mask = boundary_mask
        iknown_mask = imissing_mask[:, self.lattice.opposite_indices]
        imiddle_mask = ~(imissing_mask | iknown_mask)
        return imissing_mask, iknown_mask, imiddle_mask


    @partial(jit, static_argnums=(0,))
    def equillibrium(self, rho, u):
        """
        Compute the equillibrium distribution for given density (rho) and velocity (u) values.
        Used for applying the Zou-He boundary condition.

        Returns:
            feq: Array-like
                The equillibrium distribution.
        """
        e = self.lattice.e
        w = self.lattice.w

        udote = jnp.dot(u,e)
        udotu = jnp.square(u)
        feq = w * rho * (1.0 + 3.0 * udote + 4.5 * udote**2 - 1.5 * udotu)
        return feq

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin, timestep, implementation_step):
        """
        Apply boundary condition to the distribution f. Defined in subclass.

        Arguments:
            fout: array[float]
                Output distribution array
            fin: array[float]
                Input distribution array
            timestep: int
                Timestep used for calculating values for dynamic boundary condition.
            implementation_step: str
                Implementation step where the boundary condition will be applied. Possible values: "post_collision", "post_streaming".
        """
        pass

class EquilibriumBC(BoundaryCondition):
    """
    Equilibrium boundary condition for a lattice Boltzmann method simulation.

    This class implements an equilibrium boundary condition, where the distribution function at the boundary nodes is
    set to the equilibrium distribution function. The boundary condition is applied after the streaming step.

    Attributes:
        name : str
            The name of the boundary condition. For this class, it is "EquilibriumBC".
        implementationStep : str
            The step in the lattice Boltzmann method algorithm at which the boundary condition is applied. For this class,
            it is "PostStreaming".
        out : jax.numpy.ndarray
            The equilibrium distribution function at the boundary nodes.
    """

    def __init__(self, indices, gridInfo, precision_policy, rho, u):
        super().__init__(indices, gridInfo, precision_policy)
        self.out = self.precisionPolicy.cast_to_output(self.equilibrium(rho, u))
        self.name = "EquilibriumBC"
        self.implementationStep = "PostStreaming"

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin):
        """
        Applies the equilibrium boundary condition.

        Parameters:
            fout : jax.numpy.ndarray
                The output distribution functions.
            fin : jax.numpy.ndarray
                The input distribution functions.

        Returns:
            jax.numpy.ndarray
                The modified output distribution functions after applying the boundary condition.

        Notes:
            This method applies the equilibrium boundary condition by setting the output distribution functions at the
            boundary nodes to the equilibrium distribution function.
        """
        return self.out

class DoNothing(BoundaryCondition):
    """
    DoNothing makes no changes to the values of distribution at the boundary_indices and returns them as is. 

    Attributes:
        None
    """
    def __init__(self, lattice, indices, precision, nx, ny, nz=0):
        super().__init__(lattice, indices, precision, nx, ny, nz)
        self.is_dynamic = False
        self.is_solid = False
        self.needs_extra_configuration = False
        self.implementation_step = "post_streaming"

    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin, timestep, implementation_step):
        """
        Applies no boundary condition to the provided boundary nodes. 

        Arguments:
            fout: Array-like
                Output distribution.
            fin: Array-like
                Input distribution.
            timestep: int
                Timestep used for calculating dynamics BCs. Not used in this function.
            implementation_step: str
                Implementation to apply the boundary condition.
        """
        return fin[self.boundary_indices]

class HalfwayBounceBack(BoundaryCondition):
    """
    Implement the half-way bounce-back boundary condition to simulate a stationary wall.
    
    Attributes:
        None
    """
    def __init__(self, lattice, indices, precision, nx, ny, nz=0):
        super().__init__(lattice, indices, precision, nx, ny, nz)
        self.is_dynamic = False
        self.is_solid = True
        self.needs_extra_configuration = False
        self.implementation_step = "post_collision"

    @partial(jit, static_argnums=(0,))
    def apply(self,fout,fin):
        return fin[self.boundary_indices][..., self.lattice.opposite_indices]

class ZouHe(BoundaryCondition):
    """
    Implement the Zou-He pressure and velocity boundary condition.
    Zou, Qisu, and Xiaoyi He. “On Pressure and Velocity Boundary Conditions for the Lattice Boltzmann BGK Model.” 
    Physics of Fluids 9, no. 6 (June 1, 1997): 1591-98. https://doi.org/10.1063/1.869307.
    """
    def __init__(self, bc_density, lattice, indices, precision, nx, ny, nz=0):
        super().__init__(lattice,indices,precision,nx,ny,nz)
        self.is_solid = False
        self.is_dynamic = False
        self.needs_extra_configuration = True
    
    @partial(jit, static_argnums=(0,))
    def apply(self, fout, fin):
        """
        Apply boundary condition to the distribution f. Defined in subclass.

        Arguments:
            fout: array[float]
                Output distribution array
            fin: array[float]
                Input distribution array
        """
        pass