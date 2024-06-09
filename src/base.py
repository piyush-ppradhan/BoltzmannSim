"""
Definition of LBMBase class for defining and running a problem
"""

# Standard libraries
import time
from functools import partial  # Function modifier for identifying the static and traced variables

import jax
# JAX-specific imports
import jax.numpy as jnp
import jmp  # Mixed precision library for JAX
# Third-party imports
import numpy as np
import orbax.checkpoint as orb  # For storing and restoring checkpoints
from jax import config, jit, lax, vmap
from jax.experimental import mesh_utils
from jax.experimental.multihost_utils import process_allgather
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from termcolor import colored  # Colored output to the terminal

# Locally defined functions import
from src.lattice import D2Q9
from src.utilities import downsample_field


class LBMBase(object):
    """
    Base class to define all LBM solvers and problems

    Attributes:
        omega: float
            Inverse of relaxation time parameter tau.
        nx: int
            Number of grid points in the x-direction
        ny: int
            Number of grid points in the y-direction
        nz: int
            Number of grid points in the z-direction. For 2D problem, nz = 0
        lattice: Lattice
            Lattice stencil used for simulation as defined in lattice.py
        compute_precision: str
            Precision for computation and exporting the data. Same as lattice.precision Default: "f32"
        write_precision: str
            Precision for file output. Default: "f32"
        solid_mask: numpy.ndarray
            Mask that identifies all the solid nodes in the grid. Solid node: 0, Fluid node: 1
        boundary_conditions: BoundaryCondition
            Boundary conditions used in the problem, defined as a list of objects of class BoundaryCondition. Default: Empty i.e., no boundary conditions
        create_log: bool
            Log file is saved in current directory, it will overwrite the preci
        output_dir: str
            Directory where output files will be exported. Auto-created if it doesn't exist
        compute_mlups: bool {Optional}
            Compute the Million Lattice Updates per Second (MLUPS). Default: False
        checkpoint_rate: int {Optional}
            Rate at which checkpoint files will be written. Default: 0
        checkpoint_dir : str {Optional}
            Ignored if restart_flag is set to False. Default: ./checkpoints
        restore_checkpoint: bool {Optional}
            The simulation from the latest checkpoint if set to trur. Default: False
        write_start: int
            Timestep at which the export of data is started.
        write_control: int
            Timestep interval after which data is exported, starting from write_start.
        output_dir: str
            Default: ./output
        print_info_rate: int
            Default: 1 (i.e., every timesteps)
        downsampling_factor: int {Optional}
            Default: 1
        return_post_col_dist: bool {Optional}
            Return the post collision distribution values. Default: False
    """

    def __init__(self, **kwargs):
        self.omega = kwargs.get("omega")
        self.nx = kwargs.get("nx")
        self.ny = kwargs.get("ny")
        self.nz = kwargs.get("nz", 0)
        self.lattice = kwargs.get("lattice")
        self.force = kwargs.get("force", 0.0)
        self.compute_precision = self.set_precision(self.lattice.precision)
        self.write_precision = self.set_precision(kwargs.get("write_precision"))
        self.solid_mask = kwargs.get("solid_mask")
        self.total_timesteps = kwargs.get("total_timesteps")
        self.compute_mlups = kwargs.get("compute_MLUPS", False)
        self.checkpoint_rate = kwargs.get("checkpoint_rate", 0)
        self.checkpoint_dir = kwargs.get("checkpoint_dir", "./checkpoints")
        self.restore_checkpoint = kwargs.get("restore_checkpoint", False)
        self.write_start = kwargs.get("write_start")
        self.write_control = kwargs.get("write_control")
        self.output_dir = kwargs.get("output_dir")
        self.print_info_rate = kwargs.get("print_info_rate", 1)
        self.downsampling_factor = kwargs.get("downsampling_factor", 1)
        self.return_post_col_dist = kwargs.get("return_post_col_dist", False)

        self.n_devices = jax.device_count()
        self.backend = jax.default_backend()

        self.d = self.lattice.d
        self.q = self.lattice.q
        self.e = self.lattice.e
        self.w = self.lattice.w

        # Configure JAX to use 64-bit precision if necessary
        if self.compute_precision == jnp.float64:
            config.update("jax_enable_x64", True)
            print(colored("Using 64-bit precision for computation.\n", "yellow"))

        self.precision_policy = jmp.Policy(
            compute_dtype=self._compute_precision,
            param_dtype=self.compute_precision,
            output_dtype=self.write_precision,
        )

        # Set the checkpoint manager
        if self.checkpoint_rate > 0:
            mngr_options = orb.CheckpointManagerOptions(
                save_interval_steps=self.checkpoint_rate, max_to_keep=1
            )
            self.mngr = orb.CheckpointManager(self.checkpoint_dir, options=mngr_options)
        else:
            self.mngr = None

        # Check if nx, ny, nz has been defined appropriately
        if None in {self.nx, self.ny, self.nz}:
            print(
                "Error: at least nx and ny must be provided to perform the simulation"
            )
            exit()
        # Scale nx to be divisible by the number of devices.
        nx = self.nx
        if self.nx % self.n_devices:
            self.nx = nx + (self.n_devices - nx % self.n_devices)

        self.show_simulation_parameters()

        self.grid_info = {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "dim": self.d,
            "lattice": self.lattice,
        }

        P = PartitionSpec
        # Define the left and right permutation to communicate between different GPUs.
        # The tuple for permutation is defined as (source_index, destination_index)
        self.right_perm = [(i, (i + 1) % self.n_devices) for i in range(self.n_devices)]
        self.left_perm = [((i + 1) % self.n_devices, i) for i in range(self.n_devices)]

        # Setting up sharding
        if self.d == 2:
            self.devices = mesh_utils.create_device_mesh((self.n_devices, 1, 1))
            self.mesh = Mesh(self.devices, axis_names=("x", "y", "name"))
            self.sharding = NamedSharding(self.mesh, P("x", "y", "name"))

            self.streaming = jit(
                shard_map(
                    self.streaming_m,
                    mesh=self.mesh,
                    in_specs=P("x", None, None),
                    out_specs=P("x", None, None),
                    check_rep=False,
                )
            )
        elif self.d == 3:
            self.devices = mesh_utils.create_device_mesh((self.n_devices, 1, 1, 1))
            self.mesh = Mesh(self.devices, axis_names=("x", "y", "z", "name"))
            self.sharding = NamedSharding(self.mesh, P("x", "y", "z", "name"))

            self.streaming = jit(
                shard_map(
                    self.streaming_m,
                    mesh=self.mesh,
                    in_specs=P("x", None, None, None),
                    out_specs=P("x", None, None, None),
                    check_rep=False,
                )
            )
        else:
            raise ValueError("Dimension of the problem must be either 2 or 3")

        self.bounding_box_indices = self.bounding_box_indices_()
        self._create_boundary_data()
        self.force = self.get_force()

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value):
        if value is None:
            raise ValueError("Omega must be provided")
        if (
            not isinstance(value, float)
            and not isinstance(value, list)
            and not isinstance(value, jnp.ndarray)
        ):
            raise ValueError(
                "Omega must be float or list of floats or jax.numpy.ndarray"
            )
        self._omega = value

    @property
    def nx(self):
        return self._nx

    @nx.setter
    def nx(self, value):
        if value is None:
            raise ValueError("nx must be provided")
        if not isinstance(value, int):
            raise ValueError("nx must be integer")
        self._nx = value

    @property
    def ny(self):
        return self._ny

    @ny.setter
    def ny(self, value):
        if value is None:
            raise ValueError("ny must be provided")
        if not isinstance(value, int):
            raise ValueError("ny must be integer")
        self._ny = value

    @property
    def nz(self):
        return self._nz

    @nz.setter
    def nz(self, value):
        if value is None:
            raise ValueError("nz must be provided")
        if not isinstance(value, int):
            raise ValueError("nz must be integer")
        self._nz = value

    @property
    def lattice(self):
        return self._lattice

    @lattice.setter
    def lattice(self, value):
        if value is None:
            raise ValueError("Value must be provided")
        if self.nz == 0 and not isinstance(value, D2Q9):
            raise ValueError("For 2D simulations, lattice must be D2Q9")
        if self.nz != 0 and value.name not in ["D3Q19", "D3Q27"]:
            raise ValueError("For 3D simulations, lattice must be D3Q19 or D3Q27")
        self._lattice = value

    @property
    def compute_precision(self):
        return self._compute_precision

    @compute_precision.setter
    def compute_precision(self, value):
        if value is None:
            raise ValueError("Precision value must be provided.")
        if value not in [jnp.float16, jnp.float32, jnp.float64]:
            raise ValueError('Valid precision values are: "f16", "f32" or "f64"')
        self._compute_precision = value

    @property
    def write_precision(self):
        return self._write_precision

    @write_precision.setter
    def write_precision(self, value):
        if value is None:
            raise ValueError("Write precison value must be provided.")
        if value not in [jnp.float16, jnp.float32, jnp.float64]:
            raise ValueError('Valid presion values are: "f16", "f32" or "f64"')
        self._write_precision = value

    @property
    def total_timesteps(self):
        return self._total_timesteps

    @total_timesteps.setter
    def total_timesteps(self, value):
        if value is None:
            raise ValueError("Total timesteps value must be provided")
        if not isinstance(value, int):
            raise ValueError("Total timesteps must be an integer")
        self._total_timesteps = value

    @property
    def checkpoint_rate(self):
        return self._checkpoint_rate

    @checkpoint_rate.setter
    def checkpoint_rate(self, value):
        if value is None:
            raise ValueError("checkpoint_rate must be provided")
        if not isinstance(value, int):
            raise ValueError("checkpoint_rate must be an integer")
        self._checkpoint_rate = value

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @checkpoint_dir.setter
    def checkpoint_dir(self, value):
        if value is None:
            raise ValueError("checkpoint_dir must be provided")
        if not isinstance(value, str):
            raise ValueError("checkpoint_dir must be a string")
        self._checkpoint_dir = value

    @property
    def restore_checkpoint(self):
        return self._restore_checkpoint

    @restore_checkpoint.setter
    def restore_checkpoint(self, value):
        if value is None:
            raise ValueError("restore_checkpoint must be provided")
        if not isinstance(value, bool):
            raise ValueError("restore_checkpoint must be a boolean")
        self._restore_checkpoint = value

    @property
    def write_start(self):
        return self._write_start

    @write_start.setter
    def write_start(self, value):
        if value is None:
            raise ValueError("write_start must be provided.")
        if not isinstance(value, int):
            raise ValueError("write_start must be an integer.")
        self._write_start = value

    @property
    def write_control(self):
        return self._write_control

    @write_control.setter
    def write_control(self, value):
        if value is None:
            raise ValueError("write_control must be provided.")
        if not isinstance(value, int):
            raise ValueError("write_control must be an integer.")
        self._write_control = value

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        if value is None:
            raise ValueError("output_dir must be provided")
        if not isinstance(value, str):
            raise ValueError("output_dir must be a string")
        self._output_dir = value

    @property
    def compute_mlups(self):
        return self._compute_mlups

    @compute_mlups.setter
    def compute_mlups(self, value):
        if value is None:
            raise ValueError("compute_mlups must be provided")
        if not isinstance(value, bool):
            raise ValueError("compute_mlups must be a boolean")
        self._compute_mlups = value

    @property
    def print_info_rate(self):
        return self._print_info_rate

    @print_info_rate.setter
    def print_info_rate(self, value):
        if value is None:
            print("print_info_rate value must be provided")
        if not isinstance(value, int):
            raise ValueError("print_info_rate must be an integer.")
        self._print_info_rate = value

    @property
    def n_devices(self):
        return self._n_devices

    @n_devices.setter
    def n_devices(self, value):
        if not isinstance(value, int):
            raise TypeError("n_devices must be an integer")
        self._n_devices = value

    def set_precision(self, precision_str):
        return {"f16": jnp.float16, "f32": jnp.float32, "f64": jnp.float64}.get(
            precision_str, jnp.float32
        )

    def show_simulation_parameters(self):
        attributes_to_show = [
            "omega",
            "nx",
            "ny",
            "nz",
            "d",
            "lattice",
            "compute_precision",
            "write_precision",
            "total_timesteps",
            "checkpoint_rate",
            "checkpoint_dir",
            "restore_checkpoint",
            "write_start",
            "write_control",
            "output_dir",
            "compute_mlups",
            "downsampling_factor",
            "n_devices",
            "backend",
        ]

        descriptive_names = {
            "omega": "Omega",
            "nx": "Grid Points in X",
            "ny": "Grid Points in Y",
            "nz": "Grid Points in Z",
            "d": "Dimensionality",
            "lattice": "Lattice Type",
            "compute_precision": "Precision used for computation",
            "write_precision": "Precision used for writing the output files",
            "total_timesteps": "Total timesteps run in the simulation",
            "checkpoint_rate": "Rate at which files are generated",
            "checkpoint_dir": "Directory where checkpoint files are written",
            "restore_checkpoint": "Start simulation from a checkpoint instead of beginning",
            "write_start": "Timestep from which output export begins",
            "write_control": "The rate of output export in terms of timesteps",
            "compute_mlups": "Determines if the MLUPS will be computed",
            "downsampling_factor": "Downsampling factor used for the output.",
            "n_devices": "Number of Devices",
            "backend": "Backend",
        }
        simulation_name = self.__class__.__name__

        print(
            colored(f"**** Simulation Parameters for {simulation_name} ****", "green")
        )
        header = f"{colored('Parameter', 'blue'):>30} | {colored('Value', 'yellow')}"
        print(header)
        print("-" * 50)

        for attr in attributes_to_show:
            value = getattr(self, attr, "Attribute not set")
            descriptive_name = descriptive_names.get(
                attr, attr
            )  # Use the attribute name as a fallback
            row = (
                f"{colored(descriptive_name, 'blue'):>30} | {colored(value, 'yellow')}"
            )
            print(row)

    def _create_boundary_data(self):
        """
        Creates the data necessary for applying boundary conditions:
            1. Computing grid_mask
            2. Computing local mask and normal arrays

        Arguments:
            None
        """
        self.boundary_conditions = []
        self.set_boundary_conditions()
        solid_halo_list = [
            np.array(bc.boundary_indices).T
            for bc in self.boundary_conditions
            if bc.is_solid
        ]
        solid_halo_voxels = (
            np.unique(np.vstack(solid_halo_list), axis=0) if solid_halo_list else None
        )

        start = time.time()
        grid_mask = self.create_grid_mask(solid_halo_voxels)
        print("Time to create the grid mask: ", time.time() - start)

        start = time.time()
        for bc in self.boundary_conditions:
            assert bc.implementation_step in ["post_streaming", "post_collision"]
            bc.create_local_mask_and_normal_arrays(grid_mask)
        print("Time to create the local masks and normal arrays:", time.time() - start)

    def distributed_array_init(self, shape, ttype, init_val=0, sharding=None):
        """
        Generate a jax distributed array with given shape, type, initial value and sharding strategy.

        Arguments:
            shape: tuple
                Shape of the desired distributed array.
            ttype: type
                Data type of the array elements.
            init_val: float
                Initial value to be used while  declaring the array.
            sharding: None
                Sharding strategy to be used for distributing the array between devices

        Returns:
            x: jax.ndarray
                Distributed array with given shape, data type, initial value and sharding strategy
        """
        if sharding is None:
            sharding = self.sharding
        x = jnp.full(shape=shape, fill_value=init_val, dtype=ttype)
        return jax.lax.with_sharding_constraint(x, sharding)

    def initialize_macroscopic_fields(self):
        """
        Functions to initialize the density and velocity arrays with their corresponding initial values.
        By default, velocities are 0 everywhere and density is 1.0 everywhere.

        Note: Function must be overwritten in a subclass or instance of the class.

        Arguments:
            None by default, can be overwritten as required

        Returns:
            None, None: The default density and velocity values, both None.
            This indicates that the actual values should be set elsewhere.
        """
        print("Default initial conditions assumed: density = 1.0 and velocity = 0.0")
        print(
            "To set explicit initial values for velocity and density, use the self.initialize_macroscopic_fields function"
        )
        return None, None

    def assign_fields_sharded(self):
        """
        This function is used to initialize the distribution array using the initial velocities and velocity defined in self.initialize_macroscopic_fields function.
        To do this, function first uses the initialize_macroscopic_fields function to get the initial values of rho (rho0) and velocity (u0).

        The distribution is initialized with rho0 and u0 values, using the self.equilibrium function.

        Arguments:
            None

        Returns:
            f: A distributed JAX array of shape: (self.nx, self.ny, self.q) for 2D and (self.nx, self.ny, self.nz, self.q) for 3D.
        """
        rho0, u0 = self.initialize_macroscopic_fields()
        shape = (
            (self.nx, self.ny, self.q)
            if self.d == 2
            else (self.nx, self.ny, self.nz, self.q)
        )
        if rho0 is None or u0 is None:
            f = self.distributed_array_init(
                shape, self.precision_policy.output_dtype, init_val=self.w
            )
        else:
            f = self.initialize_distribution(rho0, u0)
        return f

    def initialize_distribution(self, rho0, u0):
        """
        This function is used to initialize the distribution array for the simulation.
        It uses the equilibrium distribution function with initial density and velocities
        being defined in self.initialize_macroscopic_fields.

        Arguments:
            None

        Returns:
            f: JAX array
                JAX array holding the distribution array used in the simulation
        """
        return self.equilibrium(rho0, u0)

    @partial(jit, static_argnums=(0,))
    def create_grid_mask(self, solid_halo_voxels):
        """
        Create the binary mask for the known and unknown directions in the lattice.

        Arguments:
            solid_mask: jax.array
                Indices of the solid lattice points in the grid.

        Returns:
            None
        """
        hw_x = self.n_devices
        hw_y = hw_z = 1
        if self.d == 2:
            grid_mask = self.distributed_array_init(
                (self.nx + 2 * hw_x, self.ny + 2 * hw_y, self.lattice.q),
                jnp.bool_,
                init_val=True,
            )
            grid_mask = grid_mask.at[
                (slice(hw_x, -hw_x), slice(hw_y, -hw_y), slice(None))
            ].set(False)
            if solid_halo_voxels is not None:
                solid_halo_voxels = solid_halo_voxels.at[:, 0].add(hw_x)
                solid_halo_voxels = solid_halo_voxels.at[:, 1].add(hw_y)
                grid_mask = grid_mask.at[tuple(solid_halo_voxels.T)].set(True)

            grid_mask = self.streaming(grid_mask)
            return lax.with_sharding_constraint(grid_mask, self.sharding)

        elif self.d == 3:
            grid_mask = self.distributed_array_init(
                (
                    self.nx + 2 * hw_x,
                    self.ny + 2 * hw_y,
                    self.nz + 2 * hw_z,
                    self.lattice.q,
                ),
                jnp.bool_,
                init_val=True,
            )
            grid_mask = grid_mask.at[
                (
                    slice(hw_x, -hw_x),
                    slice(hw_y, -hw_y),
                    slice(hw_z, -hw_z),
                    slice(None),
                )
            ].set(False)
            if solid_halo_voxels is not None:
                solid_halo_voxels = solid_halo_voxels.at[:, 0].add(hw_x)
                solid_halo_voxels = solid_halo_voxels.at[:, 1].add(hw_y)
                solid_halo_voxels = solid_halo_voxels.at[:, 2].add(hw_z)
                grid_mask = grid_mask.at[tuple(solid_halo_voxels.T)].set(True)
            grid_mask = self.streaming(grid_mask)
            return lax.with_sharding_constraint(grid_mask, self.sharding)

    def bounding_box_indices_(self):
        """
        This function calculates the indices of the bounding box of a 2D or 3D grid.
        The bounding box is defined as the set of grid points on the outer edge of the grid.

        Returns:
            bounding_box: (dict)
            A dictionary where keys are the names of the bounding box faces
            ("bottom", "top", "left", "right" for 2D; additional "front", "back" for 3D), and values
            are numpy arrays of indices corresponding to each face.
        """
        if self.d == 2:
            # For a 2D grid, the bounding box consists of four edges: bottom, top, left, and right.
            # Each edge is represented as an array of indices. For example, the bottom edge includes
            # all points where the y-coordinate is 0, so its indices are [[i, 0] for i in range(self.nx)].
            bounding_box = {
                "bottom": np.array([[i, 0] for i in range(self.nx)], dtype=int),
                "top": np.array([[i, self.ny - 1] for i in range(self.nx)], dtype=int),
                "left": np.array([[0, i] for i in range(self.ny)], dtype=int),
                "right": np.array(
                    [[self.nx - 1, i] for i in range(self.ny)], dtype=int
                ),
            }
            return bounding_box

        elif self.d == 3:
            # For a 3D grid, the bounding box consists of six faces: bottom, top, left, right, front, and back.
            # Each face is represented as an array of indices. For example, the bottom face includes all points
            # where the z-coordinate is 0, so its indices are [[i, j, 0] for i in range(self.nx) for j in range(self.ny)].
            bounding_box = {
                "bottom": np.array(
                    [[i, j, 0] for i in range(self.nx) for j in range(self.ny)],
                    dtype=int,
                ),
                "top": np.array(
                    [
                        [i, j, self.nz - 1]
                        for i in range(self.nx)
                        for j in range(self.ny)
                    ],
                    dtype=int,
                ),
                "left": np.array(
                    [[0, j, k] for j in range(self.ny) for k in range(self.nz)],
                    dtype=int,
                ),
                "right": np.array(
                    [
                        [self.nx - 1, j, k]
                        for j in range(self.ny)
                        for k in range(self.nz)
                    ],
                    dtype=int,
                ),
                "front": np.array(
                    [[i, 0, k] for i in range(self.nx) for k in range(self.nz)],
                    dtype=int,
                ),
                "back": np.array(
                    [
                        [i, self.ny - 1, k]
                        for i in range(self.nx)
                        for k in range(self.nz)
                    ],
                    dtype=int,
                ),
            }
            return bounding_box

    def send_right(self, x, axis_name):
        """
        This function sends the data to the right neighboring process in a parallel computing environment.
        It uses a permutation operation provided by the LAX library.

        Parameters
        ----------
        x: jax.numpy.ndarray
            The data to be sent.
        axis_name: str
            The name of the axis along which the data is sent.

        Returns
        -------
        jax.numpy.ndarray
            The data after being sent to the right neighboring process.
        """
        return lax.ppermute(x, perm=self.right_perm, axis_name=axis_name)

    def send_left(self, x, axis_name):
        """
        This function sends the data to the left neighboring process in a parallel computing environment.
        It uses a permutation operation provided by the LAX library.

        Parameters
        ----------
        x: jax.numpy.ndarray
            The data to be sent.
        axis_name: str
            The name of the axis along which the data is sent.

        Returns
        -------
            The data after being sent to the left neighboring process.
        """
        return lax.ppermute(x, perm=self.left_perm, axis_name=axis_name)

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def streaming_p(self, fin):
        """
        Perform streaming operation on a partitioned (in the x-direction) distribution function

        Arguments:
            f: jax.ndarray
                Distribution function which is defined over multiple devices.

        Returns:
            f: jax.ndarray
                Distribution function after streaming has been applied
        """

        def streaming_i(f, e):
            if self.d == 2:
                return jnp.roll(f, (e[0], e[1]), axis=(0, 1))
            elif self.d == 3:
                return jnp.roll(f, (e[0], e[1], e[2]), axis=(0, 1, 2))

        return vmap(streaming_i, in_axes=(-1, 0), out_axes=-1)(fin, self.e.T)

    def streaming_m(self, f):
        """
        This function will communicate the respective streamed distribution from other neighbouring array shards.
        To the current shard.

        (left_halo, right_indices)| device |(right_halo, left_indices)

        Arguments:
            f: jax.ndarray
                Sharded array storing the distribution function

        """
        f = self.streaming_p(f)
        left_comm, right_comm = (
            f[:1, ..., self.lattice.right_indices],
            f[-1:, ..., self.lattice.left_indices],
        )
        left_comm, right_comm = self.send_right(left_comm, "x"), self.send_left(
            right_comm, "x"
        )
        f = f.at[:1, ..., self.lattice.right_indices].set(left_comm)
        f = f.at[-1:, ..., self.lattice.left_indices].set(right_comm)
        return f

    def compute_macroscopic_variables(self, f):
        """
        Compute the macroscopic variables density (rho) and velocity (u) using the distributions.

        Arguments:
            f: jax.numpy.ndarray
                Distribution array, storing distribution for all lattice nodes for all directions.

        Returns:
            rho: jax.numpy.ndarray
                Density at each lattice nodes.
            u: jax.numpy.ndarray
                Velocity at each lattice nodes.
        """
        rho = jnp.sum(f, axis=-1, keepdims=True)
        e = jnp.array(self.e, dtype=self.precision_policy.compute_dtype).T
        u = jnp.dot(f, e) / rho
        return rho, u

    @partial(jit, static_argnums=(0,))
    def momentum_flux(self, fneq):
        """
        This function computes the momentum flux, which is the product of the non-equilibrium
        distribution functions (fneq) and the lattice moments (cc).

        The momentum flux is used in the computation of the stress tensor in the Lattice Boltzmann
        Method (LBM).

        Parameters:
            fneq: jax.numpy.ndarray
                The non-equilibrium distribution functions.

        Returns:
            jax.numpy.ndarray
                The computed momentum flux.
        """
        return jnp.dot(fneq, self.lattice.ee)

    @partial(jit, static_argnums=(0, 3))
    def equilibrium(self, rho, u, cast_output=True):
        """
        Compute the equillibrium distribution function using the given values of density and velocity.

        Arguments:
            rho: jax.numpy.ndarray
                Density values.
            u: jax.numpy.ndarray
                Velocity values.
            cast_output: bool {Optional}
                A flag to cast the density and velocity values to the compute and output precision. Default: True

        Returns:
            feq: Array-like
                Equillibrium distribution.
        """
        if cast_output:
            rho, u = self.precision_policy.cast_to_compute((rho, u))

        e = jnp.array(self.e, dtype=self.precision_policy.compute_dtype)
        udote = jnp.dot(u, e)
        udotu = jnp.sum(jnp.square(u), axis=-1, keepdims=True)
        feq = rho * self.w * (1.0 + udote * (3.0 + 4.5 * udote) - 1.5 * udotu)

        if cast_output:
            return self.precision_policy.cast_to_output(feq)
        else:
            return feq

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, fin):
        """
        Implementation of the collision step in the Lattice Boltzmann Method. Defined in collision model sub-class.

        Arguments:
            fin: jac.numpy.ndarray
                Distribution function.
            rho: jax.numpy.ndarray
                Density at all the lattice nodes.
            u: jax.numpy.ndarray
                Velocity at all the lattice nodes.

        Returns:
            fout: Array-like
                Distribution function.
        """
        pass

    @partial(jit, static_argnums=(0, 4))
    def apply_boundary_conditions(self, fout, fin, timestep, implementation_step):
        """
        Apply the boundary condition to the grid points identified in the boundary_indices (see boundary_conditions.py)

        Arguments:
            fout: jax.numpy.ndarray
                Output distribution function.
            fin: jax.numpy.ndarray
                Input distribution function.
            timestep: int
                Timestep to be used for applying the boundary condition.
                Useful for dynamic boundary conditions, such as moving wall boundary condition.
            implementation_step: str
                The implementation step is matched for boundary condition for all the lattice points.

        Returns:
            fout: jax.numpy.ndarray
                Output distribution values at lattice nodes.
        """
        for bc in self.boundary_conditions:
            fout = bc.prepare_populations(fout, fin, implementation_step)
            if bc.implementation_step == implementation_step:
                if bc.is_dynamic:
                    fout = bc.apply(fout, fin, timestep)
                else:
                    fout = fout.at[bc.boundary_indices].set(bc.apply(fout, fin))
        return fout

    @partial(jit, static_argnums=(0, 1))
    def calculate_mlups(self, t_total):
        """
        Calculate the performance of the LBM code using MLUPS: Million Lattice Updates Per Second (MLUPS)
        Formula:
            (mesh_size * no_of_iterations) / (total_running_time * 1e6)

            where:
                mesh_size = nx*ny in 2D and nx*ny*nz in 3D
                total_time = t_simulation_stop - t_simulation_time

        Arguments:
            t_total: float
                Total time elapsed for computing all the steps in the simulation.

        Returns:
            mlups: float
                The MLUPS value for the simulation.
        """
        if self.lattice.d == 2:
            mlups = (self.nx * self.ny * self.total_timesteps) / (t_total * 1e6)
        else:
            mlups = (self.nx * self.ny * self.nz * self.total_timesteps) / (
                t_total * 1e6
            )
        return mlups

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def step(self, f_poststreaming, timestep):
        """
        Perform one step of LBM simulation.

        Arguments:
            f_poststreaming: jax.numpy.ndarray
                Post-streaming distribution function.
            timestep: int
                Current timestep

        Returns:
            f_poststreaming: jax.numpy.ndarray
                Post-streaming distribution function.
            f_collision: jax.numpy.ndarray
                Post-collision distribution function.
        """
        f_postcollision = self.collision(f_poststreaming)
        f_postcollision = self.apply_boundary_conditions(
            f_postcollision, f_poststreaming, timestep, "post_collision"
        )
        f_poststreaming = self.streaming(f_postcollision)
        f_poststreaming = self.apply_boundary_conditions(
            f_poststreaming, f_postcollision, timestep, "post_streaming"
        )

        if self.return_post_col_dist:
            return f_poststreaming, f_postcollision
        else:
            return f_poststreaming, None

    def run(self):
        """
        This function runs the LBM simulation for a specified number of time steps.

        It first initializes the distribution functions and then enters a loop where it performs the
        simulation steps (collision, streaming, and boundary conditions) for each time step.

        The function can also print the progress of the simulation, save the simulation data, and
        compute the performance of the simulation in million lattice updates per second (MLUPS).

        Arguments:
            None
        Returns:
            f: jax.numpy.ndarray
                The distribution functions after the simulation.
        """
        f = self.assign_fields_sharded()
        start_step = 0
        if self.restore_checkpoint:
            latest_step = self.mngr.latest_step()
            if latest_step is not None:  # existing checkpoint present
                # Assert that the checkpoint manager is not None
                assert self.mngr is not None, "Checkpoint manager does not exist."
                state = {"f": f}
                abstract_state = jax.tree_util.tree_map(
                    orb.utils.to_shape_dtype_struct, state
                )
                # shardings = jax.tree_map(lambda x: x.sharding, state)
                # restore_args = orb.checkpoint_utils.construct_restore_args(
                #     state, shardings
                # )
                try:
                    f = self.mngr.restore(
                        latest_step,
                        # restore_kwargs={"restore_args": restore_args},
                        args=orb.args.StandardRestore(abstract_state),
                    )["f"]
                    print(f"Restored checkpoint at step {latest_step}.")
                except ValueError:
                    raise ValueError(
                        f"Failed to restore checkpoint at step {latest_step}."
                    )

                start_step = latest_step + 1
                if not (self.total_timesteps > start_step):
                    raise ValueError(
                        f"Simulation already exceeded maximum allowable steps (self.total_timesteps = {self.total_timesteps}). Consider increasing self.total_timesteps."
                    )

        if self.compute_mlups:
            start = time.time()

        # Loop over all time steps
        for timestep in range(start_step, self.total_timesteps + 1):
            io_flag = self.write_control > 0 and (
                (timestep - self.write_start) % self.write_control == 0
                or timestep == self.total_timesteps
            )
            print_iter_flag = (
                self.print_info_rate > 0 and timestep % self.print_info_rate == 0
            )
            checkpoint_flag = (
                self.checkpoint_rate > 0 and timestep % self.checkpoint_rate == 0
            )

            if io_flag:
                # Update the macroscopic variables and save the previous values (for error computation)
                rho_prev, u_prev = self.compute_macroscopic_variables(f)
                rho_prev = downsample_field(rho_prev, self.downsampling_factor)
                u_prev = downsample_field(u_prev, self.downsampling_factor)
                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                rho_prev = process_allgather(rho_prev)
                u_prev = process_allgather(u_prev)

            # Perform one time-step (collision, streaming, and boundary conditions)
            f, fstar = self.step(f, timestep)

            # Print the progress of the simulation
            if print_iter_flag:
                print(
                    colored("Timestep ", "blue")
                    + colored(f"{timestep}", "green")
                    + colored(" of ", "blue")
                    + colored(f"{self.total_timesteps}", "green")
                    + colored(" completed", "blue")
                )

            if io_flag:
                # Save the simulation data
                print(f"Saving data at timestep {timestep}/{self.total_timesteps}")
                rho, u = self.compute_macroscopic_variables(f)
                rho = downsample_field(rho, self.downsampling_factor)
                u = downsample_field(u, self.downsampling_factor)

                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                rho = process_allgather(rho)
                u = process_allgather(u)

                # Save the data
                self.handle_io_timestep(timestep, f, fstar, rho, u, rho_prev, u_prev)

            if checkpoint_flag:
                # Save the checkpoint
                print(
                    f"Saving checkpoint at timestep {timestep}/{self.total_timesteps}"
                )
                state = {"f": f}
                self.mngr.save(timestep, args=orb.args.StandardSave(state))

            # Start the timer for the MLUPS computation after the first timestep (to remove compilation overhead)
            if self.compute_mlups and timestep == 1:
                jax.block_until_ready(f)
                start = time.time()

        if self.compute_mlups:
            # Compute and print the performance of the simulation in MLUPS
            jax.block_until_ready(f)
            end = time.time()
            if self.d == 2:
                print(
                    colored("Domain: ", "blue")
                    + colored(f"{self.nx} x {self.ny}", "green")
                    if self.d == 2
                    else colored(f"{self.nx} x {self.ny} x {self.nz}", "green")
                )
                print(
                    colored("Number of voxels: ", "blue")
                    + colored(f"{self.nx * self.ny}", "green")
                    if self.d == 2
                    else colored(f"{self.nx * self.ny * self.nz}", "green")
                )
                print(
                    colored("MLUPS: ", "blue")
                    + colored(
                        f"{self.nx * self.ny * self.total_timesteps / (end - start) / 1e6}",
                        "red",
                    )
                )

            elif self.d == 3:
                print(
                    colored("Domain: ", "blue")
                    + colored(f"{self.nx} x {self.ny} x {self.nz}", "green")
                )
                print(
                    colored("Number of voxels: ", "blue")
                    + colored(f"{self.nx * self.ny * self.nz}", "green")
                )
                print(
                    colored("MLUPS: ", "blue")
                    + colored(
                        f"{self.nx * self.ny * self.nz * self.total_timesteps / (end - start) / 1e6}",
                        "red",
                    )
                )

        return f

    def handle_io_timestep(self, timestep, f, fstar, rho, u, rho_prev, u_prev):
        """
        This function handles the input/output (I/O) operations at each time step of the simulation.

        It prepares the data to be saved and calls the output_data function, which can be overwritten
        by the user to customize the I/O operations.

        Parameters:
            timestep: int
                The current time step of the simulation.
            f: jax.numpy.ndarray
                The post-streaming distribution functions at the current time step.
            fstar: jax.numpy.ndarray
                The post-collision distribution functions at the current time step.
            rho: jax.numpy.ndarray
                The density field at the current time step.
            u: jax.numpy.ndarray
                The velocity field at the current time step.
        """
        kwargs = {
            "timestep": timestep,
            "rho": rho,
            "rho_prev": rho_prev,
            "u": u,
            "u_prev": u_prev,
            "f_poststreaming": f,
            "f_postcollision": fstar,
        }
        self.output_data(**kwargs)

    def output_data(self, **kwargs):
        """
        This function is intended to be overwritten by the user to customize the input/output (I/O)
        operations of the simulation.

        By default, it does nothing. When overwritten, it could save the simulation data to files,
        display the simulation results in real time, send the data to another process for analysis, etc.

        Parameters:
            **kwargs: dict
                A dictionary containing the simulation data to be outputted. The keys are the names of the
                data fields, and the values are the data fields themselves.
        """
        pass

    def set_boundary_conditions(self):
        """
        This function sets the boundary conditions for the simulation.

        It is intended to be overwritten by the user to specify the boundary conditions according to
        the specific problem being solved.

        By default, it does nothing. When overwritten, it could set periodic boundaries, no-slip
        boundaries, inflow/outflow boundaries, etc.
        """
        pass

    def get_force(self):
        """
        This function computes the force to be applied to the fluid in the Lattice Boltzmann Method.

        It is intended to be overwritten by the user to specify the force according to the specific
        problem being solved.

        By default, it does nothing and returns None. When overwritten, it could implement a constant
        force term.

        Returns:
            force: jax.numpy.ndarray
                The force to be applied to the fluid.
        """
        pass

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force(self, f_postcollision, feq, rho, u):
        """
        Add force based on exact-difference method due to Kupershtokh

        Parameters:
            f_postcollision: jax.numpy.ndarray
                The post-collision distribution functions.
            feq: jax.numpy.ndarray
                The equilibrium distribution functions.
            rho: jax.numpy.ndarray
                The density field.

            u: jax.numpy.ndarray
                The velocity field.

        Returns:
            f_postcollision: jax.numpy.ndarray
                The post-collision distribution functions with the force applied.

        References:
            Kupershtokh, A. (2004). New method of incorporating a body force term into the lattice Boltzmann equation. In
            Proceedings of the 5th International EHD Workshop (pp. 241-246). University of Poitiers, Poitiers, France.

            Chikatamarla, S. S., & Karlin, I. V. (2013). Entropic lattice Boltzmann method for turbulent flow simulations:
            Boundary conditions. Physica A, 392, 1925-1930.
            Krüger, T., et al. (2017). The lattice Boltzmann method. Springer International Publishing, 10.978-3, 4-15.
        """
        delta_u = self.get_force()
        feq_force = self.equilibrium(rho, u + delta_u, cast_output=False)
        f_postcollision = f_postcollision + feq_force - feq
        return f_postcollision
