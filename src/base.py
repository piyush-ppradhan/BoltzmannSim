"""
    Definition of LBMBase class for defining and running a problem
"""

# Standard libraries
import time

# Third-party imports
from functools import partial
import numpy as np
from termcolor import colored
import jax

# JAX-specific imports
import jax.numpy as jnp
from jax import jit, lax, vmap, config
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding, NamedSharding
from jax.experimental.shard_map import shard_map
from jax.experimental.multihost_utils import process_allgather

# Locally defined functions import
from body_force import NoBodyForce, ShanChenBodyForce, GuoBodyForce
from utilities import write_vtk

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
            body_force: BodyForce
                Define any volumetric forces to be applied during simulation as defined in body_force.py. Default: NoBodyForce()
            precision: str
                Precision for computation and exporting the data. Same as lattice.precision Default: "f32"
            solid_mask: array[int]
                Array to identify all the solid nodes in the grid. Solid node: 0, Fluid node: 1
            rho0: array[float]
                Initial value of density at each lattice node. Dimension: (nx,ny) or (nx,ny,nz). Default: 1.0 for all grid points
            u0: array[float]
                Initial value of density at each lattice node. Dimension: (nx,ny,2) or (nx,ny,nz,2). Default: 0.0 for all grid points
            boundary_conditions: BoundaryCondition
                Boundary conditions used in the problem, defined as a list of objects of class BoundaryCondition.
            conv_param: ConversionParameter
                Conversion parameters to convert between Lattice Units and SI units.
            filename_prefix: str
                Prefix used in the exported VTK file. Add relative folder location to save in a different file.
                The final filename will be: filename_prefix-timestep.vtk Default: ./output-timestep.vtk
            write_start: int
                Timestep at which the export of data is started
            write_control: int
                Timestep interval after which data is exported, starting from write_start
            create_log: bool
                Log file is saved in current directory, it will overwrite the preci
            output_dir: str
                Directory where output files will be exported. Auto-created if it doesn't exist
            compute_mlups: bool
                Compute the Million Lattice Updates per Second (MLUPS) to evaluate the performance of the code. Default: False
            checkpoint_rate: int {optional}
                Rate at which checkpoint files will be written. Default: 0
            checkpoint_dir : str {optional}
                Ignored if restart_flag is set to False. Default: ./checkpoints
            restore_checkpoint: bool
                The simulation from the latest checkpoint if set to trur. Default: False
    """
    def __init__(self, **kwargs):
        self.omega = kwargs.get("omega")
        self.nx = kwargs.get("nx")
        self.ny = kwargs.get("ny")
        self.nz = kwargs.get("nz",0)
        self.lattice = kwargs.get("lattice")
        self.body_force = kwargs.get("body_force", NoBodyForce())
        self.precision = self.lattice.precision # Used for computation as well as storage of data
        self.write_precision = kwargs.get("write_precision")
        self.solid_mask = kwargs.get("solid_mask")
        self.total_timesteps = kwargs.get("total_timesteps")
        self.rho0 = kwargs.get("rho0")
        self.u0 = kwargs.get("u0")
        self.boundary_conditions = kwargs.get("boundary_conditions") # The boundary conditions are passed during the problem definition
        self.conv_param = kwargs.get("conversion_parameters")
        self.filename_prefix = kwargs.get("filename_prefix", "./output")
        self.write_start = kwargs.get("write_start")
        self.write_control = kwargs.get("write_control")
        self.create_log = kwargs.get("create_log", True)
        self.output_dir = kwargs.get("output_dir")
        self.compute_mlups = kwargs.get("compute_MLUPS", False)
        self.checkpoint_rate = kwargs.get("checkpoint_rate", 0)
        self.checkpoint_dir = kwargs.get("checkpoint_dir", 0)
        self.restore_checkpoint = kwargs.get("restore_checkpoint", False)

        self.nDevices = jax.device_count()
        self.backend = jax.default_backend()


        self.d = self.lattice.d
        self.q = self.lattice.q
        self.e = self.lattice.e
        self.w = self.lattice.w

        #Configure JAX to use 64-bit precision if necessary
        if self.precision == jnp.float64:
            config.update("jax_enable_x64", True)
            print("Using 64-bit precision for computation and file output")

        # Check if nx, ny, nz has been defined appropriately
        if None in {self.nx, self.ny, self.nz}:
            print("Error: at least nx and ny must be provided to perform the simulation")
            exit()

        # Scale nx to be divisible by the number of devices. If nx is not divisible, than scale nx to make it divisible
        nx = self.nx
        if self.nx % self.nDevices:
            self.nx = nx + (self.nDevices - nx % self.nDevices)

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value):
        if value is None:
            raise ValueError("Omega must be provided")
        if not isinstance(value, float):
            raise ValueError("Omega must be float")
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
        if self.nz == 0:
            raise ValueError("For 2D simulations, lattice must be D2Q9")
        if self.nz != 0 and value.name not in ['D3Q19', 'D3Q27']:
            raise ValueError("For 3D simulations, lattice must be D3Q19 or D3Q27")
        self._lattice = value

    @property
    def body_force(self):
        return self._body_force

    @body_force.setter
    def body_force(self, value):
        if value is None:
            raise ValueError("Body force must be provided")
        if not type(value).__name__ in ['NoBodyForce', 'ShanChenBodyForce', 'GuoBodyForce']:
            raise ValueError("Body force must be of type: NoBodyForce, ShanChenBodyForce or GuoBodyForce")
        self._body_force = value

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, value):
        if value is None:
            raise ValueError("Precison value must be provided")
        if value not in ['f16', 'f32', 'f64']:
            raise ValueError("Valid presion values are: 'f16', 'f32' or 'f64'")
        self._precision = value

    @property
    def write_precision(self):
        return self._write_precision

    @write_precision.setter
    def write_precision(self, value):
        if value is None:
            raise ValueError("Write precison value must be provided")
        if value not in ['f16', 'f32', 'f64']:
            raise ValueError("Valid presion values are: 'f16', 'f32' or 'f64'")
        self._write_precision = value

    @property
    def total_timesteps(self):
        return self._total_timesteps

    @total_timesteps.setter
    def total_timesteps(self, value):
        if value is None:
            raise ValueError("Total timesteps value must be provided")
        if not isinstance(value,int):
            raise ValueError("Total timesteps must be an integer")
        self._total_timesteps = value

    @property
    def filename_prefix(self):
        return self._filename_prefix

    @filename_prefix.setter
    def filename_prefix(self, value):
        if value is None:
            raise ValueError("filename_prefix must be provided")
        if not isinstance(value,str):
            raise ValueError("filename_prefix must be an integer")
        self._filename_prefix = value

    @property
    def write_start(self):
        return self._write_start

    @write_start.setter
    def write_start(self, value):
        if value is None:
            raise ValueError("write_start must be provided")
        if not isinstance(value,int):
            raise ValueError("write_start must be an integer")
        self._write_start = value

    @property
    def write_control(self):
        return self._write_control

    @write_control.setter
    def write_control(self, value):
        if value is None:
            raise ValueError("write_control must be provided")
        if not isinstance(value,int):
            raise ValueError("write_control must be an integer")
        self._write_control = value

    @property
    def create_log(self):
        return self._create_log

    @create_log.setter
    def create_log(self, value):
        if value is None:
            raise ValueError("create_log must be provided")
        if not isinstance(value,bool):
            raise ValueError("create_log must be a boolean")
        self._create_log = value

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        if value is None:
            raise ValueError("output_dir must be provided")
        if not isinstance(value,str):
            raise ValueError("output_dir must be a string")
        self._output_dir = value

    @property
    def compute_mlups(self):
        return self._compute_mlups

    @compute_mlups.setter
    def compute_mlups(self, value):
        if value is None:
            raise ValueError("compute_mlups must be provided")
        if not isinstance(value,bool):
            raise ValueError("compute_mlups must be a boolean")
        self._compute_mlups = value

    @property
    def checkpoint_rate(self):
        return self._checkpoint_rate

    @checkpoint_rate.setter
    def checkpoint_rate(self, value):
        if value is None:
            raise ValueError("checkpoint_rate must be provided")
        if not isinstance(value,int):
            raise ValueError("checkpoint_rate must be an integer")
        self._checkpoint_rate = value

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @checkpoint_dir.setter
    def checkpoint_dir(self, value):
        if value is None:
            raise ValueError("checkpoint_dir must be provided")
        if not isinstance(value,str):
            raise ValueError("checkpoint_dir must be a string")
        self._checkpoint_dir = value

    @property
    def restore_checkpoint(self):
        return self._restore_checkpoint

    @restore_checkpoint.setter
    def restore_checkpoint(self, value):
        if value is None:
            raise ValueError("restore_checkpoint must be provided")
        if not isinstance(value,bool):
            raise ValueError("restore_checkpoint must be a boolean")
        self._restore_checkpoint = value

    def show_simulation_parameters(self):
        attributes_to_show = [
            'omega', 'nx', 'ny', 'nz', 'd', 'lattice', 'precision', 'write_precision', 
            'total_timesteps', 'conv_param', 'write_start', 'write_control', 
            'create_log', 'output_dir', 'compute_mlups', 'checkpoint_rate',
            'checkpoint_dir', 'restore_checkpoint', 'nDevices', 'backend'
        ]

        descriptive_names = {
            'omega': 'Omega',
            'nx': 'Grid Points in X',
            'ny': 'Grid Points in Y',
            'nz': 'Grid Points in Z',
            'd': 'Dimensionality',
            'lattice': 'Lattice Type',
            'precision': 'Precision used for computation',
            'write_precision': 'Precision used for writing the output files',
            'total_timesteps': 'Total timesteps run in the simulation',
            'conv_param': 'Conversion parameters used to convert between the lattice and SI units',
            'write_start': 'Timestep from which output export begins',
            'write_control': 'The rate of output export in terms of timesteps',
            'create_log': 'Create a log file for the simulation',
            'output_dir': 'Directory used for output export',
            'compute_mlups': 'Determines if the MLUPS will be computed',
            'checkpoint_rate': 'Rate at which files are generated',
            'checkpoint_dir': 'Directory where checkpoint files are written',
            'restore_checkpoint': 'Start simulation from a checkpoint instead of beginning',
            'nDevices': 'Number of Devices',
            'backend': 'Backend'
        }
        simulation_name = self.__class__.__name__
        
        print(colored(f'**** Simulation Parameters for {simulation_name} ****', 'green'))
                
        header = f"{colored('Parameter', 'blue'):>30} | {colored('Value', 'yellow')}"
        print(header)
        print('-' * 50)
        
        for attr in attributes_to_show:
            value = getattr(self, attr, 'Attribute not set')
            descriptive_name = descriptive_names.get(attr, attr)  # Use the attribute name as a fallback
            row = f"{colored(descriptive_name, 'blue'):>30} | {colored(value, 'yellow')}"
            print(row)
            
    def compute_macroscopic_variables(self,f):
        """
            Compute the macroscopic variables density (rho) and velocity (u) using the distributions.

            Arguments:
                f: array[float]
                    Distribution array, storing distribution for all lattice nodes for all directions. 

            Returns:
                rho: array[float]
                    Density at each lattice nodes. 
                u: array[int]
                    Velocity at each lattice nodes.
        """
        rho = jnp.sum(f, axis=-1)
        u = jnp.dot(f, self.e) / rho
        return rho, u
    
    @partial(jit, static_argnums=(0,2))
    def equillibrium(self,rho,u):
        """
            Compute the equillibrium distribution function using the given values of density and velocity.

            Arguments:
                rho: Array-like
                    Density values.
                u: Array-like
                    Velocity values. 

            Returns:
                feq: Array-like
                    Equillibrium distribution.
        """
        udote = jnp.dot(u,self.e)
        udotu = jnp.sum(jnp.square(u), axis=-1, keepdims=True, dtype=self.precision)
        feq = jnp.array(self.w, dtype=self.precision) * (rho * (1.0 + (3.0 + 4.5*udote)*udote - 1.5*udotu))
        return feq

    @partial(jit, static_argnums=(0, 3), donate_argnums=(1,))
    def collision(self,fin,rho,u):
        """
            Single GPU implementation of the collision step in the Lattice Boltzmann Method. Defined in collision model sub-class.

            Arguments:
                fin: Array-like
                    Distribution function.
                rho: Array-like
                    Density at all the lattice nodes.
                u: Array-like
                    Velocity at all the lattice nodes.
            
            Returns:
                fout: Array-like
                    Distribution function.
        """
        pass

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def stream(self,f):
        """
            Single GPU implementation of streaming step in the Lattice Boltzmann Method. 
            Used for performing streaming and for computing the solid-mask.
            
            Arguments:
                f: Array-like
                    Any array which needs to be streamed. 
                    Can be the distribution array or the solid mask array (To create grid mask array).

            Returns:
                f: Array-like
                    The array obtained after streaming operation.
        """
        e = self.e
        f = jnp.roll(f,(e[0], e[1], e[2]),axis=(0, 1, 2))
        return f

    @partial(jit, static_argnums=(0,))
    def create_grid_mask(self):
        """
            Create the binary mask for the known and unknown directions in the lattice.

            Arguments:
                None

            Returns:
                None
        """
        self.grid_mask = self.stream(self.solid_mask)

    @partial(jit, static_argnums=(0,4))
    def apply_boundary_conditions(self,fout,fin,timestep,implementation_step):
        """
            Apply the boundary condition to the grid points identified in the boundary_indices (see boundary_conditions.py)

            Arguments:
                fout: Array-like
                    Output distribution function.
                fin: Array-like
                    Input distribution function.
                timestep: int
                    Timestep to be used for applying the boundary condition. 
                    Useful for dynamic boundary conditions, such as moving wall boundary condition.
                implementation_step: str
                    The implementation step is matched for boundary condition for all the lattice points.
            
            Returns:
                fout: Array-like
                    Output distribution values at lattice nodes.
        """
        fout = fin
        for bc in self.boundary_conditions:
            fout = bc.apply(fout,fin,timestep,implementation_step)

        return fout

    def initialize_arrays(self):
        """
            Create and initialize jax.numpy array with a given precision for f, rho and u

            Initial value of the distribution is the equillibrium value as determined by rho0 and u0

            Arguments:
                None
        """
        if self.d == 2:
            rho = jnp.ones((self.nx,self.ny,self.q),dtype=self.precision)
            u = jnp.zeros((self.nx,self.ny,self.q),dtype=self.precision)
        elif self.d == 3:
            rho = jnp.ones((self.nx,self.ny,self.nz,self.q),dtype=self.precision)
            u = jnp.zeros((self.nx,self.ny,self.nz,self.q),dtype=self.precision)

        # Assign the initial value to the density and velocity arrays
        rho = rho.at[:].set(self.rho0)
        u = u.at[:].set(self.u0)

        # Initialize the distribution array
        f = self.equillibrium(rho,u)
        return f, rho, u

    @partial(jit, static_argnums=(0,1))
    def calculate_mlups(self,t_total):
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
            mlups = (self.nx * self.ny * self.nz * self.total_timesteps) / (t_total * 1e6)
        return mlups
    
    @partial(jit, static_argnums=(0,3))
    def step(self,fin,rho,u):
        """
            Perform one step of LBM simulation. The sequence of operation is:
            1. Collide
            2. Apply/store values of distribution for specific nodes, if necessary (useful for the halfway bounce-back boundary condition)
            3. Stream
            4. Apply boundary conditions for the specific nodes.
            5. Modify the distribution if corresponding body force model is present.
            6. Apply body force to the macroscopic flow variables if corresponding body force model is present.
            7. Return the distribution the macroscopic flow data.

            Arguments:
                fin: Array-like
                    Input distribution function.
                rho: Array-like
                    Density values.
                u: Array-like
                    Velocity values.

            Returns:
                fout: Array-like
                    Output distribution function.
                rho: Array-like
                    Density values.
                u: Array-like
                    Velocity values.
        """
        f_postcollision = self.collision(fin,rho,u)
        f_postcollision = self.apply_boundary_conditions(f_postcollision,"post_collision")
        f_poststreaming = self.stream(f_postcollision)
        fout = self.apply_boundary_conditions(f_poststreaming,"post_streaming")

        # First apply body force using models where the distribution is modified
        fout, rho, u = self.body_force.apply(fout,rho,u,"distribution")

        # Compute new macroscopic flow variables if the distribution has been modified
        rho, u = self.compute_macroscopic_variables(fout)

        # Modify the macroscopic variables if the force model requires them to be modified instead of the distribution
        fout = self.body_force.apply(fout,rho,u,"macroscopic")
        rho, u = self.compute_macroscopic_variables(fout)

        return fout, rho, u

    def run(self):
        """
            Run the LBM simulation using the given parameters. 
            The sequence of operations are:
            1. Initialization
            2. Compute a single LBM step
            3. Export the flow variables depending on the export parameters.
            4. Post simulation steps (logging, MLUPS calculation etc....)

            Arguments:
                None
        """
        f, rho, u = self.initialize_arrays()
        t_total = 0.0

        # Perform the computation
        for t in range(self.total_timesteps):
            t_start = time.time() 
            f, rho, u = self.step(f,rho,u)
            t_end = time.time()
            t_total += (t_end - t_start)

            if (t - self.write_start > 0) and ((t - self.write_start) % self.write_control == 0):
                write_vtk(self.filename_prefix,t,rho,u,self.conv_param,self.lattice,self.precision)

        if self.compute_mlups:
            jax.block_until_ready(f)
            mlups = self.calculate_mlups(t_total)
