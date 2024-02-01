from functools import partial
from jax import jit
import jax.numpy as jnp
import numpy as np
from body_force import *
from utilities import *
import time

class LBMBase(object):
    """
        Base class to define all LBM solvers and problems

        Attributes:
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
                Initial value of density at each grid point. Dimension: (nx,ny) or (nx,ny,nz). Default: 1.0 for all grid points
            u0: array[float]
                Initial value of density at each grid point. Dimension: (nx,ny,2) or (nx,ny,nz,2). Default: 0.0 for all grid points
            conv_param: ConversionParameter
                Conversion parameters to convert between Lattice Units and SI units
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
            compute_MLUPS: bool
                Compute the Million Lattice Updates per Second (MLUPS) to evaluate the performance of the code. Default: False
            restart_flag: bool {optional}
                Restart the simulation. Useful if the simulation crashes before completion. Default: False
            restart_at: int {optional}
                Ignored if restart_flag is set to False. Default: 0
    """
    def __init__(self, **kwargs):
        try:
            self.nx = kwargs.get("nx")
            self.ny = kwargs.get("ny")
            self.nz = kwargs.get("nz",0)
            self.lattice = kwargs.get("lattice")
            self.body_force = kwargs.get("body_force",NoBodyForce())
            self.precision = self.lattice.precision # Used for computation as well as storage of data
            self.solid_mask = kwargs.get("solid_mask")
            self.total_timesteps = kwargs.get("total_timesteps")
            self.rho0 = kwargs.get("rho0")
            self.u0 = kwargs.get("u0")
            self.conv_param = kwargs.get("conv_param")
            self.filename_prefix = kwargs.get("filename_prefix","./output")
            self.write_start = kwargs.get("write_start")
            self.write_control = kwargs.get("write_control")
            self.create_log = kwargs.get("create_log",True)
            self.output_dir = kwargs.get("output_dir")
            self.compute_MLUPS = kwargs.get("compute_MLUPS",False)
            self.restart_flag = kwargs.get("restart_flag", False)
            self.restart_at = kwargs.get("restart_at", 0)
        except KeyError as e:
            print("Missing parameter: {e.args[0]}")

    def compute_macroscopic_variables(self,f):
        """
            Compute the macroscopic variables density (rho) and velocity (u) using the distributions.

            Arguments:
                f: array[float]
                    Distribution array, storing distribution for all lattice nodes for all directions. Dimension: (nx,ny,q) or (nx,ny,q)

            Returns:
                rho: array[float]
                    Density at each lattice nodes. Dimension: (nx,ny,1) or (nx,ny,nz,1)
                u: array[int]
                    Velocity at each lattice nodes. Dimension: (nx,ny,2) or (nx,ny,nz,2)
        """
        rho = jnp.sum(f,axis=-1)
        e = jnp.array(self.lattice.e)
        u = jnp.dot(f,e) / rho
        return rho,u
    
    @partial(jit, static_argnums=(0,2))
    def equillibrium(self,rho,u):
        e = jnp.transpose(jnp.array(self.lattice.e,dtype=self.precision))
        udote = jnp.dot(u,e)
        udotu = jnp.sum(jnp.square(u),axis=-1,keepdims=True,dtype=self.precision)
        feq = jnp.array(self.lattice.w,dtype=self.precision) * (rho * (1.0 + (3.0 + 4.5*udote)*udote - 1.5*udotu))
        return feq

    @partial(jit, static_argnums=(0,3), donate_argnums=(1,))
    def collision(self,f,rho,u):
        """
            Single GPU implementation of the collision step in the Lattice Boltzmann Method
            Dependent on the collision model choosen.
        """
        pass

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def stream(self,f):
        """
            Single GPU implementation of streaming step in the Lattice Boltzmann Method
            
            Arguments:
                f: array[float]
                    Any array which needs to be streamed. Can be the distribution array or the solid mask array
                    (To create grid mask array)
        """
        e = self.lattice.e
        f = jnp.roll(f,(e[0],e[1],e[2]),axis=(0,1,2))
        return f

    @partial(jit, static_argnums=(0,))
    def create_grid_mask(self):
        self.grid_mask = self.stream(self.solid_mask)

    @partial(jit, static_argnums=(0,1))
    def apply_boundary_conditions(self,f):
        pass

    def initialize_arrays(self):
        """
            Create and initialize jax.numpy array with a given precision for f, rho and u

            Initial value of the distribution is the equillibrium value as determined by rho0 and u0

            Arguments:
                None
        """
        if self.lattice.d == 2:
            rho = jnp.ones((self.nx,self.ny,self.lattice.q),dtype=self.precision)
            u = jnp.zeros((self.nx,self.ny,self.lattice.q),dtype=self.precision)
        elif self.lattice.d == 3:
            rho = jnp.ones((self.nx,self.ny,self.nz,self.lattice.q),dtype=self.precision)
            u = jnp.zeros((self.nx,self.ny,self.nz,self.lattice.q),dtype=self.precision)

        # Assign the initial value to the density and velocity arrays
        rho = rho.at[:].set(self.rho0)
        u = u.at[:].set(self.u0)

        # Initialize the distribution array
        f = self.equillibrium(rho,u)
        
        return f, rho, u

    @partial(jit, static_argnums=(0,1))
    def calculate_MLUPS(self,t_total):
        """
            Calculate the performance of the LBM code using MLUPS: Million Lattice Updates Per Second (MLUPS)
            Formula:
                (mesh_size * no_of_iterations) / (total_running_time * 1e6)

                where:
                    mesh_size = nx*ny in 2D and nx*ny*nz in 3D
                    total_time = t_simulation_stop - t_simulation_time
        """
        if self.lattice.d == 2:
            mlups = (self.nx * self.ny * self.total_timesteps) / (t_total * 1e6)
        else: 
            mlups = (self.nx * self.ny * self.nz * self.total_timesteps) / (t_total * 1e6)
        return mlups
    
    @partial(jit, static_argnums=(0,3))
    def step(self,f,rho,u):
        f = self.collision(f,rho,u)
        f = self.apply_boundary_conditions(f)
        f = self.stream(f)
        f = self.apply_boundary_conditions(f)
        f = self.body_force.apply(f,rho,u)
        if isinstance(self.body_force,NoBodyForce):
            rho, u = self.compute_macroscopic_variables(f)
        elif isinstance(self.body_force,ShanChenForce):
            rho, u = self.compute_macroscopic_variables(f)
            rho, u = self.body_force.apply(f,rho,u)
        elif isinstance(self.body_force,GuoBodyForce):
            f = self.body_force.apply(f,rho,u)
            rho, u = self.compute_macroscopic_variables(f)
        else:
            TypeError("Invalid body force type. Valid body forces type are: NoBodyForce(default), ShanChenForce, GuoBodyForce")
        return f, rho, u

    def run(self):
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

        if self.compute_MLUPS:
            mlups = self.calculate_MLUPS(t_total)
            
