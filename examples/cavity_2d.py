from conversion_parameters import *
from base import *
from collision_models import *
from lattice import *
from boundary_conditions import *
from utilities import *
import numpy as np

class Cavity2D(BGK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):

        # concatenate the indices of the left, right, and bottom walls
        walls = np.concatenate((self.bounding_box_indices["left"], self.bounding_box_indices["right"], self.bounding_box_indices["bottom"]))
        # apply bounce back boundary condition to the walls
        self.boundary_conditions.append(HalfwayBounceBack(tuple(walls.T), self.grid_info, self.precision_policy))

        # apply inlet equilibrium boundary condition to the top wall
        moving_wall = self.bounding_box_indices["top"]

        rho_wall = np.ones((moving_wall.shape[0], 1), dtype=self.precision_policy.compute_dtype)
        vel_wall = np.zeros(moving_wall.shape, dtype=self.precision_policy.compute_dtype)
        vel_wall[:, 0] = prescribed_vel
        self.boundary_conditions.append(EquilibriumBC(tuple(moving_wall.T), self.grid_info, self.precision_policy, rho_wall, vel_wall))

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs["rho"][1:-1, 1:-1])
        u = np.array(kwargs["u"][1:-1, 1:-1, :])
        timestep = kwargs["timestep"]

        fields = {"rho": rho[..., 0], "ux": u[..., 0], "uy": u[..., 1]}
        write_vtk("output", "data", timestep, fields, self.conv_param)

precision = "f32"
lattice = D2Q9(precision)

nx = 200
ny = 200

Re = 200.0
prescribed_vel = 0.1
clength = nx - 1

checkpoint_rate = 1000
checkpoint_dir = os.path.abspath("./checkpoints")

visc = prescribed_vel * clength / Re
omega = 1.0 / (3.0 * visc + 0.5)

os.system("rm -rf ./*.vtk && rm -rf ./*.png")

kwargs = {
    'lattice': lattice,
    'conversion_parameters': NoConversion(),
    'omega': omega,
    'nx': nx,
    'ny': ny,
    'nz': 0,
    'total_timesteps': 5000,
    'write_precision': precision,
    'write_start': 100,
    'write_control': 500,
    'output_dir': "output",
    'print_info_rate': 100,
    'checkpoint_rate': checkpoint_rate,
    'checkpoint_dir': checkpoint_dir,
    'restore_checkpoint': False,
}

sim = Cavity2D(**kwargs)
sim.run()
