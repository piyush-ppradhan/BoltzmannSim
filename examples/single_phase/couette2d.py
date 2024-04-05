"""
This script performs a 2D simulation of Couette flow using the lattice Boltzmann method (LBM). 
"""
import os

import numpy as np

from src.boundary_conditions import *
from src.collision_models import BGK
from src.lattice import D2Q9
from src.utilities import *

# config.update('jax_disable_jit', True)
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'

class Couette(BGK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        walls = np.concatenate((self.bounding_box_indices["top"], self.bounding_box_indices["bottom"]))
        self.boundary_conditions.append(BounceBack(tuple(walls.T), self.grid_info, self.precision_policy))

        outlet = self.bounding_box_indices["right"]
        inlet = self.bounding_box_indices["left"]

        rho_wall = np.ones((inlet.shape[0], 1), dtype=self.precision_policy.compute_dtype)
        vel_wall = np.zeros(inlet.shape, dtype=self.precision_policy.compute_dtype)
        vel_wall[:, 0] = prescribed_vel
        self.boundary_conditions.append(EquilibriumBC(tuple(inlet.T), self.grid_info, self.precision_policy, rho_wall, vel_wall))

        self.boundary_conditions.append(DoNothing(tuple(outlet.T), self.grid_info, self.precision_policy))

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs["rho"][..., 1:-1, :])
        u = np.array(kwargs["u"][..., 1:-1, :])
        timestep = kwargs["timestep"]
        u_prev = kwargs["u_prev"][..., 1:-1, :]

        u_old = np.linalg.norm(u_prev, axis=2)
        u_new = np.linalg.norm(u, axis=2)
        err = np.sum(np.abs(u_old - u_new))
        print("error= {:07.6f}".format(err))
        save_image(timestep, u)
        fields = {"rho": rho[..., 0], "u_x": u[..., 0], "u_y": u[..., 1]}
        write_vtk("output", "data", timestep, fields)

if __name__ == "__main__":
    precision = "f32"
    lattice = D2Q9(precision)
    nx = 501
    ny = 101

    Re = 100.0
    prescribed_vel = 0.1
    clength = nx - 1

    visc = prescribed_vel * clength / Re

    checkpoint_rate = 1000
    checkpoint_dir = os.path.abspath("./checkpoints")

    omega = 1.0 / (3.0 * visc + 0.5)
    assert omega < 1.98, "omega must be less than 2.0"
    os.system("rm -rf ./*.vtk && rm -rf ./*.png")

    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': 0,
        'total_timesteps': 20000,
        'write_precision': precision,
        'write_start': 100,
        'write_control': 500,
        'output_dir': "output",
        'print_info_rate': 100,
        'checkpoint_rate': checkpoint_rate,
        'checkpoint_dir': checkpoint_dir,
        'restore_checkpoint': False
    }

    sim = Couette(**kwargs)
    sim.run()
