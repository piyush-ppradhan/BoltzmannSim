import os

import numpy as np

from src.boundary_conditions import *
from src.collision_models import BGK
from src.lattice import D2Q9
from src.utilities import *

np.random.seed(0)
p_inlet = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
p_outlet = np.array([0.9, 1.0, 1.0, 1.0, 1.0, 1.0])

Q = []
dP = []

def darcy_law():
    for i in range(5):
        class DarcyLaw(BGK):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def set_boundary_conditions(self):
                # Create a mask for porous media
                mask = ~(np.random.rand(self.nx, self.ny) > 0.5)
                mask[[0, self.nx-1], :] = 0
                x_ind, y_ind = np.where(mask == 1)

                # porous media
                walls = np.array([x_ind, y_ind])
                # apply bounce back boundary condition to the walls
                self.boundary_conditions.append(HalfwayBounceBack(tuple(walls), self.grid_info, self.precision_policy))

                # Pressure boundary condition at the inlet and outlet
                ind = np.zeros((self.ny, 2), dtype=int)
                ind[:, 0] = 0
                ind[:, 1] = np.arange(self.ny)
                p_prescribed = p_inlet[i] * np.ones((np.shape(ind)[0], 1), dtype=self.precision_policy.compute_dtype)
                self.boundary_conditions.append(ZouHe(tuple(ind.T), self.grid_info, self.precision_policy, 'pressure', p_prescribed))

                ind[:, 0] = self.nx-1
                ind[:, 1] = np.arange(self.ny)
                p_prescribed = p_outlet[i] * np.ones(np.shape(ind)[0])
                self.boundary_conditions.append(ZouHe(tuple(ind.T), self.grid_info, self.precision_policy, 'pressure', p_prescribed))


            def output_data(self, **kwargs):
                # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
                rho = np.array(kwargs["rho"][1:-1, 1:-1])
                u = np.array(kwargs["u"][1:-1, 1:-1, :])
                timestep = kwargs["timestep"]
                save_image(timestep, u)
                fields = {"rho": rho[..., 0], "ux": u[..., 0], "uy": u[..., 1]}
                write_vtk("output", "data", timestep, fields)

                Q.append(np.mean(self.ny * u))
                dP.append(p_inlet[i] - p_outlet[i])

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

        sim = DarcyLaw(**kwargs)
        sim.run()

    # Save the results to csv
    data = np.array([Q, dP]).T
    np.savetxt("darcy_law.csv", data, delimiter=",", header="Q,dP", comments="")


if __name__ == "__main__":
    darcy_law()
