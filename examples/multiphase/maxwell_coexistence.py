"""
Maxwell coexistence test case
"""

from src.multiphase import *
from src.utilities import *
from src.lattice import D2Q9

EOS = [VanderWaals, Peng_Robinson, Redlich_Kwong, Redlich_Kwong_Soave, Carnahan_Starling]
for eos in EOS:
    class MaxwellCoexistence(eos):
        def output_data(self, **kwargs):
            # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
            rho = np.array(kwargs["rho"][1:-1, 1:-1])
            u = np.array(kwargs["u"][1:-1, 1:-1, :])
            timestep = kwargs["timestep"]
            save_image(timestep, u)
            fields = {"rho": rho[..., 0], "ux": u[..., 0], "uy": u[..., 1]}
            write_vtk("output", "data", timestep, fields)

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
        'total_timesteps': 30000,
        'write_precision': precision,
        'write_start': 100,
        'write_control': 500,
        'output_dir': "output",
        'print_info_rate': 100,
        'checkpoint_rate': checkpoint_rate,
        'checkpoint_dir': checkpoint_dir,
        'restore_checkpoint': False,
    }

    sim = MaxwellCoexistence(**kwargs)
    sim.run()