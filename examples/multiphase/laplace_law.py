from src.multiphase import *
from src.utilities import *
from src.lattice import D3Q19

def laplace_law(**kwargs):
    _Eos = [ShanChen, VanderWaals, Redlich_Kwong, Redlich_Kwong_Soave, Peng_Robinson, Carnahan_Starling]
    rho_vap = 1.0
    rho_liq = 2.0
    r = 10
    for eos in _Eos:
        if eos is not Carnahan_Starling:
            a = 2/49
            b = 2/21
        elif eos is VanderWaals:
            a = 9/49
            b = 2/21
        else:
            a = 1.0
            b = 4.0
        R = 1.0
        T = 1.0
        class LaplaceLaw(eos):
            def initialize_macroscopic_fields(self):
                x = np.linspace(0, self.nx-1, self.nx, dtype=int)
                y = np.linspace(0, self.ny-1, self.ny, dtype=int)
                z = np.linspace(0, self.nz-1, self.nz, dtype=int)
                x, y, z = np.meshgrid(x, y, z)
                sphere = (x - self.nx // 2)**2 + (y - self.ny // 2)**2 + (z - self.nz // 2)**2 - r**2
                idx, idy, idz = np.where(sphere <= 0)

                rho = rho_vap * np.ones((self.nx, self.ny, self.nz, 1))
                rho[idx, idy, idz] = rho_liq
                rho_tree = [rho]
                u_tree = [np.zeros((self.nx, self.ny, self.nz, 3))]
                return rho_tree, u_tree
                
            def output_data(self, **kwargs):
                # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
                rho = np.array(kwargs["rho"][1:-1, 1:-1])
                u = np.array(kwargs["u"][1:-1, 1:-1, :])
                timestep = kwargs["timestep"]
                save_image(timestep, u)
                fields = {"rho": rho[..., 0], "ux": u[..., 0], "uy": u[..., 1]}
                write_vtk("output", "data", timestep, fields)

    kwargs = {
        'lattice': D3Q19("f32"),
        'omega': [1.0],
        'nx': 50,
        'ny': 50,
        'nz': 50,
        'n_components': 1,
        'gas_constant': 1.0,
        'g_kkprime': [1.0],
        'g_kw': [1.0],
        'a': a,
        'b': b,
        'R': R,
        'T': T,
        'alpha': 1.1,
        'total_timesteps': 30000,
        'write_precision': "f32",
        'write_start': 100,
        'write_control': 1000,
        'output_dir': "output_" + eos.__name__,
        'print_info_rate': 100,
        'checkpoint_rate': 1000,
        'checkpoint_dir': os.path.abspath("./checkpoints_") + eos.__name__,
        'restore_checkpoint': False,
    } 
    sim = LaplaceLaw(**kwargs)
    sim.run()

laplace_law()
