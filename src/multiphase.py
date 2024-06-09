"""
Definition of Multiphase class for simulating a multiphase flow.
"""

import operator
import time
# System libraries
from functools import partial

import jax
import jax.numpy as jnp
# Third-party libraries
import numpy as np
import orbax.checkpoint as orb
from jax import jit
from jax.experimental.multihost_utils import process_allgather
from jax.tree_util import tree_map, tree_reduce
from termcolor import colored

# User-defined libraries
from src.collision_models import BGK
from src.utilities import downsample_field


class Multiphase(BGK):
    """
    Multiphase model, based on the Shan-Chen method. To model the fluid, an equation of state (EOS) is defined by the user.
    Sequence of computation is pressure (EOS, dependent on the density and temperature) --> effective mass (phi).
    Can model both single component multiphase (SCMP) and multi-component multiphase (MCMP).

    Attributes:
        R: float
            Gas constant
        T: float
            Temperature
        g_kk: numpy.ndarray
            Inter component interaction strength. Its a matrix of size n_components x n_components. It must be symmetric.
        g_ks: list
            Component-wall interaction strength. Its a vector of size (n_components,).

    Reference:
        1. Shan, Xiaowen, and Hudong Chen. “Lattice Boltzmann Model for Simulating Flows with Multiple Phases and Components.”
           Physical Review E 47, no. 3 (March 1, 1993): 1815-19. https://doi.org/10.1103/PhysRevE.47.1815.

        2. Yuan, Peng, and Laura Schaefer. “Equations of State in a Lattice Boltzmann Model.”
           Physics of Fluids 18, no. 4 (April 3, 2006): 042101. https://doi.org/10.1063/1.2187070.

        3. Pan, C., M. Hilpert, and C. T. Miller. “Lattice-Boltzmann Simulation of Two-Phase Flow in Porous Media.”
           Water Resources Research 40, no. 1 (2004). https://doi.org/10.1029/2003WR002120.

        4. Kang, Qinjun, Dongxiao Zhang, and Shiyi Chen. “Displacement of a Two-Dimensional Immiscible Droplet in a Channel.”
           Physics of Fluids 14, no. 9 (September 1, 2002): 3203-14. https://doi.org/10.1063/1.1499125.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.R = kwargs.get("gas_constant", 0.0)
        self.T = kwargs.get("temperature", 0.0)
        self.n_components = kwargs.get("n_components")
        self.g_kkprime = kwargs.get("g_kkprime")
        self.g_ks = kwargs.get("g_ks")
        self.force = 0.0

        self.G_ff = self.compute_ff_greens_function()
        self.G_fs = self.compute_fs_greens_function()

        # This is used for fluid-solid force computation
        self.solid_mask_repeated = jnp.repeat(
            jnp.expand_dims(self.solid_mask, axis=-1), repeats=self.q, axis=-1
        )

        self.omega = jnp.array(self.omega, dtype=self.precision_policy.compute_dtype)
        self.g_kkprime = jnp.array(
            self.g_kkprime, dtype=self.precision_policy.compute_dtype
        )
        self.g_ks = jnp.array(self.g_ks, dtype=self.precision_policy.compute_dtype)

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value):
        if not isinstance(value, list) and not isinstance(value, jax.numpy.ndarray):
            raise ValueError("omega must be a list or jax.numpy.ndarray")
        self._omega = value

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, value):
        self._R = value

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        self._T = value

    @property
    def n_components(self):
        return self._n_components

    @n_components.setter
    def n_components(self, value):
        if value is None:
            raise ValueError("n_components must be provided for multiphase simulation")
        if not isinstance(value, int):
            raise ValueError("n_components must be an integer")
        self._n_components = value

    @property
    def g_kkprime(self):
        return self._g_kkprime

    @g_kkprime.setter
    def g_kkprime(self, value):
        if not isinstance(value, np.ndarray) and not isinstance(
            value, jax.numpy.ndarray
        ):
            raise ValueError("g_kkprime must be a numpy array or jax.numpy.ndarray")
        if np.shape(value) != (self.n_components, self.n_components):
            raise ValueError(
                "g_kkprime must be a matrix of size n_components x n_components"
            )
        if not np.allclose(value, np.transpose(value), atol=1e-6):
            raise ValueError("g_kkprime must be a symmetric matrix")
        self._g_kkprime = np.array(value)

    @property
    def g_ks(self):
        return self._g_ks

    @g_ks.setter
    def g_ks(self, value):
        if len(value) != self.n_components:
            raise ValueError("g_ks must be a list size n_components")
        self._g_ks = np.array(value)

    @partial(jit, static_argnums=(0, 3))
    def equilibrium(self, rho_tree, u_tree, cast_output=True):
        """
        Compute the equillibrium distribution function using the given values of density and velocity.

        Arguments:
            rho_tree: pytree of jax.numpy.ndarray
                Pytree of density values.
            u_tree: jax.numpy.ndarray
                Pytree of velocity values.
            cast_output: bool {Optional}
                A flag to cast the density and velocity values to the compute and output precision. Default: True

        Returns:
            feq: pytree of jax.numpy.ndarray
                Pytree of equillibrium distribution.
        """
        if cast_output:
            cast = lambda rho, u: self.precision_policy.cast_to_compute((rho, u))
            rho_tree, u_tree = tree_map(cast, rho_tree, u_tree)

        e = self.precision_policy.cast_to_compute(self.e)
        udote_tree = tree_map(lambda u: jnp.dot(u, e), u_tree)
        udotu_tree = tree_map(
            lambda u: jnp.sum(jnp.square(u), axis=-1, keepdims=True), u_tree
        )
        feq_tree = tree_map(
            lambda rho, udote, udotu: rho
            * self.w
            * (1.0 + udote * (3.0 + 4.5 * udote) - 1.5 * udotu),
            rho_tree,
            udote_tree,
            udotu_tree,
        )

        if cast_output:
            return tree_map(
                lambda f_eq: self.precision_policy.cast_to_output(f_eq), feq_tree
            )
        else:
            return feq_tree

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, fin_tree):
        fin_tree = tree_map(
            lambda fin: self.precision_policy.cast_to_compute(fin), fin_tree
        )
        rho_tree, u_tree = self.compute_macroscopic_variables(fin_tree)
        feq_tree = self.equilibrium(rho_tree, u_tree, cast_output=False)
        fneq_tree = tree_map(lambda feq, fin: feq - fin, feq_tree, fin_tree)
        fout_tree = tree_map(
            lambda fin, fneq, omega: fin + omega * fneq, fin_tree, fneq_tree, self.omega
        )
        if self.force is not None:
            fout_tree = self.apply_force(fout_tree, feq_tree, rho_tree, u_tree)
        return tree_map(
            lambda fout: self.precision_policy.cast_to_output(fout), fout_tree
        )

    def compute_ff_greens_function(self):
        """
        Define the Green's function used to model interaction of kth fluid with all components.

        During computation, this G_ff is multiplied with corresponding g_kkprime value to get the Green's function:
        G_kkprime = self.g_kk[k, k_prime] * self.G_ff

        Green's function used in this case is:

        G_ff(x, x') = 1,         if |x - x'| = 1
                    = 1/sqrt(d), if |x - x'| = sqrt(d)
                    = 0,         otherwise

        which when multiplied with g_kkprime gives:

        G_kkprime(x, x') = gkkprime,         if |x - x'| = 1
                         = gkkprime/sqrt(d), if |x - x'| = sqrt(d)
                         = 0,                otherwise

        Here d is the dimension of problem and x' are the neighboring points.
        Note that the next neighbors have more interaction than next-nearest points

        Arguments:
            None

        Returns:
            G_ff: jax.numpy.ndarray
                Dimension: (q, )
        """
        e = np.array(self.lattice.e).T
        G_ff = np.zeros((self.q,), dtype=np.float64)
        el = np.linalg.norm(e, axis=-1)
        G_ff[np.isclose(el, 1.0, atol=1e-6)] = 1.0
        G_ff[np.isclose(el, np.sqrt(self.d), atol=1e-6)] = 1.0 / np.sqrt(self.d)
        return jnp.array(G_ff, dtype=self.precision_policy.compute_dtype)

    def compute_fs_greens_function(self):
        """
        Define the Green's function used to model interaction between kth fluid and solid.

        During computation, this G_fs is multiplied with corresponding g_ks value to get the Green's function:
        G_ks = self.g_ks[k] * self.G_fs

        Green's function used in this case:

        G_fs(x, x') = 1,              if |x - x'| = 1
                    = 1/sqrt(d),      if |x - x'| = sqrt(d)
                    = 0,              otherwise

        which when multiplied with g_ks gives:

        G_ks(x, x') = g_ks,           if |x - x'| = 1
                    = g_ks/sqrt(d),   if |x - x'| = sqrt(d)
                    = 0,              otherwise

        Again, d is the dimension of the problem

        Arguments:
            None

        Returns:
            G_fs: jax.numpy.ndarray
                Dimension: (q, )
        """
        e = np.array(self.lattice.e).T
        G_fs = np.zeros((self.q,), dtype=np.float64)
        el = np.linalg.norm(e, axis=-1)
        G_fs[np.isclose(el, 1.0, atol=1e-6)] = 1.0
        G_fs[np.isclose(el, np.sqrt(self.d), atol=1e-6)] = 1.0 / np.sqrt(self.d)
        return jnp.array(G_fs, dtype=self.precision_policy.compute_dtype)

    def initialize_macroscopic_fields(self):
        """
        Functions to initialize the pytrees of density and velocity arrays with their corresponding initial values.
        By default, velocities is set as 0 everywhere and density as 1.0.

        To use the default values, specify None for rho0 and u0 i.e.,

        rho0[i], u[i] = None, None

        Note:
            Function must be overwritten in a subclass or instance of the class to not use the default values.

        Arguments:
            None by default, can be overwritten as required

        Returns:
            None, None: The default density and velocity values, both None.
            This indicates that the actual values should be set elsewhere.
        """
        print(
            "Default initial conditions assumed for the missing entries in the dictionary: density = 1.0 and velocity = 0.0"
        )
        print(
            "To set explicit initial values for velocity and density, use the self.initialize_macroscopic_fields function"
        )
        return None, None

    def assign_fields_sharded(self):
        """
        This function is used to initialize pytree of the distribution arrays using the initial velocities and velocity defined in self.initialize_macroscopic_fields function.
        To do this, function first uses the initialize_macroscopic_fields function to get the initial values of rho (rho0) and velocity (u0).

        The distribution is initialized with rho0 and u0 values, using the self.equilibrium function.

        Arguments:
            None

        Returns:
            f: pytree of distributed JAX array of shape: (self.nx, self.ny, self.q) for 2D and (self.nx, self.ny, self.nz, self.q) for 3D.
        """
        rho0_tree, u0_tree = self.initialize_macroscopic_fields()
        # fmt:off
        shape = (
            (self.nx, self.ny, self.q) if (self.d == 2) else (self.nx, self.ny, self.nz, self.q)
        )
        # fmt:on
        f_tree = {}
        if rho0_tree is not None and u0_tree is not None:
            assert (
                len(rho0_tree) == self.n_components
            ), "The initial density values for all components must be provided"

            assert (
                len(u0_tree) == self.n_components
            ), "The initial velocity values for all components must be provided."

            for i in range(self.n_components):
                rho0, u0 = rho0_tree[i], u0_tree[i]
                f_tree[i] = self.initialize_distribution(rho0, u0)
        else:
            for i in range(self.n_components):
                f_tree[i] = self.distributed_array_init(
                    shape, self.precision_policy.output_dtype, init_val=self.w
                )
        return f_tree

    @partial(jit, static_argnums=(0,))
    def compute_rho(self, f_tree):
        """
        Compute the number density for all fluids using the respective distribution function.

        Arguments:
            f_tree: pytree of jax.numpy.ndarray
                Pytree of distribution arrays.

        Returns:
            rho_tree: pytree of jax.numpy.ndarray
                Pytree of density values.
        """
        rho_tree = tree_map(lambda f: jnp.sum(f, axis=-1, keepdims=True), f_tree)
        return rho_tree

    @partial(jit, static_argnums=(0,))
    def compute_macroscopic_variables(self, f_tree):
        """
        Compute the macroscopic variables (density (rho) and velocity (u)) using the distribution function.

        Arguments:
            f_tree: pytree of jax.numpy.ndarray
                Pytree of distribution arrays.

        Returns:
            rho_tree: pytree of jax.numpy.ndarray for component densities
            u_tree: pytree of jax.numpy.ndarray for component velocities
        """
        rho_tree = tree_map(lambda f: jnp.sum(f, axis=-1, keepdims=True), f_tree)
        u_tree = tree_map(
            lambda f, rho: jnp.dot(f, self.e.T) / rho, f_tree, rho_tree
        )  # Component velocity
        return rho_tree, u_tree

    @partial(jit, static_argnums=(0,))
    def compute_common_velocity(self, rho_tree, u_tree):
        """
        Compute the common velocity using component velocity and density values.

        Arguments:
            rho_tree: Pytree of jax.numpy.ndarray
                Pytree of component density values.
            u_tree: Pytree of jax.numpy.ndarray
                Pytree of component velocity values.

        Returns:
            jax.numpy.ndarray
                Common velocity values.
        """
        n = tree_reduce(
            operator.add,
            tree_map(
                lambda omega, rho, u: rho * u * omega, self.omega, rho_tree, u_tree
            ),
        )
        d = tree_reduce(
            operator.add, tree_map(lambda omega, rho: rho * omega, self.omega, rho_tree)
        )
        return n / d

    @partial(jit, static_argnums=(0,))
    def EOS(self, rho_tree):
        """
        Define the equation of state for the problem. Defined in sub-classes.

        Arguments:
            rho_tree: jax.numpy.ndarray
                Pytree of density values.

        Returns:
            p_tree: pytree of jax.numpy.ndarray
                Pytree of pressure values.
        """
        pass

    @partial(jit, static_argnums=(0,))
    def compute_psi(self, rho_tree):
        """
        Compute psi, the effective mass which is used for modelling the interaction of forces.
        This function uses the value of pressure obtained from EOS and density to compute psi.

        Arguments:
            rho_tree: pytree of jax.numpy.ndarray
                Pytree of density values.

        Returns:
            psi_tree: pytree of jax.numpy.ndarray
        """
        rho_tree = tree_map(
            lambda rho: self.precision_policy.cast_to_compute(rho), rho_tree
        )
        p_tree = self.EOS(rho_tree)
        g_tree = list(self.g_kkprime.diagonal())
        psi = (
            lambda p, rho, g: (
                2 * (p - self.lattice.c_s2 * rho) / (self.lattice.c_s2 * g)
            )
            ** 0.5
        )
        psi_tree = tree_map(
            psi, p_tree, rho_tree, g_tree
        )  # Exact value of g does not matter
        return psi_tree

    # Compute the force using the effective mass (psi) and the interaction potential (phi)
    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def compute_force(self, f_tree):
        """
        Compute the force acting on the fluids. This includes fluid-fluid, fluid-solid, and body forces.

        Arguments:
            f_tree: pytree of jax.numpy.ndarray
                Pytree of distribution array.

        Returns:
            Pytree of jax.numpy.ndarray
        """
        rho_tree = tree_map(lambda f: jnp.sum(f, axis=-1, keepdims=True), f_tree)
        psi_tree = self.compute_psi(rho_tree)
        fluid_fluid_force = self.compute_fluid_fluid_force(psi_tree)
        fluid_solid_force = self.compute_fluid_solid_force(rho_tree)
        return fluid_fluid_force + fluid_solid_force

    @partial(jit, static_argnums=(0,))
    def compute_fluid_fluid_force(self, psi_tree):
        """
        Compute the fluid-fluid interaction force using the effective mass (psi).

        Arguments:
            psi_tree: pytree of jax.numpy.ndarray
                Pytree of effective mass values.
            psi_s_tree: pytree of jax.numpy.ndarray
                Pytree of streamed effective mass values.

        Returns:
            pytree of jax.numpy.ndarray
                Pytree of fluid-fluid interaction force.
        """
        psi_s_tree = tree_map(
            lambda psi: self.streaming(jnp.repeat(psi, axis=-1, repeats=self.q)),
            psi_tree,
        )

        def ffk(g_kkprime):
            """
            g_kkprime is a row of self.gkkprime, as it represents the interaction between kth component with all components
            """
            return tree_map(
                lambda g_kkp, psi_s: jnp.dot(g_kkp * self.G_fs * psi_s, self.e),
                list(g_kkprime),
                psi_s_tree,
            )

        return tree_map(
            lambda psi, in_t: -psi * in_t,
            psi_tree,
            list(jax.vmap(ffk, in_axes=0)(self.g_kkprime)),
        )

    @partial(jit, static_argnums=(0,))
    def compute_fluid_solid_force(self, rho_tree):
        """
        Compute the fluid-fluid interaction force using the effective mass (psi).

        Arguments:
            rho_tree: Pytree of jax.numpy.ndarray
                Pytree of density of all components.

        Returns:
            Pytree of jax.numpy.ndarray
                Pytree of fluid-solid interaction force.
        """
        neighbor_terms = tree_map(
            lambda g_ks: jnp.dot(g_ks * self.G_fs * self.solid_mask_repeated, self.e),
            self.g_ks,
        )
        return tree_map(lambda rho: -rho * neighbor_terms, rho_tree)

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force(self, f_postcollision_tree, feq_tree, rho_tree, u_tree):
        """
        Modified version of the apply_force defined in LBMBase to account for modified force.

        Parameters:
            f_postcollision_tree: pytree of jax.numpy.ndarray
                pytree of post-collision distribution functions.
            feq: pytree of jax.numpy.ndarray
                pytree of equilibrium distribution functions.
            rho_tree: pytree of jax.numpy.ndarray
                pytree of density field for all components.
            u_tree: pytree of jax.numpy.ndarray
                pytree of velocity field for all components

        Returns:
            f_postcollision: jax.numpy.ndarray
                The post-collision distribution functions with the force applied.
        """
        delta_u_tree = (
            self.compute_force(f_postcollision_tree) + self.force
        )  # self.force is the external body force
        u_temp_tree = tree_map(lambda u, delta_u: u + delta_u, u_tree, delta_u_tree)
        feq_force_tree = self.equilibrium(rho_tree, u_temp_tree, cast_output=False)
        update_collision = (
            lambda f_postcollision, feq_force, feq: f_postcollision + feq_force - feq
        )
        return tree_map(
            update_collision, f_postcollision_tree, feq_force_tree, feq_tree
        )

    @partial(jit, static_argnums=(0, 4))
    def apply_boundary_conditions(
        self, fout_tree, fin_tree, timestep, implementation_step
    ):
        """
        Apply the boundary condition to the grid points identified in the boundary_indices (see boundary_conditions.py)

        Arguments:
            fout_tree: pytree of jax.numpy.ndarray
                pytree of output distribution function.
            fin_tree: pytree of jax.numpy.ndarray
                pytree of input distribution function.
            timestep: int
                Timestep to be used for applying the boundary condition.
                Useful for dynamic boundary conditions, such as moving wall boundary condition.
            implementation_step: str
                The implementation step is matched for boundary condition for all the lattice points.

        Returns:
            fout: Array-like
                Output distribution values at lattice nodes.
        """
        for bc in self.boundary_conditions:
            fout_tree = tree_map(
                lambda fin, fout: bc.prepare_populations(
                    fout, fin, implementation_step
                ),
                fin_tree,
                fout_tree,
            )
            if bc.implementation_step == implementation_step:
                if bc.is_dynamic:
                    fout_tree = tree_map(
                        lambda fin, fout: bc.apply(fout, fin, timestep),
                        fin_tree,
                        fout_tree,
                    )
                else:
                    fout_tree = tree_map(
                        lambda fin, fout: fout.at[bc.boundary_indices].set(
                            bc.apply(fout, fin)
                        ),
                        fin_tree,
                        fout_tree,
                    )
        return fout_tree

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def step(self, f_poststreaming_tree, timestep):
        """
        Perform one step of LBM simulation.

        Arguments:
            fin_tree: pytree of jax.numpy.ndarray
                pytree of post-streaming distribution function.
            timestep: int
                Current timestep

        Returns:
            f_poststreaming_tree: pytree of jax.numpy.ndarray
                pytree of post-streamed distribution function.
            f_collision_tree: pytree of jax.numpy.ndarray {Optional}
                pytree of post-collision distribution function.
        """
        f_postcollision_tree = self.collision(f_poststreaming_tree)
        f_postcollision_tree = self.apply_boundary_conditions(
            f_postcollision_tree, f_poststreaming_tree, timestep, "post_collision"
        )
        f_poststreaming_tree = tree_map(
            lambda f_postcollision: self.streaming(f_postcollision),
            f_postcollision_tree,
        )
        f_poststreaming_tree = self.apply_boundary_conditions(
            f_poststreaming_tree, f_postcollision_tree, timestep, "post_streaming"
        )

        if self.return_post_col_dist:
            return f_poststreaming_tree, f_postcollision_tree
        else:
            return f_poststreaming_tree, None

    def run(self):
        """
        This function runs the LBM simulation for a specified number of time steps.

        It first initializes the distribution functions and then enters a loop where it performs the
        simulation steps (collision, streaming, and boundary conditions) for each time step.

        The function can also print the progress of the simulation, save the simulation data, and
        compute the performance of the simulation in million lattice updates per second (MLUPS). How does this even work ?

        Arguments:
            None
        Returns:
            f_tree: pytree of jax.numpy.ndarray
                pytree of distribution functions after the simulation.
        """
        f_tree = {}
        for i in range(self.n_components):
            f_tree[i] = self.assign_fields_sharded()
        start_step = 0
        if self.restore_checkpoint:
            latest_step = self.mngr.latest_step()
            if latest_step is not None:  # existing checkpoint present
                # Assert that the checkpoint manager is not None
                assert self.mngr is not None, "Checkpoint manager does not exist."
                state = {}
                c_name = lambda i: f"component_{i}"
                for i in range(self.n_components):
                    state[c_name(i)] = f_tree[i]
                # shardings = jax.tree_map(lambda x: x.sharding, f_tree)
                # restore_args = orb.checkpoint_utils.construct_restore_args(
                #     f_tree, shardings
                # )
                abstract_state = jax.tree_util.tree_map(
                    orb.utils.to_shape_dtype_struct, state
                )
                try:
                    f_tree = self.mngr.restore(
                        latest_step,
                        # restore_kwargs={"restore_args": restore_args},
                        args=orb.args.StandardRestore(abstract_state),
                    )
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
                rho_prev_tree, u_prev_tree = self.compute_macroscopic_variables(f_tree)
                p_prev_tree = self.EOS(rho_prev_tree)
                rho_prev_tree = tree_map(
                    lambda rho_prev: downsample_field(
                        rho_prev, self.downsampling_factor
                    ),
                    rho_prev_tree,
                )
                p_prev_tree = tree_map(
                    lambda p_prev: downsample_field(p_prev, self.downsampling_factor),
                    p_prev_tree,
                )
                u_prev_tree = tree_map(
                    lambda u_prev: downsample_field(u_prev, self.downsampling_factor),
                    u_prev_tree,
                )
                rho_prev = tree_reduce(
                    operator.add,
                    tree_map(lambda omega, rho: omega * rho, self.omega, rho_prev_tree),
                )
                u_prev = tree_reduce(
                    operator.add,
                    tree_map(
                        lambda omega, rho, u: omega * rho * u,
                        self.omega,
                        rho_prev_tree,
                        u_prev_tree,
                    ),
                ) / tree_reduce(
                    operator.add,
                    tree_map(lambda omega, rho: omega * rho, self.omega, rho_prev_tree),
                )

                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                p_prev_tree = tree_map(
                    lambda p_prev: process_allgather(p_prev), p_prev_tree
                )
                rho_prev_tree = tree_map(
                    lambda rho_prev: process_allgather(rho_prev), rho_prev_tree
                )
                u_prev_tree = tree_map(
                    lambda u_prev: process_allgather(u_prev), u_prev_tree
                )
                u_prev = tree_map(lambda u: process_allgather(u), u_prev)

            # Perform one time-step (collision, streaming, and boundary conditions)
            f_tree, fstar_tree = self.step(f_tree, timestep)

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
                rho_tree, u_tree = self.compute_macroscopic_variables(f_tree)
                p_tree = self.EOS(rho_tree)
                p_tree = tree_map(
                    lambda p: downsample_field(p, self.downsampling_factor), p_tree
                )
                rho_tree = tree_map(
                    lambda rho: downsample_field(rho, self.downsampling_factor),
                    rho_tree,
                )
                u_tree = tree_map(
                    lambda u: downsample_field(u, self.downsampling_factor), u_tree
                )
                rho = tree_reduce(
                    operator.add,
                    tree_map(lambda omega, rho: omega * rho, self.omega, rho_tree),
                )
                u = tree_reduce(
                    operator.add,
                    tree_map(
                        lambda omega, rho, u: omega * rho * u,
                        self.omega,
                        rho_tree,
                        u_tree,
                    ),
                ) / tree_reduce(
                    operator.add,
                    tree_map(lambda omega, rho: omega * rho, self.omega, rho_tree),
                )

                # Gather the data from all processes and convert it to numpy arrays (move to host memory)
                p_tree = tree_map(lambda p: process_allgather(p), p_tree)
                rho_tree = tree_map(lambda rho: process_allgather(rho), rho_tree)
                u_tree = tree_map(lambda u: process_allgather(u), u_tree)
                u = tree_map(lambda u_: process_allgather(u_), u)

                # Save the data
                self.handle_io_timestep(
                    timestep,
                    f_tree,
                    fstar_tree,
                    p_tree,
                    u,
                    u_tree,
                    rho,
                    rho_tree,
                    p_prev_tree,
                    u_prev,
                    u_prev_tree,
                    rho_prev,
                    rho_prev_tree,
                )

            if checkpoint_flag:
                # Save the checkpoint
                print(
                    f"Saving checkpoint at timestep {timestep}/{self.total_timesteps}"
                )
                state = {}
                c_name = lambda i: f"component_{i}"
                for i in range(self.n_components):
                    state[c_name(i)] = f_tree[i]

                self.mngr.save(timestep, args=orb.args.StandardSave(state))

            # Start the timer for the MLUPS computation after the first timestep (to remove compilation overhead)
            if self.compute_mlups and timestep == 1:
                jax.block_until_ready(f_tree)
                start = time.time()

        if self.compute_mlups:
            # Compute and print the performance of the simulation in MLUPS
            jax.block_until_ready(f_tree)
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

        return f_tree

    def handle_io_timestep(
        self,
        timestep,
        f_tree,
        fstar_tree,
        p_tree,
        u,
        u_tree,
        rho,
        rho_tree,
        p_prev_tree,
        u_prev,
        u_prev_tree,
        rho_prev,
        rho_prev_tree,
    ):
        """
        This function handles the input/output (I/O) operations at each time step of the simulation.

        It prepares the data to be saved and calls the output_data function, which can be overwritten
        by the user to customize the I/O operations.

        Parameters:
            timestep: int
                The current time step of the simulation.
            f_tree: pytree of jax.numpy.ndarray
                Pytree of post-streaming distribution functions at the current time step.
            fstar_tree: pytree of jax.numpy.ndarray
                Pytree of post-collision distribution functions at the current time step.
            p_tree: pytree of jax.numpy.ndarray
                Pytree of pressure field at the current time step.
            u: jax.numpy.ndarray
                Common velocity field at the current time step.
            u_tree: pytree of jax.numpy.ndarray
                Pytree of velocity field at the current time step.
            rho: jax.numpy.ndarray
                Common density field at the current time step.
            rho_tree: pytree of jax.numpy.ndarray
                Pytree of density field at the current time step.
            p_prev_tree: pytree of jax.numpy.ndarray
                Pytree of pressure field at the previous time step.
            u_prev: jax.numpy.ndarray
                Common velocity field at the previous time step.
            u_prev_tree: pytree of jax.numpy.ndarray
                Pytree of velocity field at the previous time step.
            rho_prev: jax.numpy.ndarray
                Common density field at the previous time step.
            rho_prev_tree: pytree of jax.numpy.ndarray
                Pytree of density field at the previous time step.
        """
        kwargs = {
            "n_components": self.n_components,
            "timestep": timestep,
            "rho": rho,
            "rho_prev": rho_prev,
            "rho_tree": rho_tree,
            "rho_prev_tree": rho_prev_tree,
            "p_tree": p_tree,
            "p_prev_tree": p_prev_tree,
            "u": u,
            "u_prev": u_prev,
            "u_tree": u_tree,
            "u_prev_tree": u_prev_tree,
            "f_poststreaming_tree": f_tree,
            "f_postcollision_tree": fstar_tree,
        }
        self.output_data(**kwargs)


class VanderWaals(Multiphase):
    """
    Define multiphase model using the VanderWaals EOS.

    Attributes:

    Reference:
        1. Reprint of: The Equation of State for Gases and Liquids. The Journal of Supercritical Fluids,
        100th year Anniversary of van der Waals' Nobel Lecture, 55, no. 2 (2010): 403–14. https://doi.org/10.1016/j.supflu.2010.11.001.

    Notes:
        EOS is given by:
            p = (rho*R*T)/(1 - b*rho) - a*rho^2
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value is None:
            raise ValueError("a value must be provided for using VanderWaals EOS")
        self._a = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if value is None:
            raise ValueError("b value must be provided for using VanderWaals EOS")
        self._b = value

    @partial(jit, static_argnums=(0,))
    def EOS(self, rho_tree):
        rho_tree = tree_map(
            lambda rho: self.precision_policy.cast_to_compute(rho), rho_tree
        )
        eos = lambda rho: (rho * self.R * self.T) / (
            1.0 - self.b * rho
        ) - self.a * jnp.square(rho)
        return tree_map(eos, rho_tree)


class ShanChen(Multiphase):
    """
    Define the multiphase model using the original Shan-Chen EOS. For this class compute_psi is redefined.
    For this case, there is no need to define R and T as they are not used in the EOS.

    Attributes:
        rho_0: float
            rho_0 used for computing the effective mass (psi)

    Reference:
        1. Shan, Xiaowen, and Hudong Chen. “Lattice Boltzmann Model for Simulating Flows with Multiple Phases and Components.”
           Physical Review E 47, no. 3 (March 1, 1993): 1815-19. https://doi.org/10.1103/PhysRevE.47.1815.

    Notes:
        The expression for psi in this case is:
        psi = rho_0 * (1 - exp(-rho / rho_0))
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rho_0 = kwargs.get("rho_0")

    @property
    def rho_0(self):
        return self._rho_0

    @rho_0.setter
    def rho_0(self, value):
        if value is None:
            raise ValueError("rho_0 value must be provided Shan-Chen EOS")
        self._rho_0 = value

    @partial(jit, static_argnums=(0,))
    def compute_psi(self, rho_tree):
        psi = lambda rho: self.rho_0 * (1.0 - jnp.exp(-rho / self.rho_0))
        return tree_map(psi, rho_tree)

    @partial(jit, static_argnums=(0,))
    def EOS(self, rho_tree):
        rho_tree = tree_map(
            lambda rho: self.precision_policy.cast_to_compute(rho), rho_tree
        )
        rho_tree = self.precision_policy.cast_to_compute(rho_tree)
        psi_tree = self.compute_psi(rho_tree)
        eos = lambda rho, psi: self.lattice.c_s2 * rho + (
            0.5 * self.lattice.c0 * self.g * psi**2
        )
        return tree_map(eos, rho_tree, psi_tree)


class Redlich_Kwong(Multiphase):
    """
    Define multiphase model using the Redlich-Kwong EOS.

    Attributes:

    Reference:
        1. Redlich O., Kwong JN., "On the thermodynamics of solutions; an equation of state; fugacities of gaseous solutions."
        Chem Rev. 1949 Feb;44(1):233-44. https://doi.org/10.1021/cr60137a013.

    Notes:
        EOS is given by:
            p = (rho*R*T)/(1 - b*rho) - (a*rho^2)/(sqrt(T) * (1 + b*rho))
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value is None:
            raise ValueError("a value must be provided for using Redlich-Kwong EOS")
        self._a = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if value is None:
            raise ValueError("b value must be provided for using Redlich-Kwong EOS")
        self._b = value

    @partial(jit, static_argnums=(0,))
    def EOS(self, rho_tree):
        rho_tree = tree_map(
            lambda rho: self.precision_policy.cast_to_compute(rho), rho_tree
        )
        eos = lambda rho: (rho * self.R * self.T) / (1.0 - self.b * rho) - (
            self.a * rho**2
        ) / (self.T**0.5 * (1.0 + self.b * rho))
        return tree_map(eos, rho_tree)


class Redlich_Kwong_Soave(Multiphase):
    """
    Define multiphase model using the Redlich-Kwong-Soave EOS.

    Attributes:

    Reference:
        1. Giorgio Soave, "Equilibrium constants from a modified Redlich-Kwong equation of state",
        Chemical Engineering Science 27, no. 6(1972), 1197-1203, https://doi.org/10.1016/0009-2509(72)80096-4.

    Notes:
        EOS is given by:
            p = (rho*R*T)/(1 - b*rho) - (a*alpha*rho^2)/(1 + b*rho)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")
        self.alpha = kwargs.get("alpha")

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value is None:
            raise ValueError(
                "a value must be provided for using Redlich-Kwong-Soave EOS"
            )
        self._a = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if value is None:
            raise ValueError(
                "b value must be provided for using Redlich-Kwong-Soave EOS"
            )
        self._b = value

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value is None:
            raise ValueError("alpha value must be provided for using Redlich-Kwong EOS")
        self._alpha = value

    @partial(jit, static_argnums=(0,))
    def EOS(self, rho_tree):
        rho_tree = tree_map(
            lambda rho: self.precision_policy.cast_to_compute(rho), rho_tree
        )
        eos = lambda rho: (rho * self.R * self.T) / (1.0 - self.b * rho) - (
            self.a * self.alpha * rho**2
        ) / (1.0 + self.b * rho)
        return tree_map(eos, rho_tree)


class Peng_Robinson(Multiphase):
    """
    Define multiphase model using the Peng-Robinson EOS.

    Attributes:

    Reference:
        1. Peng, Ding-Yu, and Donald B. Robinson. "A new two-constant equation of state."
        Industrial & Engineering Chemistry Fundamentals 15, no. 1 (1976): 59-64. https://doi.org/10.1021/i160057a011

    Notes:
        EOS is given by:
            p = (rho*R*T)/(1 - b*rho) - (a*alpha*rho^2)/(1 + 2*b*rho - (b*rho)**2)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")
        self.alpha = kwargs.get("alpha")

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value is None:
            raise ValueError("a value must be provided for using Peng-Robinson EOS")
        self._a = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if value is None:
            raise ValueError("b value must be provided for using Peng-Robinson EOS")
        self._b = value

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value is None:
            raise ValueError("alpha value must be provided for using Peng-Robinson EOS")
        self._alpha = value

    @partial(jit, static_argnums=(0,))
    def EOS(self, rho_tree):
        rho_tree = tree_map(
            lambda rho: self.precision_policy.cast_to_compute(rho), rho_tree
        )
        eos = lambda rho: (rho * self.R * self.T) / (1.0 - self.b * rho) - (
            self.a * self.alpha * rho**2
        ) / (1.0 + 2 * self.b * rho - self.b**2 * rho**2)
        return tree_map(eos, rho_tree)


class Carnahan_Starling(Multiphase):
    """
    Define multiphase model using the Carnahan-Starling EOS.

    Attributes:

    Reference:
        1.  Carnahan, Norman F., and Kenneth E. Starling. "Equation of state for nonattracting rigid spheres."
        The Journal of chemical physics 51, no. 2 (1969): 635-636. https://doi.org/10.1063/1.1672048

    Notes:
        EOS is given by:
            p = (rho*R*T)/(1 - b*rho) - (a*alpha*rho^2)/(1 + 2*b*rho - (b*rho)**2)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if value is None:
            raise ValueError("a value must be provided for using Carnahan-Starling EOS")
        self._a = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if value is None:
            raise ValueError("b value must be provided for using Carnahan-Starling EOS")
        self._b = value

    @partial(jit, static_argnums=(0,))
    def EOS(self, rho_tree):
        rho_tree = tree_map(
            lambda rho: self.precision_policy.cast_to_compute(rho), rho_tree
        )
        eos = (
            lambda rho: rho
            * self.R
            * self.T
            * (
                1.0
                + 0.25 * self.b * rho
                + (0.25 * self.b * rho) ** 2
                - (0.25 * self.b * rho) ** 3
            )
            / (1.0 - 0.25 * self.b * rho) ** 3
            - self.a * rho**2
        )
        return tree_map(eos, rho_tree)
