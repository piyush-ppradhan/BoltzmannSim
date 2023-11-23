import jax.numpy as jnp

from typedef import model_type, d2q9, d3q15, d3q19, d3q27, \
                    boundary_condition, velocity_bc, pressure_bc,periodic_bc, no_slip_wall_bc,no_slip_wall_bc, slip_wall_bc, \
                    collision_operator, bgk, mrt, \
                    single_phase_problem

def simulate(sp_problem:single_phase_problem):
    
