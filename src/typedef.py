class model_type:
    def __init__(self,d,q) -> None:
        self.d = d
        self.q = q

    def print_model_type(self):
        print("D{self.d}Q{self.q} model\n")

class d2q9(model_type):
    def __init__(self) -> None:
        super().__init__(2,9)

    def print_model_type(self) -> None:
        super().print_model_type()

class d3q15(model_type):
    def __init__(self) -> None:
        super().__init__(3,15)

    def print_model_type(self) -> None:
        super().print_model_type()

class d3q19(model_type):
    def __init__(self) -> None:
        super().__init__(3,19)

    def print_model_type(self) -> None:
        super().print_model_type()

class d3q27(model_type):
    def __init__(self) -> None:
        super().__init__(3,27)

    def print_model_type(self) -> None:
        super().print_model_type()

# --------------------------#
# Boundary Condition
"""
Boundary Conditions are defined in a cyclic manner starting from the inlet on the left going in a anti-clockwise manner. For 3D case, the ordering is defined by backface followed by front face
Each boundary thus has an associated id to determine the face/edge where the boundary condition is applied

For 2D, the id's are:
        3
    __ __ __ __ 
    |         |
0   |         | 2
    |         |
    __ __ __ __
        1

For 3D, the id's are:
        3
    __ __ __ __ 
    |      4   |
0   |     /    | 2
    |    5     |
    __ __ __ __
        1

The front face has id 5 and the back face has id 4
"""
class boundary_condition:
    def __init__(self,id:int) -> None:
        self.id = id

# General condition for defining velocity
class zou_he_velocity_bc(boundary_condition):
    def __init__(self,id:int,velocity) -> None:
        super().__init__(id)
        # TODO
        # The bc float precision should match the given precision
        self.velocity = velocity

class zou_he_pressure_bc(boundary_condition):
    def __init__(self,id:int,pressure:float) -> None:
        super().__init__(id)
        # TODO
        # The bc float precision should match the given precision
        self.pressure = pressure

class guo_pressure_bc(boundary_condition):
    def __init__(self,id:int,pressure:float) -> None:
        super().__init__(id)
        # TODO
        # The bc float precision should match the given precision
        self.pressure = pressure

class periodic_bc_x(boundary_condition):
    def __init__(self,id:int) -> None:
        super().__init__(id)

class periodic_bc_y(boundary_condition):
    def __init__(self,id:int) -> None:
        super().__init__(id)

class periodic_bc_z(boundary_condition):
    def __init__(self,id:int) -> None:
        super().__init__(id)

class no_slip_wall_bc(boundary_condition):
    def __init__(self,id:int) -> None:
        super().__init__(id)

class slip_wall_bc(boundary_condition):
    def __init__(self,id:int,wall_velocity=[0.0,0.0]) -> None:
        super().__init__(id)
        # TODO
        # The bc float precision should match the given precision
        self.slip_wall_vel = wall_velocity

# --------------------------#
# Collision Operator
class collision_operator:
    def __init__(self,c_type:int) -> None:
        self.c_type = c_type

class bgk(collision_operator):
    def __init__(self,c_type:int=0) -> None:
        super().__init__(c_type)

#TODO
class mrt(collision_operator):
    def __init__(self,c_type:int=1) -> None:
        super().__init__(c_type)

# --------------------------#
# Body Force

class body_force:
    def __init__(self,force) -> None:
        self.force = force

class guo_body_force(body_force):
    def __init__(self, force) -> None:
        super().__init__(force)

# Unit conversion from SI to Lattice units and vice versa
# delta x = 1, delta t = 1 in lattice units
class ScaleParameters:
    def __init__(self,dx,dt,physical_density) -> None:
        self.dt = dt
        self.C_rho = physical_density
        self.C_u = dx/dt
        
def unit_converter_resolution_relax_parameter(dx,tau_star,physical_viscosity,physical_density,characteristic_physical_velocity,characteristic_length):
    nu_star = (1/3) * (tau_star - 1/2) # Non-dimensional viscosity
    lc_star = characteristic_length / dx # Non-dimensional characteristic length
    Re = characteristic_physical_velocity * characteristic_length / physical_viscosity
    u_lattice = Re * nu_star / lc_star
    if u_lattice > 0.577:
        print("The expected lattice velocity exceeds the speed of sound. Please change the parameters")
        exit(1)
    else:
        dt = (1/3) * (tau_star - 0.5) * dx * dx / physical_viscosity
        return ScaleParameters(dx,dt,physical_density)

def unit_converter_resolution_lattice_velocity(dx,lattice_velocity,physical_viscosity,physical_density,characteristic_physical_velocity,characteristic_length):
    if lattice_velocity > 0.577:
        print("The expected lattice velocity exceeds the speed of sound. Please change the parameters")
        exit(1)
    else:
        lc_star = characteristic_length / dx # Non-dimensional characteristic length
        Re = characteristic_physical_velocity * characteristic_length / physical_viscosity
        nu_star = lattice_velocity * lc_star / Re
        tau_star = 3*nu_star + 1/2
        dt = (1/3) * (tau_star - 0.5) * dx * dx / physical_viscosity
        return ScaleParameters(dx,dt,physical_density)

def unit_converter_lattice_velocity_relax_parameter(lattice_velocity,tau_star,physical_viscosity,physical_density,characteristic_physical_velocity,characteristic_length):
    if lattice_velocity > 0.577:
        print("The expected lattice velocity exceeds the speed of sound. Please change the parameters")
        exit(1)
    else:
        nu_star = (1/3) * (tau_star - 1/2) # Non-dimensional viscosity
        Re = characteristic_physical_velocity * characteristic_length / physical_viscosity
        lc_star =  Re * nu_star / lattice_velocity
        dx = characteristic_length / lc_star
        dt = (1/3) * (tau_star - 0.5) * dx * dx / physical_viscosity
        return ScaleParameters(dx,dt,physical_density)

# --------------------------#
# Single Phase Problem Definition
# The combination of dx, dt and density (rho) serve as independent unit for non-dimensionalization
# The lattice density is scaled using the value provided in BC
class single_phase_problem:
    def __init__(
        self,model_type_:model_type,collision_type:collision_operator,
        boundary_conditions:list,tau_star:float,scale_parameters:ScaleParameters,
        total_time:float,write_start:float,write_control:float,force=[0.0,0.0]
    ) -> None:
        self.model_type_ = model_type_   
        self.collision_type = collision_type
        self.boundary_conditions = boundary_conditions #list of boundary_condition
        self.tau_star = tau_star
        self.scale_parameters = scale_parameters
        self.total_time = total_time
        self.write_start = write_start
        self.write_control = write_control
        self.force = force # list of float

