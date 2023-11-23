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
class velocity_bc(boundary_condition):
    def __init__(self,id:int,velocity) -> None:
        super().__init__(id)
        # TODO
        # The bc float precision should match the given precision
        self.velocity = velocity

class pressure_bc(boundary_condition):
    def __init__(self,id:int,pressure:float) -> None:
        super().__init__(id)
        # TODO
        # The bc float precision should match the given precision
        self.pressure = pressure

class periodic_bc(boundary_condition):
    def __init__(self,id:int) -> None:
        super().__init__(id)
        match self.id:
            case 0:
                self.periodic_id = 2
            case 1:
                self.periodic_id = 3
            case 2:
                self.periodic_id = 0
            case 3:
                self.periodic_id = 1
            case 4:
                self.periodic_id = 5
            case 5:
                self.periodic_id = 4

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
# Single Phase Problem Definition
class single_phase_problem:
    def __init__(
        self,model_type_:model_type,collision_type:collision_operator,
        boundary_conditions:list,dx:float,dt:float,nu:float,
        max_timesteps:int,write_start:float,write_control:float,force=[0.0,0.0]
    ) -> None:
        self.model_type_ = model_type_   
        self.collision_type = collision_type
        self.boundary_conditions = boundary_conditions #list of boundary_condition
        self.dx = dx
        self.dt = dt
        self.nu = nu
        self.max_timesteps = max_timesteps
        self.write_start = write_start
        self.write_control = write_control
        self.force = force # list of float
