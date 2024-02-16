"""
ConversionParameters class definition to convert between Lattice and physical units
"""

class ConversionParameters(object):
    """
    ParentClass to define conversion parameters that convert values between Lattice Units (phi_star) and physical units (phi) using non-dimensional numbers.
    phi = phi_star * C_phi
    
    In a typical problem, conversion for length, time and density is enough however, additional parameters for scaling can be defined.

    For more details, refer to "The Lattice Boltzmann Method: Principles and Practice" by Krüger et. al
    
    Attributes:
        Conversion parameters: float
            Defined in sub-class.

        attributes: set
            The name of attributes contained sub-classes stored as strings.
    """
    def __init__(self):
        pass

    def to_physical_units(self, x, ttype):
        """
        Convert the given data to physical units. The definition is overwritten in sub-classes.

        Arguments:
            x: numpy.ndarray or jax.ndarray
                The data that needs to converted.
            ttype: str
                The type of x in the context of flow variables. For example, values could be "velocity", "density" etc..

        Returns:
            x: numpy.ndarray or jax.ndarray
                Converted data.
        """
        pass

    def to_lattice_units(self, x, ttype):
        """
        Convert the given data to lattice units. The definition is overwritten in sub-classes.

        Arguments:
            x: float, numpy.ndarray or jax.ndarray
                The data that needs to converted.
            ttype: str
                The type of x in the context of flow variables. For example, values could be "velocity", "density" etc...

        Returns:
            x: float, numpy.ndarray or jax.ndarray
                Converted data.
        
        """
        pass

    def print_conversion_parameters(self): 
        for attribute in self.attributes:
            print(attribute + ": " + getattr(self, attribute) + "\n")
            
class ReConversion(ConversionParameters):
    def __init__(self):
        pass

    def construct_converter_dx_tau_star(self, l_characteristic, u_charateristic, rho, nu, dx, tau_star, lattice):
        """
        Compute the conversion parameters, given Re, l_charasteristic, u_characteristic, rho, nu, dx, tau_star where:
        
        Arguments
            l_charasteristic: float
                Charateristic length of the flow {physical units} 
            u_characteristic: float
                Characteristic velocity of the flow {physical units}
            rho: float
                Density of the fluid {physical units}
            nu: float
                Kinematic viscosity of fluid {physical units}
            dx: float
                Lattice spacing {physical units}. Must be choosen to resolve necessary flow features
            tau_star: float
                Non-dimensional time parameter {physical units}. For stability and accuracy 0.5 < tau_star < 1.0
            lattice: Lattice
                Lattice defined using the subclasses of "Lattice" class defined in lattice.py

        Returns:
            None
        """
        self.C_l = dx
        self.C_rho = rho
        dt = lattice.c_s2 * (tau_star - (1.0/2.0)) * (dx**2 / nu)
        self.C_t = dt
        l_characteristic_star = l_characteristic / dx
        nu_star = lattice.c_s2 * (tau_star - (1.0/2.0))
        Re = u_charateristic * l_characteristic / nu # Re: float, Reynolds number of the flow
        u_star = Re * nu_star / l_characteristic_star
        if u_star > lattice.c_s:
            raise ValueError("Lattice velocity exceeds the speed of sound for given parameters, choose different parameters.")
        self.print_conversion_parameters()

    def construct_converter_dt_tau_star(self, l_characteristic, u_charateristic, rho, nu, dt, tau_star, lattice):
        """
        Compute the conversion parameters, given Re,l_charasteristic,u_characteristic,rho,nu,dt,tau_star where:

        Arguments:
            l_charasteristic: float
                Charateristic length of the flow {physical units} 
            u_characteristic: float
                Characteristic velocity of the flow {physical units}
            rho: float
                Density of the fluid {physical units}
            nu: float
                Kinematic viscosity of fluid {physical units}
            dx: float
                Lattice spacing {physical units}. Must be choosen to resolve necessary flow features
            dt: float
                Timestep used for simulation {physical units}. Must be choosen to resolve necessary flow features
            tau_star: float
                Non-dimensional time parameter {physical units}. For stability and accuracy 0.5 < tau_star < 1.0
            lattice: Lattice
            Lattice defined using the subclasses of "Lattice" class defined in lattice.py
        
        Returns:
            None
        """
        self.C_t = dt
        self.C_rho = rho
        dx = (nu * dt / (tau_star - (1.0/2.0)))**0.5
        self.C_l = dx
        l_characteristic_star = l_characteristic / dx
        nu_star = lattice.c_s2*(tau_star - (1.0/2.0))
        Re = u_charateristic * l_characteristic / nu
        u_star = Re * nu_star / l_characteristic_star
        if u_star > lattice.c_s:
            raise ValueError("Lattice velocity exceeds the speed of sound for given parameters, choose different parameters")
        self.print_conversion_parameters()

    def construct_converter_dx_dt(self, l_characteristic, u_charateristic, rho, nu, dx, dt, lattice):
        """
        Compute the conversion parameters, given Re,l_charasteristic,u_characteristic,rho,nu,dx,dt where:

        Arguments:
            l_charasteristic: float
                Charateristic length of the flow {physical units} 
            u_characteristic: float
                Characteristic velocity of the flow {physical units}
            rho: float
                Density of the fluid {physical units}
            nu: float
                Kinematic viscosity of fluid {physical units}
            dx: float
                Lattice spacing {physical units}. Must be choosen to resolve necessary flow features
            dt: float
                Timestep used for simulation {physical units}. Must be choosen to resolve necessary flow features
            lattice: Lattice
                Lattice defined using the subclasses of "Lattice" class defined in lattice.py

        Returns:
            None
        """
        self.C_l = dx
        self.C_t = dt
        self.C_rho = rho
        tau_star = (nu * (dt / dx**2) / lattice.c_s2) + (1.0/2.0)
        if tau_star < 0.5:
            raise ValueError("Tau must be greater than 0.5 for stability, choose different parameters")
        elif tau_star > 1.0:
            ValueError("Tau must be less than or equal to 1.0 to ensure accuracy, choose different parameters")
        self.print_conversion_parameters()

    def to_physical_units(self, x, ttype):
        match ttype:
            case "velocity":
                return x * (self.C_l / self.C_t)
            case "density":
                return x * self.C_rho
            case "force":
                return x * (self.C_rho * (self.C_l**4) * (self.C_t**0.5))
            case _:
                raise ValueError("Invalid type " + ttype)
            
    def to_lattice_units(self, x, ttype):
        match ttype:
            case "velocity":
                return x * (self.C_t / self.C_l)
            case "density":
                return x / self.C_rho
            case "force":
                return x / (self.C_rho * (self.C_l**4) * (self.C_t**0.5))
            case _:
                raise ValueError("Invalid type " + ttype)


        

        
