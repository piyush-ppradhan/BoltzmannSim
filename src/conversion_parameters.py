class ConversionParameters(object):
    """
        Convert a parameter in Lattice Units (phi_star) to SI units (phi) using Reynold's number.
        phi = phi_star * C_phi

        For more details, refer to "The Lattice Boltzmann Method: Principles and Practice" by Krüger et. al
        
        Attributes
        **********
        C_l: float
            Scaling factor for length
        C_t: float
            Scaling factor for time
        C_rho: float
            Scaling factor for density
    """
    def __init__(self):
        pass

    def construct_converter_dx_tau_star(self,l_characteristic,u_charateristic,rho,nu,dx,tau_star,lattice):
        """
            Compute the conversion parameters, given Re,l_charasteristic,u_characteristic,rho,nu,dx,tau_star where:
            
            Arguments
            *********
            l_charasteristic: float
                Charateristic length of the flow {SI units} 
            u_characteristic: float
                Characteristic velocity of the flow {SI units}
            rho: float
                Density of the fluid {SI units}
            nu: float
                Kinematic viscosity of fluid {SI units}
            dx: float
                Lattice spacing {SI units}. Must be choosen to resolve necessary flow features
            tau_star: float
                Non-dimensional time parameter {SI units}. For stability and accuracy 0.5 < tau_star < 1.0
            lattice: Lattice
                Lattice defined using the subclasses of "Lattice" class defined in lattice.py
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
            ValueError("Lattice velocity exceeds the speed of sound for given parameters, choose different parameters")
            exit()
        self.print_conversion_parameters()

    def construct_converter_dt_tau_star(self,l_characteristic,u_charateristic,rho,nu,dt,tau_star,lattice):
        """
            Compute the conversion parameters, given Re,l_charasteristic,u_characteristic,rho,nu,dt,tau_star where:

                Arguments
            *********
            l_charasteristic: float
                Charateristic length of the flow {SI units} 
            u_characteristic: float
                Characteristic velocity of the flow {SI units}
            rho: float
                Density of the fluid {SI units}
            nu: float
                Kinematic viscosity of fluid {SI units}
            dx: float
                Lattice spacing {SI units}. Must be choosen to resolve necessary flow features
            dt: float
                Timestep used for simulation {SI units}. Must be choosen to resolve necessary flow features
            tau_star: float
                Non-dimensional time parameter {SI units}. For stability and accuracy 0.5 < tau_star < 1.0
            lattice: Lattice
                Lattice defined using the subclasses of "Lattice" class defined in lattice.py
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
            ValueError("Lattice velocity exceeds the speed of sound for given parameters, choose different parameters")
            exit()
        self.print_conversion_parameters()

    def construct_converter_dx_dt(self,l_characteristic,u_charateristic,rho,nu,dx,dt,lattice):
        """
            Compute the conversion parameters, given Re,l_charasteristic,u_characteristic,rho,nu,dx,dt where:

                Arguments
            *********
            l_charasteristic: float
                Charateristic length of the flow {SI units} 
            u_characteristic: float
                Characteristic velocity of the flow {SI units}
            rho: float
                Density of the fluid {SI units}
            nu: float
                Kinematic viscosity of fluid {SI units}
            dx: float
                Lattice spacing {SI units}. Must be choosen to resolve necessary flow features
            dt: float
                Timestep used for simulation {SI units}. Must be choosen to resolve necessary flow features
            lattice: Lattice
                Lattice defined using the subclasses of "Lattice" class defined in lattice.py
        """
        self.C_l = dx
        self.C_t = dt
        self.C_rho = rho
        tau_star = (nu * (dt / dx**2) / lattice.c_s2) + (1.0/2.0)
        if tau_star < 0.5:
            ValueError("Tau must be greater than 0.5 for stability, choose different parameters")
            exit()
        elif tau_star > 1.0:
            ValueError("Tau must be less than or equal to 1.0 to ensure accuracyi, choose different parameters")
            exit()
        self.print_conversion_parameters()

    def print_conversion_parameters(self):
        """
            Helper function to print the conversion parameters C_l, C_t, C_rho

            Arguments
            *********
            None
        """
        print("C_l:",self.C_l)
        print("C_t:",self.C_t,"To simulate 1s of real world, {int(1.0/self.C_l)} timesteps are required.")
        print("C_rho:",self.C_rho)


        

        
