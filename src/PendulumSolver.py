"""
Custom Kapitza pendulum system numerical solving code.
© 2025 Neil Ghugare
"""

# todo typing, if time permits

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from tqdm import trange
from typing import override

class KapitzaPendulum:
    """
    Class that numerically solves the equations of motion for the Kapitza pendulum.
    """
    
    def __init__(self, m=1., l=1., g=1., a=1., nu=1.):
        """
        --- Description: ---
        Initializes the conditions of a Kapitza pendulum system.

        --- Parameters (Required): ---
            None

        --- Parameters (Optional): ---
            1. m - The mass of the pendulum bob.
            2. l - The length of the pendulum rod.
            3. g - The gravitational acceleration.
            4. a - The driving amplitude.
            5. nu - The driving frequency. 

        --- Returns: ---
            None
        """
        self.m = m
        self.l = l
        self.g = g
        self.a = a
        self.nu = nu
        self.nu2 = nu**2
        self.omega0 = np.sqrt(g/l)  # omega_0 natural frequency of the pendulum
    
    def dy_dt(self, t, y):
        """
        --- Description: ---
        Calculates the derivative vector [phi_dot, phi_double_dot] for a given y-vector and t.

        --- Parameters (Required): ---
            1. t - The current time.
            2. y - The current position vector [phi, phi_dot]

        --- Parameters (Optional): ---
            None

        --- Returns: ---
            The derivative vector [phi_dot, phi_double_dot], with phi_double_dot coming from the equations of motion.
        """
        return [y[1], -(self.g + self.a*self.nu2*np.cos(self.nu*t))*np.sin(y[0])/self.l]
    
    def solve_ode(self, t_pts, phi_0, phi_dot_0, abserr=1.0e-10, relerr=1.0e-10):
        """
        --- Description: ---
        Solves the equation of motions ODE using Scipy solve_ivp.

        --- Parameters (Required): ---
            1. t_pts - The finite array of time points to solve over.
            2. phi_0 - The initial phi position of the pendulum.
            3. phi_dot_0 - The initial phi velocity of the pendulum.

        --- Parameters (Optional): ---
            1. abserr - The absolute tolerance to use in the solution solver.
            2. relerr - The relative tolerance to use in the solution solver.

        --- Returns: ---
            The full motion over time, phi(t) and phi_dot(t), as two arrays.
        """
        y = [phi_0, phi_dot_0]
        solution = solve_ivp(self.dy_dt, (t_pts[0], t_pts[-1]), y, t_eval=t_pts,
                             atol=abserr, rtol=relerr)
        phi, phi_dot = solution.y
        return phi, phi_dot
    
    def space_plots(self, t_pts, phi_pts, phi_dot_pts, fig_save_path=None):
        """
        --- Description: ---
        Makes state space plots for the current pendulum system.

        --- Parameters (Required): ---
            1. t_pts - The finite array of time points to solve over.
            2. phi_pts - The phi points, phi(t).
            3. phi_dot_pts - The phi dot points, phi_dot(t).

        --- Parameters (Optional): ---
            1. fig_save_path - The path and filename to save the figure.

        --- Returns: ---
            The matplotlib figure object of the image.
        """
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))

        # --- phi(t) plot ---
        ax[0].plot(t_pts, phi_pts, label=r'$\phi$')
        ax[0].set_xlabel(r'$t$')
        ax[0].set_ylabel(r'$\phi(t)$')
        ax[0].tick_params(axis='both', top=True, right=True, which='both', direction='in')
        ax[0].set_yscale('linear')

        # --- phi_dot(t) plot ---
        ax[1].plot(t_pts, phi_dot_pts)
        ax[1].set_xlabel(r'$t$')
        ax[1].set_ylabel(r'$\dot{\phi}(t)$')
        ax[1].tick_params(axis='both', top=True, right=True, which='both', direction='in')

        # --- state space plot ---
        ax[2].plot(phi_pts, phi_dot_pts)
        ax[2].set_xlabel(r'$\phi$')
        ax[2].set_ylabel(r'$\dot{\phi}$')
        ax[2].tick_params(axis='both', top=True, right=True, which='both', direction='in')

        plt.suptitle(r'Motion and State Space for Kapitza Pendulum')
        ax[1].set_title(rf'$\omega_0={self.omega0}, a={self.a}, \nu={self.nu}$')

        plt.tight_layout()

        if fig_save_path:
            plt.savefig(fig_save_path, dpi=300)
        
        return fig

    def phi_and_driving_plot(self, t_pts, phi_pts, fig_save_path=None):
        """
        --- Description: ---
        Makes a plot for the current pendulum system of the motion and driving force versus time.

        --- Parameters (Required): ---
            1. t_pts - The finite array of time points to solve over.
            2. phi_pts - The phi points, phi(t).

        --- Parameters (Optional): ---
            1. fig_save_path - The path and filename to save the figure.

        --- Returns: ---
            The matplotlib figure object of the image.
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))

        # --- phi(t) plot ---
        ax.plot(t_pts, phi_pts, label=r'$\phi$')
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$\phi(t)$')
        ax.tick_params(axis='both', top=True, right=True, which='both', direction='in')
        ax.set_yscale('linear')
        
        # --- driving force plot on 1 ---
        driving_force = np.cos(self.nu*t_pts)
        ax.plot(t_pts, driving_force, label='Driving Force', color='red', alpha=0.5)
        ax.legend()

        plt.suptitle(r'Motion with Driving Force for Kapitza Pendulum')
        ax.set_title(rf'$\omega_0={self.omega0}, a={self.a}, \nu={self.nu}$')

        plt.tight_layout()

        if fig_save_path:
            plt.savefig(fig_save_path, dpi=300)
        
        return fig

    def Liapunov_comparison(self, t_pts, phi_0, phi_dot_0, delta_phi=1.0e-6, delta_phi_dot=0.):
        """
        --- Description: ---
        Makes a comparsion relating to the Liapunov exponent, plotting log(|Delta phi|) vs. t.

        --- Parameters (Required): ---
            1. t_pts - The finite array of time points to solve over.
            2. phi_pts - The phi points, phi(t).
            3. phi_dot_pts - The phi dot points, phi_dot(t).

        --- Parameters (Optional): ---
            1. delta_phi - The (small) displacement between the two systems, for comparison.
            2. delta_phi_dot - The (small) velocity difference between the two systems, for comparison.

        --- Returns: ---
            The matplotlib figure object of the image.
        """

        phi, phi_dot = self.solve_ode(t_pts, phi_0, phi_dot_0)
        phi2, phi_dot2 = self.solve_ode(t_pts, phi_0+delta_phi, phi_dot_0+delta_phi_dot)

        dphi = np.abs(phi2 - phi)

        fig, ax = plt.subplots()

        # --- Delta phi plot ---
        ax.plot(t_pts, dphi, label=r'$\Delta \phi$')
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)  # Horizontal line at 0
        ax.set_yscale('log')  # Log scale for y-axis
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$ \left| \Delta \phi(t) \right|$')
        ax.grid(True)
        ax.set_title(r'Comparison of $\Delta \phi$ for Chaos via Liapunov Exponent')

        fig.tight_layout()
        return fig

    def bifurcation(self, phi_0, phi_dot_0, a_range=(0.0, 2.0, 200), num_t_points=200, rough_modeling=True):
        """
        --- Description: ---
        Creates a Hopf bifurcation plot for a fixed driving frequency of the system.

        --- Parameters (Required): ---
            1. phi_0 - The initial phi position of the pendulum.
            2. phi_dot_0 - The initial phi velocity of the pendulum.

        --- Parameters (Optional): ---
            1. a_range - The tuple for the range of driving amplitudes to try of (a_start, a_end, num_points).
            2. num_t_points - The number of t points (Poincare points) to use.
            3. rough_modeling - If the rough Hopf modeling should be shown in red on the figure.
            
        --- Returns: ---
            The matplotlib figure object of the image.
        """
        # --- Setup ---
        
        # Setup the parameter space
        a_values = np.linspace(*a_range)
        T = 2. * np.pi / self.nu  # Driving period
    
        # Total integration time
        t_transient_periods = 50
        t_steady_state_periods = 100
        t_start = 0.
        t_end = (t_transient_periods + t_steady_state_periods) * T
    
        # Time points for sampling (Poincaré section)
        # Start sampling after the transient period
        t_sample_start = t_transient_periods * T
        t_sample_end = t_end
        
        # Create the time array for the sample points: t = t_sample_start + n*T
        t_pts_sample = np.linspace(t_sample_start, t_sample_end, num_t_points, endpoint=True)
    
        # Full time array to pass to the ODE solver
        t_pts_full = np.linspace(t_start, t_end, 200 * (t_transient_periods + t_steady_state_periods))
    
        all_sampled_phis = []
        corresponding_a_values = []
        model_upper, model_lower = [], []

        # --- Iteration ---
        
        # Iterate over 'a' values
        for i in trange(len(a_values)):
            a = a_values[i]
            # Initialize the system for the current 'a'
            pendulum = KapitzaPendulum(a=a, nu=self.nu, m=self.m, g=self.g, l=self.l)
            
            # Solve the ODE
            # We only care about the time points that correspond to the Poincaré section
            phi_pts_full, _ = pendulum.solve_ode(t_pts_full, phi_0, phi_dot_0)
            
            # Use interpolation to get values at the precise sample times if t_pts_full is used
            solution = solve_ivp(pendulum.dy_dt, (t_start, t_end), [phi_0, phi_dot_0], 
                                 t_eval=t_pts_sample, atol=1.0e-10, rtol=1.0e-10)
            
            sampled_phis = solution.y[0]
            
            # Filter and store the steady-state sampled $\phi$ values
            # Since t_pts_sample was created to start after t_transient_periods, 
            # all points are considered steady-state.
            
            # Normalize the angle to be between -pi and pi for plotting clarity
            normalized_phis = np.arctan2(np.sin(sampled_phis), np.cos(sampled_phis))

            # Rough Hopf bifurcation modeling
            if rough_modeling:
                # We used circular standard deviations to check, using cosines and sines.
                # See https://en.wikipedia.org/wiki/Directional_statistics.
                cos_sum = np.sum(np.cos(normalized_phis))
                sin_sum = np.sum(np.sin(normalized_phis))
                N = len(normalized_phis)
            
                # Calculate the Mean Resultant Vector Length (R)
                # R is the length of the average vector.
                R_bar = np.sqrt(cos_sum**2 + sin_sum**2) / N
            
                # Calculate the Circular Variance (V)
                # V = 1 - R_bar
                V_circ = 1 - R_bar

                # If V_circ is close to 0, it implies that the Poincaré points are concentrated
                # at the ends where phi_0 is. Otherwise it implies some chaos with Poincaré points
                # spread throughout.
                if V_circ <= 0.1:  # Small V_circ implies concentration at the ends, no spread.
                    model_upper.append(np.mean(normalized_phis[np.where(normalized_phis > 0.0)]))
                    model_lower.append(np.mean(normalized_phis[np.where(normalized_phis < 0.0)]))
                else:
                    mean = np.mean(normalized_phis)
                    model_upper.append(mean)
                    model_lower.append(mean)
    
            # Store the points for plotting
            all_sampled_phis.extend(normalized_phis)
            corresponding_a_values.extend([a] * len(normalized_phis))

        # --- Plotting ---
        
        # Plot the bifurcation diagram
        fig, ax = plt.subplots(figsize=(10, 6))
        if a_range[2] <= 300 and num_t_points <= 300:
            ax.plot(corresponding_a_values, all_sampled_phis, 'k.', markersize=2, alpha=0.5)
        else:  # Use smaller points
            ax.plot(corresponding_a_values, all_sampled_phis, ',k', alpha=0.5)
        if not rough_modeling:  # Even rougher modeling of the phi_0 and -phi_0 locations
            ax.axhline(phi_0, color='red', linestyle='--')
            ax.axhline(-phi_0, color='red', linestyle='--')
        else:  # Else plot the rough Hopf modeling
            ax.plot(a_values, model_upper, color='red', linestyle='-', linewidth=1)
            ax.plot(a_values, model_lower, color='red', linestyle='-', linewidth=1)
        
        ax.set_xlabel(r'Driving Amplitude $a$')
        ax.set_ylabel(r'Angle $\phi$ at $t=nT$ (mod $2\pi$)')
        ax.set_title(rf'Bifurcation Diagram for Kapitza Pendulum ($\nu={self.nu}$)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', top=True, right=True, which='both', direction='in')
        
        plt.tight_layout()
        return fig

    def U_snapshots(self, t_min=0.0, t_max=45.0):
        """
        --- Description: ---
        Creates thumbnail plots of the evolution of the time-dependent U as a function of phi.
        This function always splits into 10 thumbnails, including the endpoints.

        --- Parameters (Required): ---
            None

        --- Parameters (Optional): ---
            1. t_min - The starting time.
            2. t_max - The ending time.
            
        --- Returns: ---
            The matplotlib figure object of the image.
        """
        phi_pts = np.arange(0.0, 2*np.pi + 1e-3, 1e-3)  # Zero to 2pi phi values.

        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(10, 6))

        # --- Snapshots ---

        t_pts = np.linspace(t_min, t_max, 10)
        row = 0
        min_U, max_U = np.inf, -np.inf 
        for _ in range(10):
            U_vals = -self.m*self.g*(self.l*np.cos(phi_pts) + self.a*np.cos(self.nu*t_pts[_]))

            ax[row][_%5].plot(phi_pts, U_vals)
            ax[row][_%5].set_xlabel(r'$\phi$')
            ax[row][_%5].set_ylabel(rf'$U(\phi, t={t_pts[_]:.2f})$')
            ax[row][_%5].tick_params(axis='both', top=True, right=True, which='both', direction='in')
            
            if _ == 4:
                row = 1

            if np.max(U_vals) > max_U:
                max_U = np.max(U_vals)
            if np.min(U_vals) < min_U:
                min_U = np.min(U_vals)

        # --- Fixing Limits ---
        
        deltaU = np.abs((max_U-min_U)/50.0)
        for i in range(2):
            for j in range(5):
                ax[i][j].set_ylim([min_U-deltaU, max_U+deltaU])  # Static axes for better comparison
                ax[i][j].set_xlim([0.0, 2*np.pi])                # Fixed xlim
        

        fig.suptitle(r'$U(\phi, t)$ Snapshots')

        plt.tight_layout()
        return fig

    def U_eff_plot(self, phi_pts):
        """
        --- Description: ---
        Plots U_eff as a function of phi, assuming a slow-fast regime.

        --- Parameters (Required): ---
            1. phi_pts - The phi domain to plot over.

        --- Parameters (Optional): ---
            None
            
        --- Returns: ---
            The matplotlib figure object of the image.
        """
        fig, ax = plt.subplots()

        # U_eff from our known formula
        U_eff = -self.m*self.g*self.l*np.cos(phi_pts) + ((self.m*self.a*self.nu**2)/(4*self.l))*(np.sin(phi_pts))**2
        
        ax.plot(phi_pts, U_eff, label=r'$\Delta \phi$')
        ax.set_xlabel(r'$\phi$')
        ax.set_ylabel(r'$U_{\rm eff}$')
        ax.set_xlim([np.min(phi_pts), np.max(phi_pts)])
        ax.set_title(r'$U_{\rm eff}(\phi)$ For The Given System in Slow-Fast Regime')

        fig.tight_layout()
        return fig
        

class DampedKapitzaPendulum(KapitzaPendulum):
    """
    Class that numerically solves the equations of motion for a damped Kapitza pendulum.
    """
    
    def __init__(self, m=1., l=1., g=1., a=1., nu=1., gamma=0.02):
        """
        --- Description: ---
        Initializes the conditions of a damped Kapitza pendulum system.

        --- Parameters (Required): ---
            None

        --- Parameters (Optional): ---
            1. m - The mass of the pendulum bob.
            2. l - The length of the pendulum rod.
            3. g - The gravitational acceleration.
            4. a - The driving amplitude.
            5. nu - The driving frequency. 
            6. gamma - The damping parameter

        --- Returns: ---
            None
        """
        self.gamma = gamma
        super().__init__(m, l, g, a, nu)

    @override
    def dy_dt(self, t, y):
        """
        --- Description: ---
        Calculates the derivative vector [phi_dot, phi_double_dot] for a given y-vector and t.
        Overrides the super-class dy_dt.

        --- Parameters (Required): ---
            1. t - The current time.
            2. y - The current position vector [phi, phi_dot]

        --- Parameters (Optional): ---
            None

        --- Returns: ---
            The derivative vector [phi_dot, phi_double_dot], with phi_double_dot coming from the equations of motion.
        """
        return [y[1], -(self.g + self.a*self.nu2*np.cos(self.nu*t))*np.sin(y[0])/self.l - self.gamma*y[1]]       
