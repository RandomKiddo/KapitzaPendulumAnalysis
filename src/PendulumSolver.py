import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

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
        ax[0].plot(t_pts, phi_pts)
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
        ax.set_ylabel(r'$\log{\left( \left| \Delta \phi(t) \right| \right)}$')
        ax.grid(True)
        ax.set_title(r'Comparison of $\Delta \phi$ for Chaos via Liapunov Exponent')

        return fig

    def bifurcation(self):
        pass
