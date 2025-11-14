"""
Custom Manim animation example systems code for a Kapitza pendulum.
© 2025 Neil Ghugare
"""

from ManimAnimation import KapitzaPendulumAnimation, KapitzaPendulumChaosAnimation, PendulumParams
from typing import override

# --- Example Kapitza Pendulum Animation Subclasses ---

class ExampleSimplePendulum(KapitzaPendulumAnimation):
    """
    A Manim Scene animation tha animates a Kapitza pendulum's dynamics given set parameters.
    This is a subclass that is an example pendulum. This system emulates a simple pendulum approximation
    of the Kapitza system.

    The following parameters are overridden in this subclass (the rest are the same as the defaults):
    a = 6.85
    nu = 4.0
    phi_0 = pi/2
    """
    
    @override
    def construct(self):
        """
        --- Description: ---
            Constructs the Kapitza pendulum animation scene for this example pendulum.

        --- Parameters (Required): ---
            None

        --- Parameters (Optional): ---
            None

        --- Returns: ---
            None
        """
        # Override the parameters
        KapitzaPendulumAnimation.params = PendulumParams(a=6.85, nu=4., phi_0=np.pi/2, phi_dot_0=0.0)

        # Construct the animation as per the superclass
        super().construct()

class ExampleStablePiEquilibrium(KapitzaPendulumAnimation):
    """
    A Manim Scene animation tha animates a Kapitza pendulum's dynamics given set parameters.
    This is a subclass that is an example pendulum. This system emulates a system where the phi=pi
    equilibrium position is stable due to the high driving frequency.

    The following parameters are overridden in this subclass (the rest are the same as the defaults):
    a = 0.1
    nu = 30.0
    phi_0 = pi/2 - 0.1
    """
    
    @override
    def construct(self):
        """
        --- Description: ---
            Constructs the Kapitza pendulum animation scene for this example pendulum.

        --- Parameters (Required): ---
            None

        --- Parameters (Optional): ---
            None

        --- Returns: ---
            None
        """
        # Override the parameters
        KapitzaPendulumAnimation.params = PendulumParams(a=0.1, nu=30.0, phi_0=np.pi-0.1, phi_dot_0=0.0)

        # Construct the animation as per the superclass
        super().construct()

# --- Example Kapitza Pendulum Chaos Animation Subclasses ---

class ExampleNonChaosSystem(KapitzaPendulumChaosAnimation):
    """
    A Manim Scene animation that animates a Kapitza pendulum's dynamics given set parameters, specifically for chaos comparison.
    This is a subclass that is an example pendulum. This system emulates a system where the the Kapitza system is particularly
    non-chaotic.

    The following parameters are overridden in this subclass (the rest are the same as the defaults):
    a = 0.1
    nu = 2.5
    phi_0 = 0.86
    """
    
    @override
    def construct(self):
        """
        --- Description: ---
            Constructs the Kapitza pendulum chaos animation scene for this example pendulum.

        --- Parameters (Required): ---
            None

        --- Parameters (Optional): ---
            None

        --- Returns: ---
            None
        """
        # Override the parameters
        KapitzaPendulumChaosAnimation.params = PendulumParams(a=0.1, nu=2.5, phi_0=0.86, phi_dot_0=0.0)

        # Construct the animation as per the superclass
        super().construct()

