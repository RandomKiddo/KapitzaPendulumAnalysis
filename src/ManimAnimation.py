"""
Custom Manim animation code for a Kapitza pendulum.
"""

import numpy as np

from scipy.integrate import solve_ivp
from dataclasses import dataclass
from manim import *
from PendulumSolver import KapitzaPendulum as ManimKapitzaPendulum


class ScaleneTriangle(Polygon):
    """
    A custom MObject class that draws a scalene triangle.
    It takes three specific points designed to ensure unequal side lengths.
    """

    def __init__(self, a, b, c, color=GRAY, fill_opacity=0.9, **kwargs):
        """
        --- Description: ---
            Initializes a scalene triangle.

        --- Parameters (Required): ---
            1. a - The coordinate point of the first vertex.
            2. b - The coordinate point of the second vertex.
            3. c - The coordinate point of the third vertex.

        --- Parameters (Optional): ---
            1. color - The color and fill color of the triangle.
            2. fill_opacity - The fill opacity of the triangle.
            3. **kwargs - Other Polygon class keyword args.

        --- Returns: ---
            None
        """

        super().__init__(np.array(a), np.array(b), np.array(c),
                        color=color, fill_color=color, fill_opacity=fill_opacity,
                        **kwargs)


@dataclass(frozen=True)  # frozen implies immutability after construction
class PendulumParams:
    """
    Stores parameters used for the dictionary
    """

    # --- Physical Parameters ---
    m: float = 1.0           # pendulum bob mass
    l: float = 1.0           # pendulum rod length
    g: float = 1.0           # gravitational acceleration

    # --- Driving Parameters ---
    a: float = 1.0           # driving amplitude
    nu: float = 1.0          # driving frequency

    # --- Initial Conditions ---
    phi_0: float = np.pi/2   # initial phi position
    phi_dot_0: float = 0.0           # initial phi velocity

    # --- Time Constraint ---
    t_max: float = 30.0             # maximum time

    # --- Chaos Comparison Terms (Ignore For Regular Animation) ---
    delta_phi = 1.0e-6
    delta_phi_dot = 0.0


class KapitzaPendulumAnimation(Scene):
    """
    A Manim Scene animation tha animates a Kapitza pendulum's dynamics given set parameters.
    """
    
    """
    The stored parameters dictionary used for the animation.
    """
    params: PendulumParams = PendulumParams()
    
    def construct(self):
        """
        --- Description: ---
            Constructs the Kapitza pendulum animation scene.

        --- Parameters (Required): ---
            None

        --- Parameters (Optional): ---
            None

        --- Returns: ---
            None
        """
        
        # Physical parameters
        # Utilize safe parameter fetches to prevent any errors
        m, l, g, a, nu = [KapitzaPendulumAnimation.params.m, 
                          KapitzaPendulumAnimation.params.l, 
                          KapitzaPendulumAnimation.params.g, 
                          KapitzaPendulumAnimation.params.a, 
                          KapitzaPendulumAnimation.params.nu]
        phi_0, phi_dot_0 = KapitzaPendulumAnimation.params.phi_0, KapitzaPendulumAnimation.params.phi_dot_0

        # Simulation time domain
        t_min, t_max = 0., KapitzaPendulumAnimation.params.t_max
        fps = self.camera.frame_rate
        n_frames = int(fps * (t_max-t_min))  # n_frames based on quality / fps
        t_pts = np.linspace(t_min, t_max, n_frames)

        # ODE solver from earlier
        kp = ManimKapitzaPendulum(m=m, l=l, g=g, a=a, nu=nu)
        phi, phi_dot = kp.solve_ode(t_pts, phi_0, phi_dot_0)

        # Normalize some of the data for better plotting
        phi_norm, phi_dot_norm = phi/np.max(np.abs(phi)), phi_dot/np.max(np.abs(phi_dot))

        # --- State space ---

        # Define the ranges of the state space plot (normalized)
        x_range = [-1., 1.]
        y_range = [-1., 1.]
        if np.min(phi_norm) > 0:
            x_range = [np.min(phi_norm), 1.]
        elif np.max(phi_norm) < 0:
            x_range = [-1., np.max(phi_norm)]
        
        # Tick markings on the plot
        x_tick_range = (x_range[1]-x_range[0])/4
        y_tick_range = (y_range[1]-y_range[0])/4
        x_range.append(x_tick_range)
        y_range.append(y_tick_range)

        # State space plotting axes and labels
        # todo sizing of text and ticks
        ax = Axes(x_range=x_range,
                  y_range=y_range,
                  tips=False,
                  x_length=5,
                  y_length=4,
                  axis_config={"include_numbers": True, "decimal_number_config": {"num_decimal_places": 2},},
                 ).to_edge(RIGHT, buff=0.5)
        ax_labels = ax.get_axis_labels(MathTex(r'\phi', color=WHITE), 
                                       MathTex(r'\dot{\phi}', color=WHITE)
                                       )
        ax.x_axis.numbers.set_color(WHITE)
        ax.y_axis.numbers.set_color(WHITE)
        
        # The dot on the state space plot and a trace of its path
        dot = Dot(color=BLUE).move_to(ax.c2p(phi_norm[0], phi_dot_norm[0]))
        trace = TracedPath(dot.get_center, stroke_color=BLUE)

        # Add to the scene (axes, labels, trace, and dot)
        self.add(ax, ax_labels, trace, dot)

        # --- Pendulum animation ---

        # Scale factor of the scene objects and dot radii
        scale_factor = 2
        dot_radius = 0.12

        # The pivot and its base point
        pivot = Dot(color=ORANGE, radius=dot_radius).to_edge(LEFT, buff=6)
        pivot_base = pivot.get_center()

        # Make pivot move up and down as Delta a = -a*cos(nu*t) with a scale factor
        pivot_scale_factor = 0.3  # todo check
        pivot.add_updater(lambda mob: mob.move_to(
            pivot_base + pivot_scale_factor*UP * (-a * np.cos(nu * time_tracker.get_value()))
        ))

        # Real-time time tracker
        time_tracker = ValueTracker(0)

        # Helper, get array index for given time
        def idx():
            """
            --- Description: ---
            Index helper function that get's the current time index for use with simulated data.

            --- Parameters (Required): ---
                None
    
            --- Parameters (Optional): ---
                None
    
            --- Returns: ---
                The current index to use for fetching phi and phi_dot values.
            """
            i = int(time_tracker.get_value() * (n_frames - 1) / (t_max - t_min))
            return np.clip(i, 0, n_frames - 1)

        # Move state space dot
        def update_dot(mob):
            """
            --- Description: ---
            Updates the state space dot so it moves around the grid.

            --- Parameters (Required): ---
                1. mob - The MObject to move (in this case the dot).
    
            --- Parameters (Optional): ---
                None
    
            --- Returns: ---
                None
            """
            i = idx()
            mob.move_to(ax.c2p(phi_norm[i], phi_dot_norm[i]))
        dot.add_updater(update_dot)

        # Rod and bob of the pendulum
        rod = always_redraw(lambda: Line(
            start=pivot.get_center(),
            end=pivot.get_center() + np.array([
                l * np.sin(phi[idx()]),
                -l * np.cos(phi[idx()]),
                0
            ]),
            color=ORANGE
        ))
        bob = always_redraw(lambda: Dot(point=rod.get_end(), radius=dot_radius*1.5, color=YELLOW))

        # Driving pivot point (left end)
        lever_pivot = Triangle(color=GRAY, fill_color=GRAY, fill_opacity=1.).scale(0.5*scale_factor).to_edge(LEFT, buff=1.5)
        center_to_top_diff = lever_pivot.get_top()[1] - lever_pivot.get_center()[1]
        lever_pivot.move_to([lever_pivot.get_center()[0], -center_to_top_diff, 0])

        # Convert Manim's equilateral triangle to our custom ScaleneTriangle
        lever_pivot_vertices = lever_pivot.get_vertices()
        lever_pivot_vertices_range = np.abs(lever_pivot_vertices[1][0]) - np.abs(lever_pivot_vertices[2][0])
        lever_pivot_scale_factor = lever_pivot_vertices_range / 4
        lever_pivot_vertices[1][0] += np.abs(lever_pivot_scale_factor)
        lever_pivot_vertices[2][0] -= np.abs(lever_pivot_scale_factor)
        lever_pivot_sca = ScaleneTriangle(a=lever_pivot_vertices[0], b=lever_pivot_vertices[1], c=lever_pivot_vertices[2],
                                         color=GRAY, fill_opacity=0.9) 

        # Lever rod with the pivot attenuation
        lever_rod = always_redraw(lambda: Line(
            start=pivot_base + pivot_scale_factor*UP * (-a * np.cos(nu * time_tracker.get_value())),
            end=lever_pivot_sca.get_top(),
            color=ORANGE
        ))

        # Driving mechanism wheel
        # todo fix size and fix attenuation height
        wheel_rad = 0.4
        wheel = Circle(color=GRAY, fill_color=GRAY, fill_opacity=1., radius=wheel_rad).to_edge(LEFT, buff=4)
        wheel.move_to([lever_rod.get_center()[0], 0, 0])
        lever_rod_attenuator = always_redraw(lambda: Line(
            start=lever_rod.get_center(),
            end=wheel.get_center() + np.array([wheel_rad/2.5 * np.sin(nu * time_tracker.get_value()), 
                                               wheel_rad/2.5 * np.cos(nu * time_tracker.get_value()), 0]),
            color=ORANGE
        ))

        # Driving mechanism ground attacher
        wheel_base = Triangle(color=GRAY, fill_color=GRAY, fill_opacity=1.).scale(0.3*scale_factor)
        wheel_base_height = np.abs(lever_pivot.get_bottom()[1] - wheel.get_center()[1])
        wheel_base.height = wheel_base_height
        center_to_top_diff = wheel_base.get_top()[1] - wheel_base.get_center()[1]
        wheel_base.move_to([wheel.get_center()[0], wheel.get_center()[1] - center_to_top_diff, 0])

        # Convert ground attacher to scalene triangle
        wheel_base_vertices = wheel_base.get_vertices()
        wheel_base_vertices_range = np.abs(wheel_base_vertices[1][0]) - np.abs(wheel_base_vertices[2][0])
        wheel_range_scale_factor = wheel_base_vertices_range / 4
        wheel_base_vertices[1][0] += np.abs(wheel_range_scale_factor)
        wheel_base_vertices[2][0] -= np.abs(wheel_range_scale_factor)
        wheel_base_sca = ScaleneTriangle(a=wheel_base_vertices[0], b=wheel_base_vertices[1], c=wheel_base_vertices[2],
                                         color=GRAY, fill_opacity=0.9)

        # Rough vector arrow representing the movement of the pendulum
        # Related to the state space plot. Values are normalized
        arrow_scale_factor = 1  # This can be increased to extend the arrow's length if it is always small, but must be hard-coded
        movement_arrow = Arrow(
            start=rod.get_end(),
            end=rod.get_end() + np.array([
                arrow_scale_factor * phi_dot_norm[0] * np.cos(phi[0]),
                arrow_scale_factor * phi_dot_norm[0] * np.sin(phi[0]),
                0.
            ]),
            color=RED,
            stroke_width=8,
            buff=0,  # no buffer so the tail is at the center of the bob
            tip_length=0.2
        )
        movement_arrow.add_updater(lambda m: m.become(Arrow(
            start=rod.get_end(),
            end=rod.get_end() + np.array([
                arrow_scale_factor * phi_dot_norm[idx()] * np.cos(phi[idx()]),
                arrow_scale_factor * phi_dot_norm[idx()] * np.sin(phi[idx()]),
                0.
            ]),
            color=RED,
            stroke_width=8,
            buff=0,  # no buffer so the tail is at the enter of the bob
            tip_length=0.2
        )))

        # Vector group the entire set and add it
        group = VGroup(lever_pivot_sca, wheel_base_sca, wheel, lever_rod, pivot, rod, movement_arrow, bob, lever_rod_attenuator)
        self.add(group)

        # Overall title 
        omega0, epsilon, f = np.sqrt(g/l), a/l, nu/np.sqrt(g/l)
        title = MathTex(r'\text{Kapitza Pendulum: }\omega_0 = %.2f,\ \epsilon = %.2f,\ f = %.2f,\ \phi_0 = %.2f,\ \dot{\phi}_0 = %.2f'
                        % (omega0, epsilon, f, phi_0, phi_dot_0),
                        color=WHITE
                        )
        title.scale(0.8)
        title.to_edge(UP)
        self.add(title)

        # Normalization note of state space below axes
        text = MathTex(r'\text{(Normalized, }\phi\text{ may not be centered at 0.)}', color=WHITE)
        text.scale(0.5)
        text_loc = (ax.x_axis.get_start() +ax.x_axis.get_end())/2
        text_loc[1] = ax.y_axis.get_start()[1]
        text.next_to(text_loc, DOWN, buff=0.4)
        self.add(text)

        # todo add actual min/max of phi/phidot
        text2 = MathTex(r'(\min \phi = %.2f, \max \phi = %.2f, \min \dot{\phi} = %.2f, \max \dot{\phi} = %.2f)' % (
            np.min(phi), np.max(phi), np.min(phi_dot), np.max(phi_dot)
        ))
        text2.scale(0.4)
        text2.next_to(text, DOWN, buff=0.1)
        self.add(text2)

        # Time tracker text
        time_label = MathTex(r't = ')
        time_value = DecimalNumber(0, num_decimal_places=2, include_sign=False, unit=r'\text{ s}')  # Using DecimalNumber for better efficiency
        time_group = VGroup(time_label, time_value).arrange(RIGHT, buff=0.1).scale(0.7)
        time_group.next_to(title, DOWN, aligned_edge=LEFT, buff=0.1)
        time_value.add_updater(lambda m: m.set_value(time_tracker.get_value()))  # Update the current time in an efficient manner
        self.add(time_group)

        # Run real-time animation
        self.play(time_tracker.animate.set_value(t_max),
                  run_time=t_max,
                  rate_func=linear)  # <-- Linear is very important to make the sim time work in real-time

        # Cleanup
        dot.remove_updater(update_dot)
        self.wait()  # Adds an extra second to the animation


class KapitzaPendulumChaosAnimation(Scene):
    """
    A Manim Scene animation that animates a Kapitza pendulum's dynamics given set parameters, specifically for chaos comparison.
    """
    
    """
    The stored parameters dictionary used for the animation.
    """
    params: PendulumParams = PendulumParams()
    
    def construct(self):
        """
        --- Description: ---
            Constructs the Kapitza pendulum animation scene.

        --- Parameters (Required): ---
            None

        --- Parameters (Optional): ---
            None

        --- Returns: ---
            None
        """
        
        # Physical parameters
        # Utilize safe parameter fetches to prevent any errors
        m, l, g, a, nu = [KapitzaPendulumChaosAnimation.params.m, 
                          KapitzaPendulumChaosAnimation.params.l, 
                          KapitzaPendulumChaosAnimation.params.g, 
                          KapitzaPendulumChaosAnimation.params.a, 
                          KapitzaPendulumChaosAnimation.params.nu]
        phi_0, phi_dot_0 = KapitzaPendulumChaosAnimation.params.phi_0, KapitzaPendulumChaosAnimation.params.phi_dot_0
        delta_phi, delta_phi_dot = KapitzaPendulumChaosAnimation.params.delta_phi, KapitzaPendulumChaosAnimation.params.delta_phi_dot

        # Simulation time domain
        t_min, t_max = 0., KapitzaPendulumChaosAnimation.params.t_max
        fps = self.camera.frame_rate
        n_frames = int(fps * (t_max-t_min))  # n_frames based on quality / fps
        t_pts = np.linspace(t_min, t_max, n_frames)

        # ODE solver from earlier
        kp = ManimKapitzaPendulum(m=m, l=l, g=g, a=a, nu=nu)
        phi, phi_dot = kp.solve_ode(t_pts, phi_0, phi_dot_0)

        # ODE solve the second system
        phi2, phi_dot2 = kp.solve_ode(t_pts, phi_0+delta_phi, phi_dot_0+delta_phi_dot)

        # --- Liapunov space ---
        dphi = np.abs(phi2-phi)
        log_dphi = np.log(dphi)

        # Adapted from StackOverflow: https://stackoverflow.com/questions/64183806/extracting-the-exponent-from-scientific-notation.
        def find_exp(number) -> int:
            """
            --- Description: ---
            Finds the exponent value of an exponential number.
            I.e., 5e10 returns 10.

            --- Parameters (Required): ---
                1. number - The number to find the exponent power.
    
            --- Parameters (Optional): ---
                None
    
            --- Returns: ---
                The exponent power.

            --- Note: ---
            Adapted from StackOverflow: https://stackoverflow.com/questions/64183806/extracting-the-exponent-from-scientific-notation.
            """

            # Find the exponential value
            base10 = np.log10(abs(number))
            return int(np.floor(base10))

        x_range = [0., t_max]
        y_range = [np.floor(np.min(log_dphi)), np.ceil(np.max(log_dphi))]
        if y_range[0] > y_range[1]:  # A quick check on the bounds and flip them if needed
            y_range[1], y_range[0] = y_range[0], y_range[1]

        # Tick markings on the plot
        x_tick_range = 5
        y_tick_range = 1
        x_range.append(x_tick_range)
        y_range.append(y_tick_range)

        ax = Axes(x_range=x_range,
                  y_range=y_range,
                  tips=False,
                  x_length=5,
                  y_length=4,
                  axis_config={"include_numbers": True, "decimal_number_config": {"num_decimal_places": 0},},
                  x_axis_config={"font_size": 20},
                  y_axis_config={"font_size": 20},
                 ).to_edge(RIGHT, buff=0.5)
        ax_labels = ax.get_axis_labels(MathTex(r't', color=WHITE, font_size=30), 
                                       MathTex(r'\log \left|\Delta\phi\right|', color=WHITE, font_size=30))
        
        # The dot on the state space plot and a trace of its path
        dot = Dot(color=BLUE).move_to(ax.c2p(t_pts[0], log_dphi[0]))
        trace = TracedPath(dot.get_center, stroke_color=BLUE)

        # Add to the scene (axes, labels, trace, and dot)
        self.add(ax, ax_labels, trace, dot)

        # Real-time time tracker
        time_tracker = ValueTracker(0)

        # Scale factor of the scene objects and dot radii
        scale_factor = 2
        dot_radius = 0.12

        # The pivot and its base point
        pivot = Dot(color=ORANGE, radius=dot_radius).to_edge(LEFT, buff=6)
        pivot_base = pivot.get_center()

        # Make pivot move up and down as Delta a = -a*cos(nu*t) with a scale factor
        pivot_scale_factor = 0.3  # todo check
        pivot.add_updater(lambda mob: mob.move_to(
            pivot_base + pivot_scale_factor*UP * (-a * np.cos(nu * time_tracker.get_value()))
        ))

        # Helper, get array index for given time
        def idx():
            """
            --- Description: ---
            Index helper function that get's the current time index for use with simulated data.

            --- Parameters (Required): ---
                None
    
            --- Parameters (Optional): ---
                None
    
            --- Returns: ---
                The current index to use for fetching phi and phi_dot values.
            """
            i = int(time_tracker.get_value() * (n_frames - 1) / (t_max - t_min))
            return np.clip(i, 0, n_frames - 1)

        # Move state space dot
        def update_dot(mob):
            """
            --- Description: ---
            Updates the state space dot so it moves around the grid.

            --- Parameters (Required): ---
                1. mob - The MObject to move (in this case the dot).
    
            --- Parameters (Optional): ---
                None
    
            --- Returns: ---
                None
            """
            i = idx()
            mob.move_to(ax.c2p(t_pts[i], log_dphi[i]))
        dot.add_updater(update_dot)

        # Rod and bob of the pendulum
        rod = always_redraw(lambda: Line(
            start=pivot.get_center(),
            end=pivot.get_center() + np.array([
                l * np.sin(phi[idx()]),
                -l * np.cos(phi[idx()]),
                0
            ]),
            color=ORANGE
        ))
        bob = always_redraw(lambda: Dot(point=rod.get_end(), radius=dot_radius*1.5, color=YELLOW, fill_opacity=0.65))

        rod2 = always_redraw(lambda: Line(
            start=pivot.get_center(),
            end=pivot.get_center() + np.array([
                l * np.sin(phi2[idx()]),
                -l * np.cos(phi2[idx()]),
                0
            ]),
            color=ORANGE
        ))
        bob2 = always_redraw(lambda: Dot(point=rod2.get_end(), radius=dot_radius*1.5, color=BLUE, fill_opacity=0.65))

        # Driving pivot point (left end)
        lever_pivot = Triangle(color=GRAY, fill_color=GRAY, fill_opacity=1.).scale(0.5*scale_factor).to_edge(LEFT, buff=1.5)
        center_to_top_diff = lever_pivot.get_top()[1] - lever_pivot.get_center()[1]
        lever_pivot.move_to([lever_pivot.get_center()[0], -center_to_top_diff, 0])

        # Convert Manim's equilateral triangle to our custom ScaleneTriangle
        lever_pivot_vertices = lever_pivot.get_vertices()
        lever_pivot_vertices_range = np.abs(lever_pivot_vertices[1][0]) - np.abs(lever_pivot_vertices[2][0])
        lever_pivot_scale_factor = lever_pivot_vertices_range / 4
        lever_pivot_vertices[1][0] += np.abs(lever_pivot_scale_factor)
        lever_pivot_vertices[2][0] -= np.abs(lever_pivot_scale_factor)
        lever_pivot_sca = ScaleneTriangle(a=lever_pivot_vertices[0], b=lever_pivot_vertices[1], c=lever_pivot_vertices[2],
                                         color=GRAY, fill_opacity=0.9) 

        # Lever rod with the pivot attenuation
        lever_rod = always_redraw(lambda: Line(
            start=pivot_base + pivot_scale_factor*UP * (-a * np.cos(nu * time_tracker.get_value())),
            end=lever_pivot_sca.get_top(),
            color=ORANGE
        ))

        # Driving mechanism wheel
        # todo fix size and fix attenuation height
        wheel_rad = 0.4
        wheel = Circle(color=GRAY, fill_color=GRAY, fill_opacity=1., radius=wheel_rad).to_edge(LEFT, buff=4)
        wheel.move_to([lever_rod.get_center()[0], 0, 0])
        lever_rod_attenuator = always_redraw(lambda: Line(
            start=lever_rod.get_center(),
            end=wheel.get_center() + np.array([wheel_rad/2.5 * np.sin(nu * time_tracker.get_value()), 
                                               wheel_rad/2.5 * np.cos(nu * time_tracker.get_value()), 0]),
            color=ORANGE
        ))

        # Driving mechanism ground attacher
        wheel_base = Triangle(color=GRAY, fill_color=GRAY, fill_opacity=1.).scale(0.3*scale_factor)
        wheel_base_height = np.abs(lever_pivot.get_bottom()[1] - wheel.get_center()[1])
        wheel_base.height = wheel_base_height
        center_to_top_diff = wheel_base.get_top()[1] - wheel_base.get_center()[1]
        wheel_base.move_to([wheel.get_center()[0], wheel.get_center()[1] - center_to_top_diff, 0])

        # Convert ground attacher to scalene triangle
        wheel_base_vertices = wheel_base.get_vertices()
        wheel_base_vertices_range = np.abs(wheel_base_vertices[1][0]) - np.abs(wheel_base_vertices[2][0])
        wheel_range_scale_factor = wheel_base_vertices_range / 4
        wheel_base_vertices[1][0] += np.abs(wheel_range_scale_factor)
        wheel_base_vertices[2][0] -= np.abs(wheel_range_scale_factor)
        wheel_base_sca = ScaleneTriangle(a=wheel_base_vertices[0], b=wheel_base_vertices[1], c=wheel_base_vertices[2],
                                         color=GRAY, fill_opacity=0.9)

        # Rough vector arrow representing the movement of the pendulum
        # Related to the state space plot. Values are normalized
        '''
        arrow_scale_factor = 1  # This can be increased to extend the arrow's length if it is always small, but must be hard-coded
        movement_arrow = Arrow(
            start=rod.get_end(),
            end=rod.get_end() + np.array([
                arrow_scale_factor * phi_dot_norm[0] * np.cos(phi[0]),
                arrow_scale_factor * phi_dot_norm[0] * np.sin(phi[0]),
                0.
            ]),
            color=RED,
            stroke_width=8,
            buff=0,  # no buffer so the tail is at the center of the bob
            tip_length=0.2
        )
        movement_arrow.add_updater(lambda m: m.become(Arrow(
            start=rod.get_end(),
            end=rod.get_end() + np.array([
                arrow_scale_factor * phi_dot_norm[idx()] * np.cos(phi[idx()]),
                arrow_scale_factor * phi_dot_norm[idx()] * np.sin(phi[idx()]),
                0.
            ]),
            color=RED,
            stroke_width=8,
            buff=0,  # no buffer so the tail is at the enter of the bob
            tip_length=0.2
        )))
        '''

        # Vector group the entire set and add it
        group = VGroup(lever_pivot_sca, wheel_base_sca, wheel, lever_rod, pivot, rod, bob, rod2, bob2, lever_rod_attenuator)
        self.add(group)

        # Animation title
        omega0, epsilon, f = np.sqrt(g/l), a/l, nu/np.sqrt(g/l)

        delta_title = r',\ '
        if delta_phi == 0.0:
            delta_title += r'\Delta\phi_0 = 0.00,\ '
        else:
            exp_dphi = find_exp(delta_phi)
            num_dphi = delta_phi/(10**(exp_dphi))
            delta_title += r'\Delta\phi_0 = %.1f \times 10^{%d},\ ' % (num_dphi, exp_dphi)
        if delta_phi_dot == 0.0:
            delta_title += r'\Delta\dot{\phi}_0 = 0.00'
        else:
            exp_dphid = find_exp(delta_phi_dot)
            num_dphid = delta_phi_dot/(10**(exp_dphid))
            delta_title += r'\Delta\dot{\phi}_0 = %.2f \times 10^{%d}' % (num_dphid, exp_dphid)
        title = MathTex(r'\text{Kapitza Pendulum: }\omega_0 = %.2f,\ \epsilon = %.2f,\ f = %.2f,\ \phi_0 = %.2f,\ \dot{\phi}_0 = %.2f'
                        % (omega0, epsilon, f, phi_0, phi_dot_0,) + delta_title,
                        color=WHITE
                        )
        title.scale(0.5)
        title.to_edge(UP)
        self.add(title)

        # Time tracker text
        time_label = MathTex(r't = ')
        time_value = DecimalNumber(0, num_decimal_places=2, include_sign=False, unit=r'\text{ s}')  # Using DecimalNumber for better efficiency
        time_group = VGroup(time_label, time_value).arrange(RIGHT, buff=0.1).scale(0.7)
        time_group.next_to(title, DOWN, aligned_edge=LEFT, buff=0.1)
        time_value.add_updater(lambda m: m.set_value(time_tracker.get_value()))  # Update the current time in an efficient manner
        self.add(time_group)

        # Run real-time animation
        self.play(time_tracker.animate.set_value(t_max),
                  run_time=t_max,
                  rate_func=linear)  # <-- Linear is very important to make the sim time work in real-time

        # Cleanup
        dot.remove_updater(update_dot)
        self.wait()  # Adds an extra second to the animation

