import numpy as np
from guidance.tools.wrap import wrap, wrap2


class Controller:

    def __init__(self, sim_settings) -> None:
        self.chi_inf = np.radians(50)  # approach angle for large distance from straight-line path
        self.k_path = 0.05  # path gain for straight-line path following
        self.k_orbit = 5 #10.0  # path gain for orbit following
        self.gravity = 9.8
        self.control_disabled = sim_settings.control_disabled


    def update(self, path, state):
        # check to see if control is disabled (for testing purposes)
        if self.control_disabled:
            return 0
        else:
            # return the heading rate that will get the ownship to the path
            if path.type == 'line':
                return self._follow_straight_line(path, state)
            elif path.type == 'orbit':
                return self._follow_orbit(path, state)

    def _follow_straight_line(self, path, state):
        theta_q = np.arctan2(path.line_direction.item(1),
                           path.line_direction.item(0))
        theta_q = wrap2(theta_q, state.theta)
        ep = np.array([[state.pos.item(0)], [state.pos.item(1)]]) - path.line_origin
        path_error = -np.sin(theta_q) * ep.item(0) + np.cos(theta_q) * ep.item(1)
        # course command
        theta_des = theta_q - self.chi_inf * (2 / np.pi) * np.arctan(self.k_path * path_error)

        return des_to_heading_rate(theta_des, state.theta)

    def _follow_orbit(self, path, state):
        if path.orbit_direction == 'CW':
            direction = 1.0
        else:
            direction = -1.0
        # distance from orbit center
        d = np.sqrt((state.pos.item(0) - path.orbit_center.item(0))**2
                    + (state.pos.item(1) - path.orbit_center.item(1))**2)
        # compute wrapped version of angular position on orbit
        varphi = np.arctan2(state.pos.item(1) - path.orbit_center.item(1),
                            state.pos.item(0) - path.orbit_center.item(0))
        varphi = wrap2(varphi, state.theta)
        # compute normalized orbit error
        orbit_error = (d - path.orbit_radius) / path.orbit_radius
        # course command
        theta_des = varphi + direction * (np.pi/2.0 + np.arctan(self.k_orbit * orbit_error))

        return des_to_heading_rate(theta_des, state.theta)
    
    
def des_to_heading_rate(theta_des, theta_state):
    return np.arctan(wrap(theta_des - theta_state))