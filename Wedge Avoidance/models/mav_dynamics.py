import numpy as np
from message_types.msg_state import MsgState
import parameters.simulation_parameters as SIM


class MavDynamics:
    def __init__(self, x0):
        self._ts_simulation = SIM.ts_simulation
        self._state = np.array([[x0.pos[0, 0]],  # (0)
                               [x0.pos[1, 0]],   # (1)
                               [x0.theta],   # (2)
                               [x0.vel],  # (3)
                                ])
        # initialize true_state message
        self.state = MsgState(pos=x0.pos,
                              vel=x0.vel,
                              theta=x0.theta)

    ###################################
    # public functions
    def update(self, u):
        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._derivatives(self._state[0:4], u)
        k2 = self._derivatives(self._state[0:4] + time_step / 2. * k1, u)
        k3 = self._derivatives(self._state[0:4] + time_step / 2. * k2, u)
        k4 = self._derivatives(self._state[0:4] + time_step * k3, u)
        self._state[0:4] += time_step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        # update the message class for the true state
        self._update_true_state()

    def camera(self, intruders):
        bearings = []  # angles to intruders (rad)
        sizes = []  # sizes of intruder on camera (rad)
        for intruder in intruders:
            los = intruder.state.pos - self._state[0:2]
            bearing = np.arctan2(los[1, 0], los[0, 0]) - self._state[2, 0]
            size = SIM.uav_size / (np.linalg.norm(los) + .1)
            bearings.append(bearing)
            sizes.append(size)
        return bearings, sizes

    ###################################
    # private functions
    def _derivatives(self, state, u):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        north_dot = state[3, 0] * np.cos(state[2, 0])
        east_dot = state[3, 0] * np.sin(state[2, 0])
        theta_dot = u
        vel_dot = 0.
        # collect the derivative of the states
        x_dot = np.array([[north_dot, east_dot, theta_dot, vel_dot]]).T
        return x_dot

    def _update_true_state(self):
        self.state.pos = self._state[0:2]
        self.state.theta = self._state[2, 0]
        self.state.vel = self._state[3, 0]
