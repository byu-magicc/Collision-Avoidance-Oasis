"""
constantVelocity
    - This file implements the dynamic equations of motion for an object moving at constant velocity in the 2D plane
"""
import numpy as np
from copy import deepcopy
from msg.twoDYawState import TwoDYawState

class ConstantVelocity:
    def __init__(self, Ts, initialPosition:np.ndarray, initial_yaw:float, velocity:float) -> None:
        self._ts = Ts

        self._state = deepcopy(initialPosition)
        self._state = np.append(self._state, np.array([[initial_yaw, velocity]]).T, axis=0)
        self.true_state = TwoDYawState()
        self.true_state.fromArray(self._state)
    
    def update(self, psid=0.):
        """
            Integrate the differential equations defining dynamics.
            Ts is the time step between function calls.
        """

        # Constant velocity doesn't need fancy RK4 algorithm,
        # just use Newton's method
        psi = self._state.item(2)
        vel = self._state.item(3)

        self._state[0:3] += np.array([[vel*np.sin(psi), vel*np.cos(psi), psid]]).T*self._ts
        self.true_state.fromArray(self._state)
        