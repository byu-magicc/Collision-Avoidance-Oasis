"""
constantVelocity
    - This file implements the dynamic equations of motion for an object moving at constant velocity
"""
import numpy as np
from copy import deepcopy
from msg.threeDState import ThreeDState

class ConstantVelocity:
    def __init__(self, Ts, initialPosition:np.ndarray, velocity:np.ndarray) -> None:
        self._ts = Ts

        self._state = np.concatenate((initialPosition, velocity), axis=0)
        self.true_state = ThreeDState()
        self.true_state.fromArray(self._state)
    
    def update(self, accel=np.array([[0.,0.,0.]]).T):
        """
            Integrate the differential equations defining dynamics.
            Ts is the time step between function calls.
        """

        # Constant velocity doesn't need fancy RK4 algorithm,
        # just use Newton's method
        self._state[0:3] += self._state[3:]*self._ts
        self._state[3:] += accel*self._ts
        self.true_state.fromArray(self._state)
        