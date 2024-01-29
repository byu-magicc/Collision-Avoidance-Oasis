"""
    Controller designed to make the derivative of bearing to each obstacle non-zero
"""

import numpy as np
from typing import List
from tools.dirty_derivative import DirtyDerivative
from msg.bearing_msg import BearingMsg

class TwoDBearingNonzeroer:
    def __init__(self, Ts, num_targets, lower_bound, upper_bound) -> None:
        self.Ts = Ts
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.yaw_derivative = DirtyDerivative(Ts, 5*Ts)
        self.bearing_derivatives = []
        for i in range(num_targets):
            bearing_derivative = DirtyDerivative(Ts, 5*Ts)
            self.bearing_derivatives.append(bearing_derivative)


    def update(self, measurements:List[BearingMsg]):
        yawd = self.yaw_derivative.update(measurements[0].yaw)
        smallestbd = float('inf')
        indexsmallest = -1
        for i, measurement in enumerate(measurements):
            bd = self.bearing_derivatives[i].update(measurement.bearing)# + yawd
            if np.abs(bd) < smallestbd:
                smallestbd = bd
                indexsmallest = i
        gain = 10e-4
        if smallestbd == 0.0:
            return self.saturate(-gain/(smallestbd+0.001))
        return self.saturate(-gain/smallestbd)

    def saturate(self, input):
        if input > self.upper_bound:
            return self.upper_bound
        elif input < self.lower_bound:
            return self.lower_bound
        return input
        
