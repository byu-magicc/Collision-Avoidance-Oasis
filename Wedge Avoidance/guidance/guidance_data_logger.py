import numpy as np
from guidance.tools.dirty_derivative import DirtyDerivative
import parameters.simulation_parameters as SIM


class Guidance:
    def __init__(self):
        foo = 0.  # do nothing at initialization
        self.prev_bearing = 0.0
        self.isInitialized = False

        self.size_derivative = DirtyDerivative(
            Ts=SIM.ts_simulation, tau=5 * SIM.ts_simulation)
        self.bearing_derivative = DirtyDerivative(
            Ts=SIM.ts_simulation, tau=5 * SIM.ts_simulation)

    def update(self, ownship_state, bearings, sizes):

        # get the largest intruder
        intruder_index = np.argmax(sizes)

        bearing = bearings[intruder_index]
        bearing_dot = self.bearing_derivative.update(z=bearing)

        bearing_delta = bearing - self.prev_bearing
        print("bearing change: ", bearing_delta)
        print("bearing dirty derivative: ", bearing_dot)

        self.prev_bearing = bearing
        u = 0.  # no guidance
        return u
