
import numpy as np

class BearingMsg:
    def __init__(self, bearing=0., yaw=0.) -> None:
        self.bearing = bearing
        self.yaw = yaw