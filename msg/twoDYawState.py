
import numpy as np

class TwoDYawState:
    def __init__(self, xpos=0., ypos=0., yaw=0., vel=0.) -> None:
        self.xpos = xpos
        self.ypos = ypos
        self.yaw = yaw
        self.vel = vel
    
    def toArray(self):
        return np.array([[self.xpos, self.ypos, self.yaw, self.vel]]).T
    
    def fromArray(self, array:np.ndarray):
        self.xpos = array.item(0)
        self.ypos = array.item(1)
        self.yaw = array.item(2)
        self.vel = array.item(3)