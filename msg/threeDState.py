
import numpy as np

class ThreeDState:
    def __init__(self, xpos=0., ypos=0., zpos=0., xvel=0., yvel=0., zvel=0.) -> None:
        self.xpos = xpos
        self.ypos = ypos
        self.zpos = zpos
        self.xvel = xvel
        self.yvel = yvel
        self.zvel = zvel
    
    def toArray(self):
        return np.array([[self.xpos, self.ypos, self.zpos, self.xvel, self.yvel, self.zvel]]).T
    
    def fromArray(self, array:np.ndarray):
        self.xpos = array.item(0)
        self.ypos = array.item(1)
        self.zpos = array.item(2)
        self.xvel = array.item(3)
        self.yvel = array.item(4)
        self.zvel = array.item(5)
    
    def getPos(self):
        return np.array([[self.xpos, self.ypos, self.zvel]]).T