import numpy as np


class MsgState:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
        
    def __init__(self,
                 pos=np.array([[0.], [0.]]),
                 vel=0.,
                 theta=0.,
                 ):
        self.pos = pos  # position in inertial frame
        self.vel = vel  # speed
        self.theta = theta  # angle from north
    
    def print(self):
        print('position=', self.pos,
              'velocity=', self.vel,
              'theta=', self.theta)
