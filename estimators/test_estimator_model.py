
import numpy as np
from estimators.target_ekf import TargetEKF
from msg.twoDYawState import TwoDYawState

class TestEstimatorModel:

    def __init__(self, bearing, ttc, vel, yawi, yaw, ts) -> None:
        self.x = np.array([[bearing, ttc, vel, yawi, yaw]]).T
        self.ekf = TargetEKF(bearing, yaw,ts)
        self.ts = ts

        self.state = TwoDYawState()

    def update(self,uav_state, input):
        timestep = self.ts
        x1 = self.ekf._f(self.x, None, uav_state, input)
        x2 = self.ekf._f(self.x+timestep/2.*x1, None, uav_state, input)
        x3 = self.ekf._f(self.x+timestep/2.*x2, None, uav_state, input)
        x4 = self.ekf._f(self.x+timestep*x3, None, uav_state, input)
        self.x += timestep/6.*(x1 + 2*x2 + 2*x3 + x4)

        theta = self.x.item(4)+self.x.item(0)
        vo = uav_state.vel
        d = self.x.item(1)*vo
        xpos = d*np.sin(theta) + uav_state.getPos().item(0)
        ypos = d*np.cos(theta) + uav_state.getPos().item(1)

        self.state = TwoDYawState(xpos, ypos, self.x.item(3),self.x.item(2))