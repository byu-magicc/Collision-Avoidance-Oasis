"""
    Simulation of a UAV flying with a bunch of moving obstacles in a 2D plane
"""

import numpy as np
from dynamics.constantVelocity import ConstantVelocity
from estimators.target_ekf import TargetEKF
from sensors.bearingSensor import BearingSensor

t = 0.
ts = 0.01

tend = 60

v0 = 20
initial_yaw = 0
initial_pos = np.array([[0.,0.]]).T
uav = ConstantVelocity(ts, initial_pos, initial_yaw, v0)

# create target to that will collide with the uav
tc = 30.
targetvel = 15
targetyaw = np.pi/2
xi = initial_pos.item(0)+tc*v0*np.sin(initial_yaw)-tc*targetvel*np.sin(targetyaw)
yi = initial_pos.item(1)+tc*v0*np.cos(initial_yaw)-tc*targetvel*np.cos(targetyaw)

target = ConstantVelocity(ts, np.array([[xi,yi]]).T, targetyaw, targetvel)

while t < tend:
    pass
