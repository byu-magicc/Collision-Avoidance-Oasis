"""
    Simulation of a UAV flying with a bunch of moving obstacles in a 2D plane
"""

import numpy as np
from dynamics.constantVelocity import ConstantVelocity
from estimators.target_ekf import TargetEKF
from sensors.bearingSensor import BearingSensor
from controllers.twodbearingunzeroer import TwoDBearingNonzeroer
from viz.twoDViz import twoDViz

limits=[[-500,500],[-20,620]]
viz = twoDViz(limits)

t = 0.
ts = 0.01

tend = 60

max_roll = 20.*np.pi/180

v0 = 20
initial_yaw = 0
initial_pos = np.array([[0.,0.]]).T
uav = ConstantVelocity(ts, initial_pos, initial_yaw, v0)
commanded_yaw_rate = 0.

max_yaw_d = 9.81/v0 * np.tan(max_roll)

# create target to that will collide with the uav
tc = 30.
targetvel = 15
targetyaw = np.pi/2
xi = initial_pos.item(0)+tc*v0*np.sin(initial_yaw)-tc*targetvel*np.sin(targetyaw)
yi = initial_pos.item(1)+tc*v0*np.cos(initial_yaw)-tc*targetvel*np.cos(targetyaw)

target = ConstantVelocity(ts, np.array([[xi,yi]]).T, targetyaw, targetvel)

#setup the sensor
sensor = BearingSensor()
dif = np.reshape(target.true_state.getPos() - uav.true_state.getPos(),(2))
bearing = np.arctan2(dif[0], dif[1])
target_estimator = TargetEKF(bearing,initial_yaw, ts)

#setup the controller
controller = TwoDBearingNonzeroer(ts, 1, -max_yaw_d, max_yaw_d)

while t < tend:
    measurements = sensor.update(uav.true_state.getPos(), uav.true_state.yaw, [target.true_state.getPos()])
    target_estimator.update(measurements[0],uav.true_state,commanded_yaw_rate)

    commanded_yaw_rate = controller.update(measurements)

    uav.update(commanded_yaw_rate)
    target.update()

    viz.update(uav.true_state, [target.true_state])
    t += ts
