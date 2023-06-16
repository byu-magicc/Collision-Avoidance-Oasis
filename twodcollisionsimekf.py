"""
    Simulation of a UAV flying with a bunch of moving obstacles in a 2D plane
"""

import numpy as np
from dynamics.constantVelocity import ConstantVelocity
from estimators.target_ekf import TargetEKF
from estimators.inverse_depth_ekf import InverseDepthEKF
from sensors.bearingSensor import BearingSensor
from controllers.twodbearingunzeroer import TwoDBearingNonzeroer
from viz.twoDViz import twoDViz
from viz.twoDEstimatorViz import TwoDEstimatorViz
from viz.inverseDEstimatorViz import InverseDEstimatorViz
from estimators.test_inverse_depth_model import TestInverseDepthModel

USE_INVERSE = False

limits=[[-500,500],[-100,1200]]
viz = twoDViz(limits)

t = 0.
ts = 0.01

steps = 0
plotsteps = 10

tend = 60

max_roll = 20.*np.pi/180

v0 = 20
initial_yaw = 0.
initial_pos = np.array([[0.,0.]]).T
uav = ConstantVelocity(ts, initial_pos, initial_yaw, v0)

max_yaw_d = 9.81/v0 * np.tan(max_roll)
commanded_yaw_rate = -0.01#-max_yaw_d

# create target to that will collide with the uav
tc = 30.
targetvel = 15
targetyaw = np.pi/2
xi = initial_pos.item(0)+tc*v0*np.sin(initial_yaw)-tc*targetvel*np.sin(targetyaw)+50
yi = initial_pos.item(1)+tc*v0*np.cos(initial_yaw)-tc*targetvel*np.cos(targetyaw)

if (USE_INVERSE):
    estimator_viz = InverseDEstimatorViz(targetvel,targetyaw)
else:
    estimator_viz = TwoDEstimatorViz(targetvel, targetyaw)

target = ConstantVelocity(ts, np.array([[xi,yi]]).T, targetyaw, targetvel)

#setup the test of the estimator model with the info on the target
dif = np.array([[xi,yi]]).T - initial_pos
bearing = np.arctan2(dif.item(0), dif.item(1)) - initial_yaw
rho0 = 1/np.linalg.norm(dif)
testmodel = TestInverseDepthModel(bearing, rho0, targetvel, targetyaw, initial_yaw, ts)

#setup the sensor
sensor = BearingSensor()
target_estimator = None

#setup the controller
# controller = TwoDBearingNonzeroer(ts, 1, -max_yaw_d, max_yaw_d)

while t < tend:
    measurements = sensor.update(uav.true_state.getPos(), uav.true_state.yaw, [target.true_state.getPos()])
    if target_estimator is not None:
        target_estimator.update(measurements[0],uav.true_state,commanded_yaw_rate)
    else:
        if USE_INVERSE:
            target_estimator = InverseDepthEKF(measurements[0].bearing,uav.true_state.yaw, ts)
        else:
            target_estimator = TargetEKF(measurements[0].bearing, uav.true_state.yaw, ts)

    # commanded_yaw_rate = controller.update(measurements)

    uav.update(commanded_yaw_rate)
    target.update()
    testmodel.update(uav.true_state, commanded_yaw_rate)
    
    estimator_viz.update(uav.true_state, target.true_state, target_estimator.xhat, t, measurements[0])
    t += ts
    steps += 1
    if steps % plotsteps == 0:
        viz.update(uav.true_state, [target.true_state, testmodel.state])
        estimator_viz.update_plots()
