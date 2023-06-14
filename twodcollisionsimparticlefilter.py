"""
    Simulation of a UAV flying with a bunch of moving obstacles in a 2D plane
"""

import numpy as np
from dynamics.constantVelocity import ConstantVelocity
from estimators.ttc_particle_filter import TTCParticleFilter
from estimators.inverse_depth_particle_filter import InverseDepthParticleFilter
from sensors.bearingSensor import BearingSensor
from controllers.twodbearingunzeroer import TwoDBearingNonzeroer
from viz.twoDVizWithParticles import twoDVizWithParticles

USE_INVERSE = False

limits=[[-1000,500],[-100,1500]]
viz = twoDVizWithParticles(limits)

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
commanded_yaw_rate = -0.05#-max_yaw_d

# create target to that will collide with the uav
tc = 30.
targetvel = 15
targetyaw = np.pi/2
xi = initial_pos.item(0)+tc*v0*np.sin(initial_yaw)-tc*targetvel*np.sin(targetyaw)+50
yi = initial_pos.item(1)+tc*v0*np.cos(initial_yaw)-tc*targetvel*np.cos(targetyaw)

target = ConstantVelocity(ts, np.array([[xi,yi]]).T, targetyaw, targetvel)

#setup the sensor
sensor = BearingSensor()
target_estimator = None

#setup the controller
# controller = TwoDBearingNonzeroer(ts, 1, -max_yaw_d, max_yaw_d)
measurements = sensor.update(uav.true_state.getPos(), uav.true_state.yaw, [target.true_state.getPos()])
if USE_INVERSE:
    target_estimator = InverseDepthParticleFilter(measurements[0].bearing, measurements[0].yaw, ts)
else:
    target_estimator = TTCParticleFilter(measurements[0].bearing, measurements[0].yaw, ts)

while t < tend:
    measurements = sensor.update(uav.true_state.getPos(), uav.true_state.yaw, [target.true_state.getPos()])

    target_estimator.propagate_model(uav.true_state, commanded_yaw_rate)
    target_estimator.measurement_update(measurements[0])
    target_estimator.resample(measurements[0])

    # commanded_yaw_rate = controller.update(measurements)

    uav.update(commanded_yaw_rate)
    target.update()
    t += ts
    steps += 1
    if steps % plotsteps == 0:
        viz.update(uav.true_state, target.true_state, target_estimator.get_particle_states(uav.true_state),t)
