import numpy as np
from controllers.helical_navigation_law import HelicalNavigationLaw
from dynamics.constant_velocity3D import ConstantVelocity
from estimators.plkf_3d import PseudoLinearKF
from sensors.unitVectorSensor import UnitVectorSensor
from viz.threeDViz import ThreeDViz
from viz.plkfViz3D import PLKFViz

USE_INVERSE = True

limits=[[-100,1100],[-16,16], [-16,16]]
viz = ThreeDViz(limits)

t = 0.
ts = 0.01

steps = 0
plotsteps = 10

tend = 80

radius = 10.
omega = 1.

v0 = np.array([[20, 0., 0.]]).T
initial_pos = np.array([[0.,0.,0.]]).T
uav = ConstantVelocity(ts, initial_pos + np.array([[0., radius, 0.]]).T, v0+np.array([[0., 0., radius*omega]]).T)

# create target to that will collide with the uav
tc = 30.
targetVel = np.array([[-15., 0., 0.]]).T
targetPos = initial_pos + (v0 - targetVel)*tc

estimator_viz = PLKFViz(targetVel)

target = ConstantVelocity(ts,targetPos,targetVel)

#setup the sensor
sensor = UnitVectorSensor()
target_estimator = None

#setup the controller
controller = HelicalNavigationLaw(ts, radius, omega)

while t < tend:
    measurements = sensor.update(uav.true_state.getPos(), [target.true_state.getPos()])
    if target_estimator is None:
        target_estimator = PseudoLinearKF(ts, uav.true_state.toArray(), measurements[0])
    else:
        target_estimator.update(uav.true_state.toArray(),measurements[0])

    commanded_accel = controller.update(np.array([[1., 0., 0.]]).T, uav.true_state.toArray()[3:])

    uav.update(commanded_accel)
    target.update()
    # testmodel.update(uav.true_state, commanded_yaw_rate)
    
    estimator_viz.update(uav.true_state, target.true_state, target_estimator.xhat, t)
    t += ts
    steps += 1
    if steps % plotsteps == 0:
        viz.update(uav.true_state, [target.true_state])
        estimator_viz.update_plots()
