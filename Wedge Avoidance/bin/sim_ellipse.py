import os
import sys

# ----------------------------------------------------
# Collision approaching from the left
# ----------------------------------------------------

# insert parent directory at beginning of python search path
from pathlib import Path
sys.path.insert(0, os.fspath(Path(__file__).parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from viewers.world_viewer import WorldViewer
import parameters.simulation_parameters_ellipse as SIM
from models.mav_dynamics import MavDynamics
from guidance.guidance_ellipse import Guidance

# initialize goal point (x, y)
goal = np.array([[100, 400]])
# initialize viewer
world_view = WorldViewer(goal=goal)
# initialize vehicles
ownship = MavDynamics(SIM.ownship0)
intruders = []
for intruder0 in SIM.intruders0:
    intruders.append(MavDynamics(intruder0))
# initialize guidance algorithm
guidance = Guidance(goal=goal)
# initialize the simulation time
sim_time = SIM.start_time
# main simulation loop
while sim_time < SIM.end_time:
    # Propagate dynamics in between plot samples
    t_next_plot = sim_time + SIM.ts_plotting
    while (sim_time < t_next_plot):  # updates control and dynamics at faster simulation rate

        # get camera measurements
        bearings, sizes = ownship.camera(intruders)
        # compute guidance action
        u, collision = guidance.update(ownship.state, bearings, sizes)
        if collision['has_collision']:
            world_view.drawPoint(collision['collision_point'])
            intr_est = collision['intruder_estimate']
            world_view.drawEllipse(
                center=intr_est[0], major_length=intr_est[1], minor_length=intr_est[2], angle=intr_est[3])
            world_view.drawLine(
                collision['testing_top_line'][0], collision['testing_top_line'][1])
            world_view.drawLine(
                collision['testing_bottom_line'][0], collision['testing_bottom_line'][1])
            world_view.drawLine(
                collision['testing_mid_line'][0], collision['testing_mid_line'][1])
            world_view.drawPoint(collision['next_mid'])
            world_view.drawPoint(collision['top_intercept'])
            world_view.drawPoint(collision['bottom_intercept'])

            next_ellipse = collision['testing_next_ellipse']
            world_view.drawEllipse(
                center=next_ellipse[0], major_length=next_ellipse[1], minor_length=next_ellipse[2], angle=next_ellipse[3])

            plt.waitforbuttonpress()
        # move ownship
        ownship.update(u)
        # move intruders
        for intruder in intruders:
            intruder.update(0.0)
        sim_time += SIM.ts_simulation

    # -------update viewers-------------
    world_view.update(ownship, intruders, bearings, sizes)
    # the pause causes the figure to be displayed during the simulation
    plt.pause(0.0001)
    if sim_time > 5:
        pass

# Keeps the program from closing until the user presses a button.
print("Press key to close")
plt.waitforbuttonpress()
plt.close()
