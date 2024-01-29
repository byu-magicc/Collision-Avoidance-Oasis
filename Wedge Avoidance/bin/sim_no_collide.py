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
import parameters.simulation_parameters_no_collide as SIM
from models.mav_dynamics import MavDynamics
from guidance.guidance_dyn_wedge import Guidance

# initialize goal point (north, east)
goal = np.array([[400, 0]])
# initialize viewer
world_view = WorldViewer(goal=goal)
# initialize vehicles
ownship = MavDynamics(SIM.ownship0)
intruders = []
for intruder0 in SIM.intruders0:
    intruders.append(MavDynamics(intruder0))
# initialize guidance algorithm
guidance = Guidance()
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

        # plot the points of interest on the world view
        if collision['has_collision']:
            world_view.drawPoint(collision['closest_point'], color="green")
            world_view.drawPoint(collision['farthest_point'])
            # this is the 'east' coordinates of the bottom line
            bottom_x_data = collision['bottom_bound_line'][1]
            # this is the 'north' coordinates of the bottom line
            bottom_y_data = collision['bottom_bound_line'][0]
            # this is the 'east' coordinates of the top line
            top_x_data = collision['top_bound_line'][1]
            # this is the 'north' coordinates of the top line
            top_y_data = collision['top_bound_line'][0]

            world_view.drawLine(
                x_data=bottom_x_data, y_data=bottom_y_data)

            world_view.drawLine(
                x_data=top_x_data, y_data=top_y_data)


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
