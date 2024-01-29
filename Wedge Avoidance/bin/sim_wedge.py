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
from guidance.guidance_wedge import Guidance

goal = np.array([[400, 100]])
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
        u, collision = guidance.update(
            ownship.state, bearings, sizes, SIM.ts_simulation)
        if collision['has_collision']:
            world_view.drawPoint(collision['collision_point'])

            # Bound points
            # world_view.drawPoint(
            #     collision['min_intruder_point'], color="purple")
            # world_view.drawPoint(
            #     collision['max_intruder_point'], color="green")
            # Boudning envelope lines ---------------------
            # world_view.drawLine(
            #     collision['testing_top_line'][0], collision['testing_top_line'][1])
            # world_view.drawLine(
            #     collision['testing_bottom_line'][0], collision['testing_bottom_line'][1])
            # ---------------------------------------------

            # Initial Wedge points
            # world_view.drawPoint(collision['close_point_right'], color="blue")
            # world_view.drawPoint(collision['close_point_left'], color="green")
            # world_view.drawPoint(collision['far_point_right'], color="orange")
            # world_view.drawPoint(collision['far_point_left'], color="yellow")

            # Next wedges
            future_wedges = collision['future_wedges']
            for i in range(len(future_wedges)):
                current_wedge = future_wedges[i]
                world_view.drawWedge(close_right=current_wedge['cr'], far_right=current_wedge['fr'],
                                     far_left=current_wedge['fl'], close_left=current_wedge['cl'], color=current_wedge['color'])

            # Initial wedge
            initial_wedge = collision['initial_wedge']
            world_view.drawWedge(close_right=initial_wedge['cr'], far_right=initial_wedge['fr'],
                                 far_left=initial_wedge['fl'], close_left=initial_wedge['cl'])

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
