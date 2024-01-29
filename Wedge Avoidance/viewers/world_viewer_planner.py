from planning.dubins_params import DubinsParameters

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import keyboard
from matplotlib.path import Path
from matplotlib.widgets import Button, Slider
import numpy as np
plt.ion()  # enable interactive drawing

# for the rainbow lines
import matplotlib.cm as cm


class WorldViewer:
    def __init__(self, goal, sim_settings):
        self.sim_settings = sim_settings
        self.rad2deg = 180. / np.pi
        self.InitFlag = True
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2)
        self.fig.set_figheight(6)
        self.fig.set_figwidth(8)
        self.fig.subplots_adjust(bottom=0.25)
        self.ax[0].set(xlim=(-sim_settings.world_size, sim_settings.world_size),
                       ylim=(-sim_settings.world_size, sim_settings.world_size))
        self.ax[0].set_title('World View')
        self.ax[0].set_aspect('equal')
        self.ax[0].plot(goal[1, 0], goal[0, 0], marker="o", markersize=5)
        self.ax[1].set(xlim=(-1, 1), ylim=(-1, 1))
        self.ax[1].set_title('Camera View')
        self.ax[1].set_aspect('equal')
        self.world_handles = []
        self.camera_handles = []
        self.pixel_bearing = []
        self.pixel_size = []
        self.estimation_handles = {
            'lower_bound_line': False,
            'upper_bound_line': False,
            'current_wedge': False,
            'future_wedge_handles': [],
            'exact_lines': False
        }
        self.estimation_settings = sim_settings.estimation_settings
        self.path_handles = []
        self.dubins_radius = sim_settings.turn_radius

        # add the pause button to the figure
        button_ax = self.fig.add_axes([0.8, 0.1, 0.15, 0.08])
        button = Button(button_ax, 'Next Step')
        def play_pause(event):
            keyboard.press_and_release('p')
        button.on_clicked(play_pause)
        button_ax._button = button

        # add an exit button to stop the simulation
        exit_button_ax = self.fig.add_axes([0.8, 0.2, 0.10, 0.08])
        exitButton = Button(exit_button_ax, 'Exit')
        def exit_sim(event):
            keyboard.press_and_release('q')
        exitButton.on_clicked(exit_sim)
        exit_button_ax._exitbutton = exitButton

        future_ax = self.fig.add_axes([0.25, 0.1, 0.4, 0.05])
        self.future_slider = Slider(
            ax=future_ax,
            label='Future time steps',
            valmin=0,
            valmax=30,
            valinit=0,
            valstep=1.0
        )
        def update_slider(future_timestep):
            self.drawFutureIntruder(future_timestep=future_timestep)
            self.drawFutureWedge(future_timestep=future_timestep)
            self.drawFutureOwnship(future_timestep=future_timestep)
            self.drawFutureNoControlOwnship(future_timestep=future_timestep)

        self.future_slider.on_changed(update_slider)
        
        self.full_estimation_showing = False
        self.future_wedge_showing = False
        self.future_intruder_showing = False
        self.future_ownship_showing = False
        self.future_no_control_showing = False
        self.future_handles = {
            'intruder': False,
            'ownship': False,
            'wedge': False
        }




    def update(self, ownship, intruders, bearings, sizes, waypoints, estimation={}):
        if self.InitFlag == True:
            # draw camera ring
            self.camera_handles.append(mpatches.Wedge(
                center=(0., 0.), r=1.,
                theta1=0, theta2=360, width=0.2, color='green'))
            self.ax[1].add_patch(self.camera_handles[0])
            # draw vehicles and pixels
            self.drawMAV(ownship.state, color='blue')
            for intruder in intruders:
                self.drawMAV(intruder.state, color='red')
            self.drawCamera(bearings, sizes)
            self.InitFlag = False
        else:
            # clear the first axes object so that the estimation lines dont just pile up
            self.drawMAV(ownship.state, handle_id=0)
            # this will draw the wedge areas if there is an estimation object passed
            self.drawEstimation(estimation)

            id = 1
            for intruder in intruders:
                self.drawMAV(intruder.state, handle_id=id)
                id += 1
            self.drawCamera(bearings, sizes)

            # draw the path that is planned
            if waypoints.num_waypoints != 0:
                self.drawWaypoints(waypoints)
            
            # update the estimation quantities
            self.ownship = ownship.state
            if estimation['has_collision']:
                self.estimation = estimation
                self.intruder = intruder

            # Pause the sim and allow for slider to work
            keyboard.press_and_release('p')

            
        return self.pixel_bearing, self.pixel_size


    def drawCamera(self, bearings, sizes, handle_id=[]):
        if self.InitFlag == True:
            # draw bearing and size on ring
            for i in range(0, len(bearings)):
                self.camera_handles.append(mpatches.Wedge(
                    center=(0., 0.), r=1.,
                    theta1=self.rad2deg *
                    (np.pi / 2 - bearings[i] - sizes[i]),
                    theta2=self.rad2deg *
                    (np.pi / 2 - bearings[i] + sizes[i]),
                    width=0.2, color='brown'))
                self.ax[1].add_patch(self.camera_handles[-1])
        else:
            # update camera pixels
            for i in range(0, len(bearings)):
                self.camera_handles[i + 1].set(
                    theta1=self.rad2deg * (np.pi / 2 - bearings[i] - sizes[i]),
                    theta2=self.rad2deg * (np.pi / 2 - bearings[i] + sizes[i]),
                )

    def drawMAV(self, state, handle_id=[], color='blue'):
        pts = self.sim_settings.uav_size * np.array([
            [7., 0.],
            [5., 2.],
            [5., 12.],
            [2., 10.],
            [2., 2.],
            [-5., 2.],
            [-5., 5.],
            [-7., 5.],
            [-7., -5.],
            [-5., -5.],
            [-5., -2.],
            [2., -2.],
            [2., -10.],
            [5., -12.],
            [5., -2.],
            [7., 0.]]).T
        R = np.array([
            [np.cos(state.theta), np.sin(state.theta)],
            [-np.sin(state.theta), np.cos(state.theta)],
        ])
        pts = R.T @ pts
        pts = pts + np.tile(state.pos, (1, pts.shape[1]))
        R_world = np.array([[0., 1.], [1., 0.]])
        pts = R_world @ pts
        xy = np.array(pts.T)

        if self.InitFlag == True:
            self.world_handles.append(mpatches.Polygon(
                xy, facecolor=color, edgecolor=color))
            # Add the patch to the axes
            self.ax[0].add_patch(self.world_handles[-1])
        else:
            self.world_handles[handle_id].set_xy(xy)         # Update polygon

    def drawEstimation(self, estimation):
        if estimation and estimation['has_collision']:
            # this is the 'east' coordinates of the bottom line
            bottom_x_data = estimation['bottom_bound_line'][1]
            # this is the 'north' coordinates of the bottom line
            bottom_y_data = estimation['bottom_bound_line'][0]
            # this is the 'east' coordinates of the top line
            top_x_data = estimation['top_bound_line'][1]
            # this is the 'north' coordinates of the top line
            top_y_data = estimation['top_bound_line'][0]

            if self.estimation_settings['show_bounding_lines']:
                self.drawBoundingLines(
                    bottom_x_data, bottom_y_data, top_x_data, top_y_data)

            if self.estimation_settings['show_exact_estimates']:
                line_count = len(bottom_x_data)
                cmap = cm.get_cmap('rainbow', line_count)
                for i in range(0, line_count):
                    bottom_point = ([bottom_y_data[i], bottom_x_data[i]])
                    top_point = ([top_y_data[i], top_x_data[i]])

                    # world_view.drawPoint(np.array(bottom_point), color="blue")
                    # world_view.drawPoint(np.array(top_point), color="blue")
                    self.drawLine(x_data=[bottom_point[1], top_point[1]], y_data=[
                        bottom_point[0], top_point[0]], color=cmap(line_count - (i - 1)))

            if self.estimation_settings['show_wedges']:
                # make the wedge as a polygon patch

                initial_wedge = estimation['initial_wedge']

                # clear off the old patch and put in the new one
                if self.full_estimation_showing:
                    if self.estimation_handles['current_wedge']:
                        self.estimation_handles['current_wedge'].remove()
                self.estimation_handles['current_wedge'] = self.makeWedgePatch(close_right=initial_wedge['cr'], far_right=initial_wedge['fr'],
                                                                               far_left=initial_wedge['fl'], close_left=initial_wedge['cl'], color='red')
                self.ax[0].add_patch(
                    self.estimation_handles['current_wedge'])

                # now draw all of the future wedges
                if self.estimation_handles['future_wedge_handles']:
                    if self.full_estimation_showing:
                        # remove all the wedges
                        handle_count = len(
                            self.estimation_handles['future_wedge_handles'])
                        for i in range(handle_count):
                            self.estimation_handles['future_wedge_handles'][0].remove(
                            )
                            self.estimation_handles['future_wedge_handles'].pop(0)
                # add the future wedge patches
                for i in range(len(estimation['future_data']['wedges'])):
                    next_wedge = estimation['future_data']['wedges'][i]
                    self.estimation_handles['future_wedge_handles'].append(self.makeWedgePatch(
                        close_right=next_wedge['cr'], close_left=next_wedge['cl'], far_right=next_wedge[
                            'fr'], far_left=next_wedge['fl'], color=next_wedge['color']
                    ))

                for wedge_patch in self.estimation_handles['future_wedge_handles']:
                    self.ax[0].add_patch(wedge_patch)
            self.full_estimation_showing = True

    def drawWaypoints(self, waypoints):
        # First check if there are any handles currently in the path_handles list,
        # and if so, clear them out
        if len(self.path_handles) > 0:
            for handle in self.path_handles:
                handle.remove()
            self.path_handles.clear()
        
        dubins_path = DubinsParameters()
        for j in range(0, waypoints.num_waypoints-1):
            dubins_path.compute_parameters(
                waypoints.ne[:, j:j+1],
                waypoints.course.item(j),
                waypoints.ne[:, j+1:j+2],
                waypoints.course.item(j+1),
                self.dubins_radius)
            if j == 0:
                points = dubins_path.compute_points()
            else:
                points = np.concatenate((points, dubins_path.compute_points()), axis=0)

        self.dubins_points = points
        # self.dubins_timestamps

        vertices = [(points[0].item(1), points[0].item(0))]
        codes = [Path.MOVETO]

        for i in range(1, points.shape[0]):
            vertices.append((points[i].item(1), points[i].item(0)))
            codes.append(Path.LINETO)
            # TODO: get the future ownship points working correctly - right now, 
            # the points do not predict the correct position of the ownship,
            # they are simply just the points of the dubins path.
            # self.drawPoint(coords=points[i])

        path = Path(vertices=vertices, codes=codes)
        path_handle = mpatches.PathPatch(path, facecolor='none', edgecolor='green')
        self.path_handles.append(path_handle)
        self.ax[0].add_patch(path_handle)
        
        return

    def drawBoundingLines(self, bottom_x_data, bottom_y_data, top_x_data, top_y_data):
        path_codes = [
            Path.MOVETO,
            Path.LINETO
        ]
        # lower bounding line
        lower_vertices = [
            (bottom_x_data[0], bottom_y_data[0]),
            (bottom_x_data[-1], bottom_y_data[-1])
        ]

        lower_path = Path(lower_vertices, path_codes)

        if self.estimation_handles['lower_bound_line']:
            self.estimation_handles['lower_bound_line'].remove()
        self.estimation_handles['lower_bound_line'] = mpatches.PathPatch(
            lower_path, edgecolor="red")

        self.ax[0].add_patch(
            self.estimation_handles['lower_bound_line'])

        # upper bounding line
        upper_vertices = [
            (top_x_data[0], top_y_data[0]),
            (top_x_data[-1], top_y_data[-1])
        ]

        upper_path = Path(upper_vertices, path_codes)
        if self.estimation_handles['upper_bound_line']:
            self.estimation_handles['upper_bound_line'].remove()
        self.estimation_handles['upper_bound_line'] = mpatches.PathPatch(
            upper_path, edgecolor="red")
        self.ax[0].add_patch(
            self.estimation_handles['upper_bound_line'])

    def drawPoint(self, coords, color="red", marker_size=3):
        self.ax[0].plot(coords.item(1), coords.item(0),
                        marker="o", markersize=marker_size, color=color)

    def drawEllipse(self, center, major_length, minor_length, angle):
        ellipse = mpatches.Ellipse((center.item(1), center.item(
            0)), major_length, minor_length, angle=angle * 180 / np.pi, edgecolor='red', facecolor='none')
        self.ax[0].add_patch(ellipse)

    def drawLine(self, x_data, y_data, color="red", line_style='-'):
        self.ax[0].plot(x_data, y_data, linestyle=line_style, color=color)

    def drawWedge(self, close_right, far_right, far_left, close_left, color='red'):
        # find the four points in 2d Space to make a wedge

        wedge = mpatches.Polygon(
            xy=self.getWedgeXY(close_right, far_right, far_left, close_left), closed=True, edgecolor=color, facecolor='none')
        self.ax[0].add_patch(wedge)

    def makeWedgePatch(self, close_right, far_right, far_left, close_left, color='red'):
        return mpatches.Polygon(
            xy=self.getWedgeXY(close_right, far_right, far_left, close_left), closed=True, edgecolor=color, facecolor='none')

    def getWedgeXY(self, close_right, far_right, far_left, close_left):
        return ([[close_left.item(1), close_left.item(0)], [close_right.item(1), close_right.item(0)], [far_right.item(1), far_right.item(0)], [far_left.item(1), far_left.item(0)]])
    
    def drawFutureIntruder(self, future_timestep):
        intruder = self.intruder
        if self.future_intruder_showing:
            #remove the future intruder in order to redraw
            if self.future_handles['intruder']:
                self.future_handles['intruder'].remove()
        future_intruder_point = intruder.state.pos + future_timestep * intruder.state.vel * np.array([[np.cos(intruder.state.theta)], [np.sin(intruder.state.theta)]])
        self.future_handles['intruder'] = mpatches.Circle((future_intruder_point.item(1), future_intruder_point.item(0)), radius=18.0, color='red')
        self.ax[0].add_patch(self.future_handles['intruder'])
        self.future_intruder_showing = True

    def drawFutureWedge(self, future_timestep):
        estimation = self.estimation
        if future_timestep == 0:
            self.drawEstimation(estimation)
        else:
            if self.full_estimation_showing:
                # first, clean out the wedges that are on the plot --------------------------------------------
                if self.estimation_handles['current_wedge']:
                    self.estimation_handles['current_wedge'].remove()

                if self.estimation_handles['future_wedge_handles']:
                    # remove all the wedges
                    handle_count = len(
                        self.estimation_handles['future_wedge_handles'])
                    for i in range(handle_count):
                        self.estimation_handles['future_wedge_handles'][0].remove(
                        )
                        self.estimation_handles['future_wedge_handles'].pop(0)
                self.full_estimation_showing = False
                
                # ---------------------------------------------------------------------------------------------
            if self.future_handles['wedge']:
                self.future_handles['wedge'].remove()
             
            wedge_timesteps = estimation['future_data']['timestamps']
            wedges = estimation['future_data']['wedges']

            found_wedge = False
            current_wedge_index = 0
            while not found_wedge and current_wedge_index < len(wedge_timesteps) - 1:
                if future_timestep > wedge_timesteps[current_wedge_index]:
                    current_wedge_index += 1
                else:
                    found_wedge = True

            intruder_wedge = wedges[current_wedge_index] 
            self.future_handles['wedge'] = self.makeWedgePatch(close_right=intruder_wedge['cr'], far_right=intruder_wedge['fr'], far_left=intruder_wedge['fl'], close_left=intruder_wedge['cl'])
            self.ax[0].add_patch(self.future_handles['wedge'])
            
    def drawFutureOwnship(self, future_timestep):
        # remove the future ownship patch if it is there
        if self.future_ownship_showing:
            if self.future_handles['ownship']:
                self.future_handles['ownship'].remove()

        integer_timestep = int(future_timestep)
        if integer_timestep > self.dubins_points.shape[0]:
            integer_timestep = self.dubins_points.shape[0]

        future_ownship_point = self.dubins_points[integer_timestep]

        self.future_handles['ownship'] = mpatches.Circle((future_ownship_point.item(1), future_ownship_point.item(0)), radius=18.0, color='blue')
        self.ax[0].add_patch(self.future_handles['ownship'])
        self.future_ownship_showing = True


        
    def drawFutureNoControlOwnship(self, future_timestep):
        if future_timestep == 0:
            self.future_handles['no_control_ownship'].remove()
            self.future_no_control_showing = False
        else:
            ownship = self.ownship
            # remove the future ownship patch if it is there
            if self.future_no_control_showing:
                self.future_handles['no_control_ownship'].remove()

            future_no_control_point = ownship.pos + future_timestep * ownship.vel * np.array([[np.cos(ownship.theta)], [np.sin(ownship.theta)]])
            self.future_handles['no_control_ownship'] = mpatches.Circle((future_no_control_point.item(1), future_no_control_point.item(0)), radius=18.0, color='green')
            self.ax[0].add_patch(self.future_handles['no_control_ownship'])
            self.future_no_control_showing = True



    def wait(self):
        plt.waitforbuttonpress()
