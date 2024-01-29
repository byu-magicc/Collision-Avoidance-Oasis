import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import numpy as np
import parameters.simulation_parameters as SIM
plt.ion()  # enable interactive drawing

# for the rainbow lines
import matplotlib.cm as cm


class WorldViewer:
    def __init__(self, goal, estimation_settings):
        self.rad2deg = 180. / np.pi
        self.InitFlag = True
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2)
        self.ax[0].set(xlim=(-SIM.world_size / 2, SIM.world_size / 2),
                       ylim=(-SIM.world_size / 2, SIM.world_size / 2))
        self.ax[0].set_title('World View')
        self.ax[0].set_aspect('equal')
        self.ax[0].plot(goal[0, 1], goal[0, 0], marker="o", markersize=5)
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
        self.estimation_settings = estimation_settings

    def update(self, ownship, intruders, bearings, sizes, estimation={}):
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
        pts = SIM.uav_size * np.array([
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
                if self.estimation_handles['current_wedge']:
                    self.estimation_handles['current_wedge'].remove()
                self.estimation_handles['current_wedge'] = self.makeWedgePatch(close_right=initial_wedge['cr'], far_right=initial_wedge['fr'],
                                                                               far_left=initial_wedge['fl'], close_left=initial_wedge['cl'], color='red')
                self.ax[0].add_patch(
                    self.estimation_handles['current_wedge'])

                # now draw all of the future wedges
                if self.estimation_handles['future_wedge_handles']:
                    # remove all the wedges
                    handle_count = len(
                        self.estimation_handles['future_wedge_handles'])
                    for i in range(handle_count):
                        self.estimation_handles['future_wedge_handles'][0].remove(
                        )
                        self.estimation_handles['future_wedge_handles'].pop(0)
                # add the future wedge patches
                for i in range(len(estimation['future_wedges'])):
                    next_wedge = estimation['future_wedges'][i]
                    self.estimation_handles['future_wedge_handles'].append(self.makeWedgePatch(
                        close_right=next_wedge['cr'], close_left=next_wedge['cl'], far_right=next_wedge[
                            'fr'], far_left=next_wedge['fl'], color=next_wedge['color']
                    ))

                for wedge_patch in self.estimation_handles['future_wedge_handles']:
                    self.ax[0].add_patch(wedge_patch)

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

    def drawPoint(self, coords, color="red"):
        self.ax[0].plot(coords.item(1), coords.item(0),
                        marker="o", markersize=3, color=color)

    def drawEllipse(self, center, major_length, minor_length, angle):
        ellipse = mpatches.Ellipse((center.item(1), center.item(
            0)), major_length, minor_length, angle=angle * 180 / np.pi, edgecolor='red', facecolor='none')
        self.ax[0].add_patch(ellipse)

    def drawLine(self, x_data, y_data, color="red"):
        self.ax[0].plot(x_data, y_data, linestyle='-', color=color)

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
