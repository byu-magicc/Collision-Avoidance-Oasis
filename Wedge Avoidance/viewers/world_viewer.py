import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import parameters.simulation_parameters as SIM
plt.ion()  # enable interactive drawing


class WorldViewer:
    def __init__(self, goal):
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

    def update(self, ownship, intruders, bearings, sizes):
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
            self.drawMAV(ownship.state, handle_id=0)
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
            xy=[[close_left.item(1), close_left.item(0)], [close_right.item(1), close_right.item(0)], [far_right.item(1), far_right.item(0)], [far_left.item(1), far_left.item(0)]], closed=True, edgecolor=color, facecolor='none')
        self.ax[0].add_patch(wedge)
