import numpy as np
from message_types.msg_state import MsgState
import parameters.simulation_parameters as SIM
from guidance.tools.dirty_derivative import DirtyDerivative
import matplotlib.cm as cm
import matplotlib.pyplot as plt


class Estimation:
    def __init__(self, goal=np.array([[450, 450]]), time_delta=1, future_steps=10):
        # Initialize the goal destination
        self.goal = goal

        self.size_derivative = DirtyDerivative(
            Ts=SIM.ts_simulation, tau=5 * SIM.ts_simulation)
        self.bearing_derivative = DirtyDerivative(
            Ts=SIM.ts_simulation, tau=5 * SIM.ts_simulation)

        self.heading_derivative = DirtyDerivative(
            Ts=SIM.ts_simulation, tau=5 * SIM.ts_simulation)
        
        # TODO: Edit these to be accurate
        self.smallest_intruder_area = 200  # meters squared
        self.largest_intruder_area = 210  # meters squared

        # This is the uncertainty in our bearing measurement, in radians
        self.bearing_uncertainty = 0.2

        # Number of future predictions that are used
        # this is how many steps forward we are predicting
        self.future_timesteps = future_steps
        # this is the magnitude (in seconds) of each timestep
        self.timestep_magnitude = time_delta
        self.wedge_quantity = self.future_timesteps * self.timestep_magnitude

        # this is the amount of observations that are needed before the path is planned
        self.measurements_required = 30
        # this is a counter of how many measurements that the estimator has already obtained
        self.measurements_obtained = 0


    def update(self, ownship_state, bearings, sizes):
        # set placeholders that will get filled in later
        estimation_data = {'has_collision': False}

        # the avoiding is based on who is closest, determined by the pixel size;
        intruder_index = np.argmax(sizes)

        size = sizes[intruder_index]
        # bearing is positive when the intruder is to the ownship's left
        # not sure why I need to make it negative to get it to show up correctly on the world viewer...
        bearing = bearings[intruder_index]
        size_dot = self.size_derivative.update(z=size)
        bearing_dot = self.bearing_derivative.update(z=bearing)
        theta = ownship_state.theta
        omega = self.heading_derivative.update(z=theta)

        # the intruder is large enough to start planning and it is growing in size
        if size > .007 and self.measurements_obtained > self.measurements_required:
            

            # set bounds on the intruder's range
            # size = the size in pixels of the intruder
            intruder_min_range = self.smallest_intruder_area / (size * 100)
            intruder_max_range = self.largest_intruder_area / (size * 100)
            # tells us how much to scale the minimum range to get the maximum range

            los_body = np.array([[np.cos(bearing)], [np.sin(bearing)]])
            los_dot_body = np.array(
                [[-bearing_dot * np.sin(bearing)], [bearing_dot * np.cos(bearing)]])

            R_b_to_i = self.rotation_matrix(theta)
            los = R_b_to_i @ los_body
            los_dot = R_b_to_i @ los_dot_body + R_b_to_i @ self.skew(omega) @ los_body 

            # set up the ownship velocity vector
            heading_unit_vector = np.array([[np.cos(theta)], [np.sin(theta)]])
            ownship_velocity_vector = heading_unit_vector * ownship_state.vel

            velocity_evolution = los_dot - (size_dot / size) * los_dot
            close_velocity = ownship_velocity_vector + intruder_min_range * velocity_evolution
            far_velocity = ownship_velocity_vector + intruder_max_range * velocity_evolution


            # this is rotated into the inertial frame

            # **************************************************************************************************************
            # Here's where we create the bounding lines of the farthest possible intruder and the closest possible intruder
            #
            # We also record all of the (north, east) coordinates of each bounding line, as well as the ownship for
            # all of the steps within the future envelope
            #
            # **************************************************************************************************************

            # intruder close and far points
            close_intr_pos = ownship_state.pos + los * intruder_min_range
            far_intr_pos = ownship_state.pos + los * intruder_max_range


            close_pts_north = [close_intr_pos.item(0)]
            close_pts_east = [close_intr_pos.item(1)]

            far_pts_north = [far_intr_pos.item(0)]
            far_pts_east = [far_intr_pos.item(1)]

            ownship_pts_north = [ownship_state.pos.item(0)]
            ownship_pts_east = [ownship_state.pos.item(1)]

            # loop through to add multiple points to the arrays
            for i in range(1, self.future_timesteps):
                step_magnitude = self.timestep_magnitude * i

                close_intr_future_pos = close_intr_pos + \
                    close_velocity * step_magnitude
                close_pts_north.append(close_intr_future_pos.item(0))
                close_pts_east.append(close_intr_future_pos.item(1))

                far_intr_future_pos = far_intr_pos + far_velocity * step_magnitude
                far_pts_north.append(far_intr_future_pos.item(0))
                far_pts_east.append(far_intr_future_pos.item(1))

                ownship_future_pos = ownship_state.pos + \
                    ownship_state.vel * heading_unit_vector * step_magnitude
                ownship_pts_north.append(ownship_future_pos.item(0))
                ownship_pts_east.append(ownship_future_pos.item(1))
            

            # ******************************************************************************************************************
            # Now we construct the wedges with all that information
            #
            # **************************************************************************************************************

            initial_wedge = self.get_initial_wedge(
                bearing_direction=los, close_point=close_intr_pos, far_point=far_intr_pos, ownship_point=ownship_state.pos)

            close_bound_pts = self.zip_points(close_pts_north, close_pts_east)
            far_bound_pts = self.zip_points(far_pts_north, far_pts_east)
            ownship_pts = self.zip_points(ownship_pts_north, ownship_pts_east)

            future_wedges, future_timesteps = self.get_future_wedges(
                lower_bound_points=close_bound_pts, upper_bound_points=far_bound_pts, ownship_points=ownship_pts)

            # **************************************************************************************************************
            #
            # Now package it up for the return value
            #
            # **************************************************************************************************************
            estimation_data = {
                'has_collision': True,
                'closest_point': close_intr_pos,
                'farthest_point': far_intr_pos,
                'bottom_bound_line': [close_pts_north, close_pts_east],
                'top_bound_line': [far_pts_north, far_pts_east],
                'initial_wedge': initial_wedge,
                'future_data': {
                    'wedges':future_wedges,
                    'timestamps': future_timesteps
                    }
            }
        else:
            self.measurements_obtained += 1
        return estimation_data

    def zip_points(self, pts_north, pts_east):
        return np.column_stack((
            np.array([pts_north]).T, np.array([pts_east]).T))

    def get_initial_wedge(self, bearing_direction, close_point, far_point, ownship_point):

        # these are for the wedge points
        rotated_bearing_unit = self.rotation_matrix(
            theta=np.pi / 2) @ bearing_direction
        perp_factor = rotated_bearing_unit * \
            np.tan(self.bearing_uncertainty)

        # close wedge points

        min_intruder_dist = np.linalg.norm(ownship_point - close_point)
        close_right = close_point + min_intruder_dist * perp_factor
        close_left = close_point - min_intruder_dist * perp_factor

        # far wedge points
        max_intruder_dist = np.linalg.norm(ownship_point - far_point)
        far_right = far_point + max_intruder_dist * perp_factor
        far_left = far_point - max_intruder_dist * perp_factor
        return {
            'cr': close_right,
            'cl': close_left,
            'fr': far_right,
            'fl': far_left,
        }

    def get_future_wedges(self, lower_bound_points, upper_bound_points, ownship_points):
        wedge_collection = []
        wedge_timesteps = []

        cmap = cm.get_cmap('rainbow', self.future_timesteps)
        # Everything in this section needs to be looped -------------------------------------------------------------------------
        for i in range(1, self.future_timesteps):

            # find the 'future bearing unit direction', which is the bearing direction from the
            #  future position of the ownship to the future position of the intruder family

            # we need it in order to find the direction to extend the wedge points
            next_close_point = lower_bound_points[i]
            next_far_point = upper_bound_points[i]
            next_ownship_point = ownship_points[i]

            direction_full = next_far_point - next_close_point
            direction_unit = direction_full / np.linalg.norm(direction_full)
            rotated_direction = self.rotation_matrix(
                theta=np.pi / 2) @ direction_unit

            # TODO: the uncertainty needs to evolve as we predict forward in time...
            perp_factor = rotated_direction * \
                np.tan(self.bearing_uncertainty)

            # find the corners of the wedge that are the closest to the ownship
            min_wingspan_dist = np.linalg.norm(
                next_ownship_point - next_close_point)
            close_lateral_dist = min_wingspan_dist * perp_factor

            next_close_point_right = next_close_point + close_lateral_dist
            next_close_point_left = next_close_point - close_lateral_dist

            # find the corners of the wedge that are the furthest from the ownship
            max_velocity_dist = np.linalg.norm(
                next_ownship_point - next_far_point)

            far_lateral_dist = max_velocity_dist * perp_factor

            next_far_point_right = next_far_point + far_lateral_dist
            next_far_point_left = next_far_point - far_lateral_dist

            # construct the dictionary of the four points and add it to the future wedges list
            wedge_points = {
                'cr': np.array([next_close_point_right]).T,
                'cl': np.array([next_close_point_left]).T,
                'fr': np.array([next_far_point_right]).T,
                'fl': np.array([next_far_point_left]).T,
                'color': cmap(self.future_timesteps - (i - 1)),
            }
            wedge_collection.append(wedge_points)
            wedge_timesteps.append(i * self.timestep_magnitude)

        # -----------------------------------------------------------------------------------------------------------------------

        return wedge_collection, wedge_timesteps

    def rotation_matrix(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])

    def skew(self, omega):
        return np.array([[0, -omega], [omega, 0]])