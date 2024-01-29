import numpy as np
from message_types.msg_state import MsgState
import parameters.simulation_parameters as SIM
from guidance.tools.wrap import wrap
from guidance.tools.dirty_derivative import DirtyDerivative
import matplotlib.cm as cm


class Guidance:
    def __init__(self, goal=np.array([[450, 450]])):
        # Initialize the goal destination
        self.goal = goal

        self.previous_bearing = 0
        self.size_derivative = DirtyDerivative(
            Ts=SIM.ts_simulation, tau=5 * SIM.ts_simulation)
        self.bearing_derivative = DirtyDerivative(
            Ts=SIM.ts_simulation, tau=5 * SIM.ts_simulation)

        self.min_wingspan = 80  # meters?
        self.max_velocity_dist = 450  # meters?
        # This is the uncertainty in our bearing measurement, in radians
        self.bearing_uncertainty = 0.1

        self.delta_x = 10  # in meters, or whatever the distance value is?
        self.ttc = -1  # in seconds, set to a negative value to start. Will get changed if there is a collision ahead of the ownship
        # A flag to run some setup for the first iteration of the collision estimator
        self.needsInitialize = True

        # Number of future predictions that are used
        self.future_wedge_count = 30

        # Variable to hold the current estimate of the collision point
        self.collision_point = np.array([[0], [0]])

    def update(self, ownship_state, bearings, sizes, time_delta):
        # set placeholders that will get filled in later
        u = 0
        collision_data = {'has_collision': False}

        # the avoiding is based on who is closest, determined by the pixel size;
        intruder_index = np.argmax(sizes)

        size = sizes[intruder_index]
        # bearing is positive when the intruder is to the ownship's left
        bearing = bearings[intruder_index]
        size_dot = self.size_derivative.update(z=size)
        bearing_dot = self.bearing_derivative.update(z=bearing)

        if size > .0005 and size_dot > .001:

            # store some useful variables
            theta = ownship_state.theta
            ownship_direction = np.array([[np.cos(theta)], [np.sin(theta)]])

            if self.needsInitialize:
                # There is no previous bearing, but I can still run the prediction for a 'straight ahead' collision
                # there is also no previous time to collision that I have computed
                # so compute one
                # TODO: actually determine the TTC using the correct equation
                ttc = 11  # in seconds
                self.ttc = ttc

                # determine the first collision point (without prior bearing or ttc data)
                point_from_ownship = self.ttc * ownship_state.vel * ownship_direction
                self.collision_point = ownship_state.pos + point_from_ownship
            else:
                # update the time to collision
                self.ttc = self.ttc - time_delta

                # determine the collision point (with prior bearing information)
                # the improved collision point should include an x-coordinate modification of the old collision point
                # but the y coordinate should stay the same
                # TODO: this needs updating to estimate the intruder collision point
                self.collision_point = self.collision_point + \
                    np.array([[0], [5]])

            # Find the centerpoint by going along the bearing vector, with the lower bound being the smallest wingspan
            # and the highest bound being the maximum velocity bound that an intruder can be travelling

            bearing_unit_direction = self.rotation_matrix(theta) @ np.array(
                [[np.cos(bearing)], [np.sin(bearing)]])
            rotated_bearing_unit = self.rotation_matrix(
                theta=np.pi / 2) @ bearing_unit_direction

            # find the corners of the wedge that are the closest to the ownship
            min_wingspan_point = ownship_state.pos + \
                bearing_unit_direction * self.min_wingspan

            close_point_right = min_wingspan_point + rotated_bearing_unit * \
                self.min_wingspan * np.tan(self.bearing_uncertainty)

            close_point_left = min_wingspan_point - rotated_bearing_unit * \
                self.min_wingspan * np.tan(self.bearing_uncertainty)

            # find the corners of the wedge that are the furthest from the ownship
            max_velocity_point = ownship_state.pos + \
                bearing_unit_direction * self.max_velocity_dist

            far_point_right = max_velocity_point + rotated_bearing_unit * \
                self.max_velocity_dist * np.tan(self.bearing_uncertainty)

            far_point_left = max_velocity_point - rotated_bearing_unit * \
                self.max_velocity_dist * np.tan(self.bearing_uncertainty)

            # find the "midpoint" of the wedge - the point that is the midpoint of the symmetry axis
            wedge_midpoint = ownship_state.pos + bearing_unit_direction * \
                ((self.min_wingspan + self.max_velocity_dist) / 2)
            # record the wedge angle
            wedge_angle = -bearing - theta

            # now, draw a line from the top of the wedge to the collision point and beyond
            # as well as a line from the bottom of the wedge to the collision point and beyond

            # top line through the collision point
            top_slope = self.find_slope(
                max_velocity_point, self.collision_point)
            top_y_intercept = self.find_y_intercept(
                top_slope, self.collision_point)

            # bottom line through the collision point
            bottom_slope = self.find_slope(
                min_wingspan_point, self.collision_point)
            bottom_y_intercept = self.find_y_intercept(
                bottom_slope, self.collision_point)

            # midpoint line through the collision point
            mid_slope = self.find_slope(wedge_midpoint, self.collision_point)
            mid_y_intercept = self.find_y_intercept(
                mid_slope, self.collision_point)

            # ownship line through the collision point
            own_slope = self.find_slope(
                ownship_state.pos, self.collision_point)
            own_y_intercept = self.find_y_intercept(
                own_slope, self.collision_point)

            future_wedges_collection = []
            cmap = cm.get_cmap('rainbow', self.future_wedge_count)
            # Everything in this section needs to be looped -------------------------------------------------------------------------
            for i in range(1, self.future_wedge_count):

                # now find the next point along the midpoint line, which will be the center of the next ellipse
                next_midpoint_x = wedge_midpoint.item(1) + self.delta_x * (i)
                next_midpoint_y = mid_slope * next_midpoint_x + mid_y_intercept
                next_midpoint = np.array(
                    [[next_midpoint_y], [next_midpoint_x]])

                # use the next midpoint to find a line that extends in the direction of the bearing angle (in the world frame)
                direction_step_x = next_midpoint_x + \
                    np.cos(wedge_angle + np.pi / 2)
                direction_step_y = next_midpoint_y + \
                    np.sin(wedge_angle + np.pi / 2)
                direction_step_point = np.array(
                    [[direction_step_y], [direction_step_x]])

                next_axis_slope = self.find_slope(
                    pt2=next_midpoint, pt1=direction_step_point)
                next_axis_y_intercept = self.find_y_intercept(
                    slope=next_axis_slope, pt=direction_step_point)

                # get the top and bottom points by finding where the two lines intersect
                top_intercept = self.find_intercept_point(
                    slope1=top_slope, y_intercept1=top_y_intercept, slope2=next_axis_slope, y_intercept2=next_axis_y_intercept)
                bottom_intercept = self.find_intercept_point(
                    slope1=bottom_slope, y_intercept1=bottom_y_intercept, slope2=next_axis_slope, y_intercept2=next_axis_y_intercept)

                own_line_intercept = self.find_intercept_point(
                    slope1=own_slope, y_intercept1=own_y_intercept, slope2=next_axis_slope, y_intercept2=next_axis_y_intercept)

                # find the corners of the wedge that are the closest to the ownship
                min_wingspan_dist = np.linalg.norm(
                    own_line_intercept - bottom_intercept)
                close_lateral_dist = rotated_bearing_unit * \
                    min_wingspan_dist * np.tan(self.bearing_uncertainty)

                next_close_point_right = bottom_intercept + close_lateral_dist
                next_close_point_left = bottom_intercept - close_lateral_dist

                # find the corners of the wedge that are the furthest from the ownship
                max_velocity_dist = np.linalg.norm(
                    own_line_intercept - top_intercept)
                far_lateral_dist = rotated_bearing_unit * \
                    max_velocity_dist * np.tan(self.bearing_uncertainty)

                next_far_point_right = top_intercept + far_lateral_dist
                next_far_point_left = top_intercept - far_lateral_dist

                # construct the dictionary of the four points and add it to the future wedges list
                wedge_points = {
                    'cr': next_close_point_right,
                    'cl': next_close_point_left,
                    'fr': next_far_point_right,
                    'fl': next_far_point_left,
                    'color': cmap(self.future_wedge_count - (i - 1))
                }
                future_wedges_collection.append(wedge_points)

            # -----------------------------------------------------------------------------------------------------------------------

            # These are just for vizualization, and not needed for the ellipse generation ***************************************
            # They are points on the top, mid, and bottom lines
            top_points_x = np.array([max_velocity_point.item(
                1), self.collision_point.item(1), self.collision_point.item(1) + 100])
            top_points_y = top_slope * top_points_x + top_y_intercept

            mid_points_x = np.array([wedge_midpoint.item(
                1), self.collision_point.item(1), self.collision_point.item(1) + 100])
            mid_points_y = mid_slope * mid_points_x + mid_y_intercept

            bottom_points_x = np.array([min_wingspan_point.item(
                1), self.collision_point.item(1), self.collision_point.item(1) + 100])
            bottom_points_y = bottom_slope * bottom_points_x + bottom_y_intercept
            # ****************************************************************************************************

            if self.needsInitialize:
                # Need to set the ttc on the class object so that I have it next time
                # Also need to set needsInitialize to false so that this doesn't run again
                self.needsInitialize = False

            collision_data = {
                'has_collision': True,
                'collision_point': self.collision_point,
                'max_intruder_point': max_velocity_point,
                'min_intruder_point': min_wingspan_point,
                'initial_wedge': {
                    'cr': close_point_right,
                    'cl': close_point_left,
                    'fr': far_point_right,
                    'fl': far_point_left,
                },
                'top_bound_line': [top_points_x, top_points_y],
                'bottom_bound_line': [bottom_points_x, bottom_points_y],
                'mid_line': [mid_points_x, mid_points_y],
                'future_wedges': future_wedges_collection
            }

            # at the end of the prediction, set the previous bearing to the current bearing measurement
            self.previous_bearing = bearing

        else:

            # when not avoiding, the heading rate should converge so that the ownship is pointing at the goal
            # u is the heading rate. Positive values are turning to the right, negative values are turning left
            u = self.point_to_goal(ownship_state)
            # u = 0

        return u, collision_data

    def find_slope(self, pt2, pt1):
        return (pt2.item(0) - pt1.item(0)) / (pt2.item(1) - pt1.item(1))

    def find_y_intercept(self, slope, pt):
        return -slope * pt.item(1) + pt.item(0)

    def find_intercept_point(self, slope1, y_intercept1, slope2, y_intercept2):
        x_intercept = (y_intercept2 - y_intercept1) / (slope1 - slope2)
        y_intercept = slope1 * x_intercept + y_intercept1
        return np.array([[y_intercept], [x_intercept]])

    def saturate(self, input, lower, upper):
        output = input
        if input < lower:
            output = lower
        elif input > upper:
            output = upper

        return output

    def rotation_matrix(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])

    def point_to_goal(self, ownship_state):
        own_to_goal = self.goal.T - ownship_state.pos
        goal_angle = np.arctan(own_to_goal[1, 0] / own_to_goal[0, 0])

        ownship_angle = ownship_state.theta
        turn_angle = wrap(goal_angle - ownship_angle)
        return np.arctan(turn_angle)
