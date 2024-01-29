import numpy as np
from message_types.msg_state import MsgState
import parameters.simulation_parameters as SIM
from guidance.tools.wrap import wrap
from guidance.tools.dirty_derivative import DirtyDerivative


class Guidance:
    def __init__(self, goal=np.array([[450, 450]])):
        # Initialize the goal destination
        self.goal = goal

        # set up a member variable to store the current flight regime
        self.avoiding = False
        # set up an array to keep track of intruder size change and bearing change for one intruder
        # TODO: change this so that it can handle any number of intruders
        self.size_derivative = DirtyDerivative(
            Ts=SIM.ts_simulation, tau=5 * SIM.ts_simulation)
        self.bearing_derivative = DirtyDerivative(
            Ts=SIM.ts_simulation, tau=5 * SIM.ts_simulation)

        self.min_wingspan = 60  # meters?
        self.max_detect_dist = 600  # meters?
        # These are just placeholder values.... Not sure what units they actually are
        self.bearing_uncertainty = 20

        self.delta_x = 50  # in meters, or whatever the distance value is

    def update(self, ownship_state, bearings, sizes):
        u = 0
        collision = {'has_collision': False,
                     'collision_point': np.array([0, 0])}
        # every update, we need to check to see if we are in an "avoiding" or a "cruising" state
        # "avoiding" is when we are actively maneuvering to get away from a collision
        # "cruising" is the opposite of avoiding, when we are flying toward the global goal.
        # because the "avoiding" state value is binary, there is no "cruising" variable, it is just when self.avoiding is False

        # the avoiding is based on who is closest, determined by the pixel size;
        # the intruder that is the closest will have the biggest pixel size.
        intruder_index = np.argmax(sizes)

        size = sizes[intruder_index]
        # bearing is positive when the intruder is to the ownship's left
        bearing = bearings[intruder_index]
        size_dot = self.size_derivative.update(z=size)
        bearing_dot = self.bearing_derivative.update(z=bearing)

        if size > .008 and size_dot > .001:

            ttc = 11  # in seconds
            theta = ownship_state.theta
            unit_direction = np.array([[np.cos(theta)], [np.sin(theta)]])
            point_from_ownship = ttc * ownship_state.vel * unit_direction
            collision_point = ownship_state.pos + point_from_ownship

            # construct an estimate ellipse, by passing [centerpoint_x, centerpoint_y, minor_axis_length, major_axis_length]
            # Find the centerpoint by going along the bearing vector, with the lower bound being the smallest wingspan
            # and the highest bound being the furthest distance that we can sense

            bearing_unit_direction = self.rotation_matrix(theta) @ np.array(
                [[np.cos(-bearing)], [np.sin(-bearing)]])

            major_axis_length = (self.max_detect_dist - self.min_wingspan)

            min_wingspan_point = ownship_state.pos + \
                bearing_unit_direction * self.min_wingspan
            max_detection_point = ownship_state.pos + \
                bearing_unit_direction * self.max_detect_dist
            ellipse_midpoint = ownship_state.pos + bearing_unit_direction * \
                ((self.min_wingspan + self.max_detect_dist) / 2)

            ellipse_angle = bearing - theta

            # now, draw a line from the top of the ellipse to the collision point and beyond
            # as well as a line from the bottom of the ellipse to the collision point and beyond

            # top line
            top_slope = self.find_slope(max_detection_point, collision_point)
            top_y_intercept = self.find_y_intercept(top_slope, collision_point)

            # bottom line
            bottom_slope = self.find_slope(min_wingspan_point, collision_point)
            bottom_y_intercept = self.find_y_intercept(
                bottom_slope, collision_point)

            # midpoint line
            mid_slope = self.find_slope(ellipse_midpoint, collision_point)
            mid_y_intercept = self.find_y_intercept(mid_slope, collision_point)

            # now find the next point along the midpoint line, which will be the center of the next ellipse
            next_midpoint_x = ellipse_midpoint.item(1) + self.delta_x
            next_midpoint_y = mid_slope * next_midpoint_x + mid_y_intercept
            next_midpoint = np.array([[next_midpoint_y], [next_midpoint_x]])

            # use the next midpoint to find a line that extends in the direction of the bearing angle (in the world frame)
            direction_step_x = next_midpoint_x + \
                np.cos(ellipse_angle + np.pi / 2)
            direction_step_y = next_midpoint_y + \
                np.sin(ellipse_angle + np.pi / 2)
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

            # These are just for vizualization, and not needed for the ellipse generation ------------------------
            top_points_x = np.array([max_detection_point.item(
                1), collision_point.item(1), collision_point.item(1) + 100])
            top_points_y = top_slope * top_points_x + top_y_intercept

            mid_points_x = np.array([ellipse_midpoint.item(
                1), collision_point.item(1), collision_point.item(1) + 100])
            mid_points_y = mid_slope * mid_points_x + mid_y_intercept

            bottom_points_x = np.array([min_wingspan_point.item(
                1), collision_point.item(1), collision_point.item(1) + 100])
            bottom_points_y = bottom_slope * bottom_points_x + bottom_y_intercept
            # ----------------------------------------------------------------------------------------------------

            collision = {'has_collision': True,
                         'collision_point': collision_point,
                         'intruder_estimate': [
                             ellipse_midpoint,
                             self.bearing_uncertainty * 2,
                             major_axis_length,
                             ellipse_angle,
                         ],
                         'testing_top_line': [top_points_x, top_points_y],
                         'testing_bottom_line': [bottom_points_x, bottom_points_y],
                         'testing_mid_line': [mid_points_x, mid_points_y],
                         'testing_next_ellipse': [
                             next_midpoint,
                             self.bearing_uncertainty * 2,
                             np.linalg.norm(top_intercept - bottom_intercept),
                             ellipse_angle,
                         ],
                         'next_mid': np.array([[next_midpoint_y], [next_midpoint_x]]),
                         'top_intercept': top_intercept,
                         'bottom_intercept': bottom_intercept,

                         }

        else:

            # when not avoiding, the heading rate should converge so that the ownship is pointing at the goal
            # u is the heading rate. Positive values are turning to the right, negative values are turning left
            u = self.point_to_goal(ownship_state)

        return u, collision

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
        own_to_goal = np.flip(self.goal).T - ownship_state.pos
        goal_angle = np.arctan(own_to_goal[1, 0] / own_to_goal[0, 0])

        ownship_angle = ownship_state.theta
        turn_angle = wrap(goal_angle - ownship_angle)
        return np.arctan(turn_angle)

    def right_side_avoid(self, size, bearing, size_dot, bearing_dot):
        if bearing < self.ang_left_front:
            return 1
        else:
            self.regime = 0
            self.avoiding = False
            return 0

    def left_side_avoid(self, size, bearing, size_dot, bearing_dot):
        if bearing > self.ang_right_rear:
            return -1.0
        else:
            self.regime = 0
            self.avoiding = False
            return 0

    def head_on_avoid(self, size, bearing, size_dot, bearing_dot):
        if bearing < self.ang_left_front + np.pi / 4:
            return 1
        elif bearing < np.pi:
            return 0
        else:
            self.regime = 0
            self.avoiding = False
            return 0

    def overtaken_avoid(self, size, bearing, size_dot, bearing_dot):
        print(bearing_dot)
        if size_dot > 0:
            # intruder is approaching
            if abs(bearing_dot) < 0.5 and bearing > -5 * np.pi / 4:
                return -1.0
            else:
                return 0
        else:
            # intruder is going away
            self.regime = 0
            self.avoiding = False
            return 0

    def overtaking_avoid(self, size, bearing, size_dot, bearing_dot):
        return 1

    # get the line of sight unit vector
    # los = np.array([[np.sin(bearing)], [np.cos(bearing)], [0]])
    # velocity = ownship_state.vel * \
    #     np.array([[np.sin(ownship_state.theta)],
    #               [np.cos(ownship_state.theta)], [0]])

    # # get the desired proportion of acceleration
    # # this is the magnitude of acceleration
    # alpha = -self.nav_constant * bearing_dot * ownship_state.vel

    # # compute the desired acceleration direction vector
    # v_cross_los_cross_v = np.cross(
    #     np.cross(velocity.T, los.T), velocity.T)
    # accel_des = alpha * v_cross_los_cross_v / \
    #     np.linalg.norm(v_cross_los_cross_v)

    # # turn desired acceleration into desired heading rate, which is the input to the dynamics
    # sign_accel_des = np.sign(np.cross(velocity.T, accel_des)[0, 2])
    # if alpha != 0:
    #     u = ownship_state.vel / alpha * sign_accel_des
    # else:
    #     u = 0.5 * sign_accel_des
