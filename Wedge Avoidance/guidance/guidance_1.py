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

        # for proportional navigation
        self.nav_constant = 2
        # this is the id number of which regime the ownship is following
        # 0 - No regime
        # 1 - Avoid intruder right side
        # 2 - Avoid left side
        # 3 - Avoid Head-on
        # 4 - Avoid overtaken
        # 5 - Avoid overtaking
        self.regime = 0

        # set up the limit anlges of bearing classification
        self.ang_left_front = np.pi / 4.0
        self.ang_left_rear = 110.0 * np.pi / 180.0
        self.ang_right_front = -1 * self.ang_left_front
        self.ang_right_rear = -1 * self.ang_left_rear

    def update(self, ownship_state, bearings, sizes):
        u = 0
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

        if size > .009 and size_dot > .001:
            self.avoiding = True

            # if the intruder is too close, try to purely react and avoid
            if size > 0.15:
                print("Reacting")
                min_bearing_dot = np.maximum(bearing_dot, 0.1)
                u = self.nav_constant / min_bearing_dot

                if bearing > 140 * np.pi / 180.0 or bearing < -140 * np.pi / 180:
                    # intruder is behind
                    if size_dot > 0:
                        u = -1
                    else:
                        u = 0

                u = self.saturate(u, -1.2, 1.2)

            elif self.regime > 0:
                # if the ownship is already in a regime, continue that regime
                match self.regime:
                    case 1:
                        u = self.right_side_avoid(
                            size, bearing, size_dot, bearing_dot)
                    case 2:
                        u = self.left_side_avoid(
                            size, bearing, size_dot, bearing_dot)
                    case 3:
                        u = self.head_on_avoid(
                            size, bearing, size_dot, bearing_dot)
                    case 4:
                        u = self.overtaken_avoid(
                            size, bearing, size_dot, bearing_dot)
                    case 5:
                        u = self.overtaking_avoid(
                            size, bearing, size_dot, bearing_dot)
                    case _:
                        # if none match, print a warning
                        print("Unknown regime!", self.regime)

            else:
                # determine where the intruder is, so as to set which avoidance regime we will use

                # bearing is positive when the intruder is to the ownship's left
                if bearing < self.ang_left_front and bearing > 0 or bearing > self.ang_right_front and bearing < 0:
                    # intruder is in front
                    self.regime = 3
                    self.head_on_avoid(size, bearing, size_dot, bearing_dot)
                elif bearing > self.ang_left_front and bearing < self.ang_left_rear:
                    # intruder is on the left
                    self.regime = 2
                    self.left_side_avoid(size, bearing, size_dot, bearing_dot)
                elif bearing < self.ang_right_front and bearing > self.ang_right_rear:
                    # intruder is on the right
                    self.regime = 1
                    self.right_side_avoid(size, bearing, size_dot, bearing_dot)
                else:
                    # intruder is behind
                    self.regime = 4
                    self.overtaken_avoid(
                        size, bearing, size_dot, bearing_dot)
        else:
            self.avoiding = False

        if not self.avoiding:
            # when not avoiding, the heading rate should converge so that the ownship is pointing at the goal
            # u is the heading rate. Positive values are turning to the right, negative values are turning left
            u = self.point_to_goal(ownship_state)
        return u

    def saturate(self, input, lower, upper):
        output = input
        if input < lower:
            output = lower
        elif input > upper:
            output = upper

        return output

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
