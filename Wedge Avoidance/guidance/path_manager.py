from message_types.msg_path import MsgPath
from planning.dubins_params import DubinsParameters
import numpy as np


class PathManager:
    def __init__(self, sim_settings):
        # message sent to path follower
        self.path = MsgPath()
        # pointers to previous, current, and next waypoints
        self.ptr_previous = 0
        self.ptr_current = 1
        self.ptr_next = 2
        self.num_waypoints = 0
        self.halfspace_n = np.inf * np.ones((3, 1))
        self.halfspace_r = np.inf * np.ones((3, 1))
        # state of the manager state machine
        self.manager_state = 1
        self.manager_requests_waypoints = True
        self.dubins_path = DubinsParameters()
        self.radius = sim_settings.turn_radius


    def update(self, waypoints, state):
        if waypoints.num_waypoints == 0:
            self.manager_requests_waypoints = True
            self.construct_straight_line(state)
            return self.path
        
        if self.manager_requests_waypoints is True \
                and waypoints.flag_waypoints_changed is True:
            self.manager_requests_waypoints = False

        # this modifies the self.path object
        self.dubins_manager(waypoints, state)

        return self.path

    def dubins_manager(self, waypoints, state):
        radius = self.radius
        mav_pos = np.array([[state.pos.item(0), state.pos.item(1)]]).T
        close_distance = 10
        # if the waypoints have changed, update the waypoint pointer
        if waypoints.flag_waypoints_changed is True:
            waypoints.flag_waypoints_changed = False
            self.num_waypoints = waypoints.num_waypoints
            self.initialize_pointers()
            # dubins path parameters
            self.dubins_path.compute_parameters(
                ps=waypoints.ne[:, self.ptr_previous:self.ptr_previous+1],
                chis=waypoints.course.item(self.ptr_previous),
                pe=waypoints.ne[:, self.ptr_current:self.ptr_current+1],
                chie=waypoints.course.item(self.ptr_current),
                R=radius)
            self.construct_dubins_circle_start(waypoints)
            if np.linalg.norm(self.dubins_path.p_s - self.dubins_path.r1) < close_distance:
                self.construct_dubins_line(waypoints)
                self.manager_state = 3
            elif self.inHalfSpace(mav_pos):
                self.manager_state = 1
            else:
                self.manager_state = 2
        # state machine for dubins path
        if self.manager_state == 1:
            # skip the first circle if distance along circle is small
            if ((np.linalg.norm(self.dubins_path.p_s - self.dubins_path.r1) < close_distance)
                    # follow start orbit until out of H1
                    or not self.inHalfSpace(mav_pos)):
                self.manager_state = 2
        elif self.manager_state == 2:
            # skip the first circle if distance along circle is small
            if ((np.linalg.norm(self.dubins_path.p_s - self.dubins_path.r1) < close_distance)
                    # follow start orbit until cross into H1
                    or self.inHalfSpace(mav_pos)):
                self.construct_dubins_line(waypoints)
                self.manager_state = 3
        elif self.manager_state == 3:
            # skip line if it is short
            if ((np.linalg.norm(self.dubins_path.r1 - self.dubins_path.r2) < close_distance)
                    or self.inHalfSpace(mav_pos)):
                self.construct_dubins_circle_end(waypoints)
                if self.inHalfSpace(mav_pos):
                    self.manager_state = 4
                else:
                    self.manager_state = 5
        elif self.manager_state == 4:
            # distance along end orbit is small
            if ((np.linalg.norm(self.dubins_path.r2 - self.dubins_path.p_e) < close_distance)
                    # follow start orbit until out of H3
                    or not self.inHalfSpace(mav_pos)):
                self.manager_state = 5
        elif self.manager_state == 5:
            # skip circle if small
            if ((np.linalg.norm(self.dubins_path.r2 - self.dubins_path.p_e) < close_distance)
                    # follow start orbit until cross into H3
                    or self.inHalfSpace(mav_pos)):
                self.increment_pointers()
                self.dubins_path.compute_parameters(
                    waypoints.ne[:, self.ptr_previous:self.ptr_previous+1],
                    waypoints.course.item(self.ptr_previous),
                    waypoints.ne[:, self.ptr_current:self.ptr_current+1],
                    waypoints.course.item(self.ptr_current),
                    radius)
                self.construct_dubins_circle_start(waypoints)
                self.manager_state = 1
                # requests new waypoints when reach end of current list
                if self.ptr_current == 0:
                    self.manager_requests_waypoints = True

    def initialize_pointers(self):
        if self.num_waypoints >= 3:
            self.ptr_previous = 0
            self.ptr_current = 1
            self.ptr_next = 2
        else:
            print('Error Path Manager: need at least three waypoints')

    def increment_pointers(self):
        self.ptr_previous = self.ptr_current
        self.ptr_current = self.ptr_next
        self.ptr_next = self.ptr_next + 1
        if self.ptr_next > self.num_waypoints-1:
            # self.ptr_next = 9999
            self.ptr_next = 0
        if self.ptr_current > self.num_waypoints-1:
            # self.ptr_current = 9999
            self.ptr_current = 0

    def construct_dubins_circle_start(self, waypoints):
        dubins_path = self.dubins_path

        self.path.type = 'orbit'
        self.path.airspeed = waypoints.airspeed.item(self.ptr_current)
        self.path.orbit_radius = dubins_path.radius
        self.path.orbit_center = dubins_path.center_s
        if dubins_path.dir_s == 1:
            self.path.orbit_direction = 'CW'
        else:
            self.path.orbit_direction = 'CCW'
        self.halfspace_n = dubins_path.n1
        self.halfspace_r = dubins_path.r1
        self.path.plot_updated = False

    def construct_dubins_line(self, waypoints):
        dubins_path = self.dubins_path
        self.path.type = 'line'
        self.path.airspeed = waypoints.airspeed.item(self.ptr_current)
        self.path.line_origin = dubins_path.r1
        self.path.line_direction = dubins_path.n1
        self.halfspace_n = dubins_path.n1
        self.halfspace_r = dubins_path.r2
        self.path.plot_updated = False


    def construct_dubins_circle_end(self, waypoints):

        dubins_path = self.dubins_path
        self.path.type = 'orbit'
        self.path.airspeed = waypoints.airspeed.item(self.ptr_current)
        self.path.orbit_radius = dubins_path.radius
        self.path.orbit_center = dubins_path.center_e
        if dubins_path.dir_e == 1:
            self.path.orbit_direction = 'CW'
        else:
            self.path.orbit_direction = 'CCW'
        self.halfspace_n = dubins_path.n3
        self.halfspace_r = dubins_path.r3
        self.path.plot_updated = False

    def construct_straight_line(self, state):
        self.path.type = 'line'
        self.path.airspeed = state.vel
        self.path.line_origin = state.pos
        self.path.line_direction = np.array([[1],[0] ])

    def inHalfSpace(self, pos):
        if (pos-self.halfspace_r).T @ self.halfspace_n >= 0:
            return True
        else:
            return False
