import numpy as np
import random
from message_types.msg_waypoints import MsgWaypoints
from planning.dubins_params import DubinsParameters


class RRTDubins:
    def __init__(self, sim_settings):
        # Set up the dubins path - get this from the mavsim?
        self.dubins_path = DubinsParameters()
        self.radius = sim_settings.turn_radius
        self.segment_length = 4 * self.radius  # standard length of path segments

        # Save the dimensions of the world
        self.world_width = sim_settings.world_size
        self.world_length = sim_settings.world_size
        self.end_pose = sim_settings.end_pose
        self.airspeed = sim_settings.ownship_velocity
        self.paths_to_find = 3
        self.safety_margin = 10
        self.tree = MsgWaypoints()

    def find_path(self, current_ownship, estimation, world_viewer ):
        # Poses for the planner are: [north, east, course_angle]
        # the start pose is determined from the state of the current ownship
        start_pose = np.array(
            [[current_ownship.state.pos.item(0)], [current_ownship.state.pos.item(1)], [current_ownship.state.theta]])
        end_pose = self.end_pose
        Va = self.airspeed

        radius = self.radius
        self.segment_length = 4 * radius
        # reset the tree to zero
        self.tree = MsgWaypoints()
        # add the start pose to the tree
        self.tree.add(ne=start_pose[0:2], airspeed=Va,
                      course=start_pose.item(2), timestamp=0)
        # check to see if start_pose connects directly to end_pose
        dist_start_to_end = distance(start_pose, end_pose)
        time_to_endpoint = dist_start_to_end / Va
        if dist_start_to_end <= self.segment_length and self.collision(start_pose, 0, end_pose, time_to_endpoint, estimation) is False:
            # if this case passes, then the start_pose can connect directly to the end_pose
            self.tree.add(ne=end_pose[0:2], airspeed=Va, course=end_pose.item(2), timestamp=time_to_endpoint,
                          cost=dist_start_to_end, parent=1, connect_to_goal=1)
        else:
            num_paths = 0  # this is the number of paths that connect to the goal
            while num_paths < 3: 
                is_successful_path = self.extendTree(
                    end_pose, estimation)
                num_paths = num_paths + is_successful_path
        # find path with minimum cost to end_node
        rough_waypoints = self.findMinimumPath()
        final_waypoints = self.smoothPath(
            rough_waypoints, Va , estimation)

        # -------------------------------------------------------------------------------------
        # Visualizaton
        # world_viewer.drawEstimation(estimation)
        # world_viewer.drawWaypoints(final_waypoints)
        # world_viewer.wait()
        # -------------------------------------------------------------------------------------
        return final_waypoints

    def extendTree(self, end_pose, intruder_data):
        Va = self.airspeed
        radius = self.radius
        tree = self.tree
        # extend tree by randomly selecting pose and extending tree toward that pose
        has_added_waypoint = False
        hit_end_goal = False
        while has_added_waypoint is False:
            random_pose = self.randomPose()
            # find leaf on tree that is closest to new_pose
            # by subracting the north and east coords from each position already in the tree
            # then seeing which result has the lowest norm
            tmp = tree.ne - np.tile(random_pose[0:2], (1, tree.num_waypoints))
            tmp1 = np.diag(tmp.T @ tmp)
            idx = np.argmin(tmp1)
            dist = np.sqrt(tmp1.item(idx))
            L = np.max([np.min([dist, self.segment_length]), 3 * radius])
            prev_timestamp = tree.timestamps.item(idx)
            next_timestamp = prev_timestamp + L / Va
            cost = tree.cost.item(idx) + L
            # get the point on the tree that is closest to the random point
            branch_point = column(tree.ne, idx)
            # this is the vector from the branch point to the new point
            tmp = random_pose[0:2] - branch_point

            new_ne = branch_point + L * (tmp / np.linalg.norm(tmp))
            new_chi = np.arctan2(new_ne.item(1) - tree.ne[1, idx],
                                 new_ne.item(0) - tree.ne[0, idx])
            new_pose = np.concatenate((new_ne, np.array([[new_chi]])), axis=0)
            tree_pose = np.concatenate(
                (column(tree.ne, idx), np.array([[tree.course.item(idx)]])), axis=0)

            hit_end_goal = False
            if self.collision(tree_pose, prev_timestamp, new_pose, next_timestamp, intruder_data) is False:
                tree.add(ne=new_pose[0:2], airspeed=Va, course=new_chi, timestamp=next_timestamp,
                         cost=cost, parent=idx)
                has_added_waypoint = True

                # check to see if the node that we just added connects directly to end_node
                dist_new_to_end = distance(new_pose, end_pose)
                new_to_end_timestep = dist_new_to_end / Va
                if dist_new_to_end >= 3 * radius \
                        and dist_new_to_end < self.segment_length \
                        and self.collision(new_pose, next_timestamp, end_pose, new_to_end_timestep, intruder_data) is False:
                    # mark node as connecting to end.
                    tree.connect_to_goal[-1] = 1
                    hit_end_goal = True

        return hit_end_goal

    def findMinimumPath(self):
        end_pose = self.end_pose
        # find the lowest cost path to the end node

        # find nodes that connect to end_node
        connecting_node_ids = []
        for i in range(self.tree.num_waypoints):
            if self.tree.connect_to_goal.item(i) == 1:
                connecting_node_ids.append(i)
        # find minimum cost last node
        idx = np.argmin(self.tree.cost[connecting_node_ids])
        # construct lowest cost path order
        last_node = connecting_node_ids[idx]
        path_node_ids = [last_node]  # last node that connects to end node
        # set the current node to look at the parent of the node that connects to the goal
        current_node = self.tree.parent.item(last_node)
        while current_node >= 1:
            # get the parent node id
            parent_id = int(current_node)
            # this inserts the id of the parent node into the front of the path
            path_node_ids.insert(0, parent_id)
            # now set the next node to investigate to the current node's parent node
            current_node = self.tree.parent.item(parent_id)
        # finish by putting in a zero at the beginning of the path
        path_node_ids.insert(0, 0)
        # construct waypoint path
        rough_waypoints = MsgWaypoints()
        for i in path_node_ids:
            rough_waypoints.add(ne=column(self.tree.ne, i),
                                airspeed=self.tree.airspeed.item(i),
                                course=self.tree.course.item(i),
                                )
        rough_waypoints.add(ne=end_pose[0:2],
                            airspeed=self.tree.airspeed[-1],
                            course=end_pose.item(2),
                            )
        return rough_waypoints

    def collision(self, start_pose, start_time, end_pose, end_time, estimation):

        # if there is no collision data from the estimator,
        # then the collision check should always return False,
        # meaning that there is no collision
        if not estimation['has_collision']:
            return False

        radius = self.radius

        self.dubins_path.compute_parameters(
            start_pose[0:2], start_pose.item(2), end_pose[0:2], end_pose.item(2), radius)
        points = self.dubins_path.compute_points()

        wedge_timesteps = estimation['future_data']['timestamps']
        wedges = estimation['future_data']['wedges']

        point_interval = (end_time - start_time) / points.shape[0]

        # set up the loop to check all the points in the current segment
        i = 0
        collision_flag = False
        while not collision_flag and i < points.shape[0]:

            # find the intruder wedge that corresponds with this timestep
            dubins_point_timestep = start_time + point_interval * i
            found_wedge = False
            current_wedge_index = 0
            while not found_wedge and current_wedge_index < len(wedge_timesteps) - 1:
                if dubins_point_timestep > wedge_timesteps[current_wedge_index]:
                    current_wedge_index += 1
                else:
                    found_wedge = True

            intruder_wedge = wedges[current_wedge_index] 

            # now test if the point at the current time step is within the wedge
            if has_intruder_collision(intruder_wedge, column(points.T, i), safety_margin=self.safety_margin):
                collision_flag = True

            # increment the index
            i += 1

        return collision_flag

    def smoothPath(self, waypoints, Va, intruder_data):
        radius = self.radius
        # smooth the waypoint path
        smooth = [0]  # add the first waypoint index
        ptr = 1
        while ptr <= waypoints.num_waypoints - 2:
            
            start_position = column(waypoints.ne, smooth[-1])
            start_pose = np.concatenate((start_position, np.array([[waypoints.course[smooth[-1]]]])),
                                        axis=0)
            start_timestep = waypoints.timestamps[smooth[-1]]

            end_position = column(waypoints.ne, ptr + 1)
            end_pose = np.concatenate(
                (end_position, np.array([[waypoints.course[ptr + 1]]])), axis=0)
            dist_start_to_end = np.linalg.norm(start_position - end_position)
            end_timestep = dist_start_to_end / Va

            if self.collision(start_pose, start_timestep, end_pose, end_timestep, intruder_data) is True \
                    and distance(start_pose, end_pose) >= 2 * radius:
                smooth.append(ptr)
            ptr += 1
        smooth.append(waypoints.num_waypoints - 1)
        # construct smooth waypoint path
        smooth_waypoints = MsgWaypoints()
        for i in smooth:
            smooth_waypoints.add(ne=column(waypoints.ne, i),
                                 airspeed=waypoints.airspeed.item(i),
                                 course=waypoints.course.item(i),
                                 )
        return smooth_waypoints

    def randomPose(self):
        # generate a random pose
        pn = random.randint(-self.world_length, self.world_length)
        pe = random.randint(-self.world_width, self.world_width)
        chi = 0
        pose = np.array([[pn], [pe], [chi]])
        return pose


def distance(start_pose, end_pose):
    # compute distance between start and end pose
    d = np.linalg.norm(start_pose[0:2] - end_pose[0:2])
    return d


def has_intruder_collision(intruder_wedge, point, safety_margin):
    # if any of the dot product tests return true, then the point is outside of the wedge
    # also test the minimum distance from the point to the wedge.

    # set up a minimum distance variable
    min_dist = 1000

    p1 = point - intruder_wedge['cl']
    s1 = intruder_wedge['cr'] - intruder_wedge['cl']
    if (p1).T @ rot_by_90(s1) > 0:
        # This is outside of close side
        # Now we need to check the minimum distance to the wedge line segment
        min_dist_1 = get_min_dist_to_line(point=p1, line_vector=s1)
        if min_dist_1 < min_dist:
            min_dist = min_dist_1


    p2 = point - intruder_wedge['cr']
    s2 = intruder_wedge['fr'] - intruder_wedge['cr']
    if (p2).T @ rot_by_90(s2) > 0:
        # This is outside of right side
        # Now we need to check the minimum distance to the wedge line segment
        min_dist_2 = get_min_dist_to_line(point=p2, line_vector=s2)
        if min_dist_2 < min_dist:
            min_dist = min_dist_2

    p3 = point - intruder_wedge['fr']
    s3 = intruder_wedge['fl'] - intruder_wedge['fr']
    if (p3).T @ rot_by_90(s3) > 0:
        # This is outside of far side
        # Now we need to check the minimum distance to the wedge line segment
        min_dist_3 = get_min_dist_to_line(point=p3, line_vector=s3)
        if min_dist_3 < min_dist:
            min_dist = min_dist_3


    p4 = point - intruder_wedge['fl']
    s4 = intruder_wedge['cl'] - intruder_wedge['fl']
    if (p4).T @ rot_by_90(s4) > 0:
        # This is outside of close side
        # Now we need to check the minimum distance to the wedge line segment
        min_dist_4 = get_min_dist_to_line(point=p4, line_vector=s4)
        if min_dist_4 < min_dist:
            min_dist = min_dist_4

    # now check the minimum distance the point has to the wedge
    if min_dist > safety_margin:
        # if the minimum distance to one of the wedge sides is greater than the safety margin,
        # then the chosen point is deemed safe, no collision
        return False
    
    # if we got here, that means the point is within the wedge
    return True

def mod(x):
    # force x to be between 0 and 2*pi
    while x < 0:
        x += 2 * np.pi
    while x > 2 * np.pi:
        x -= 2 * np.pi
    return x

def rot_by_90(vector):
    return np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)],
                              [np.sin(np.pi/2), np.cos(np.pi/2)]]) @ vector

def column(A, i):
    # extracts the ith column of A and return column vector
    tmp = A[:, i]
    col = tmp.reshape(A.shape[0], 1)
    return col

def get_min_dist_to_line(point, line_vector):
    norm_s = np.linalg.norm(line_vector)
    min_perp_dist = (line_vector.T @ point / norm_s).item(0)
    return np.linalg.norm(point - np.max((np.min((min_perp_dist, norm_s)), 0.)) * line_vector / norm_s)