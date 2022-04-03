"""
Assignment #2 Template file
"""
import random
import math
import numpy as np

"""
Problem Statement
--------------------
Implement the planning algorithm called Rapidly-Exploring Random Trees (RRT)
for the problem setup given by the RRT_DUBINS_PROMLEM class.

INSTRUCTIONS
--------------------
1. The only file to be submitted is this file rrt_planner.py. Your implementation
   can be tested by running RRT_DUBINS_PROBLEM.PY (check the main function).
2. Read all class and function documentation in RRT_DUBINS_PROBLEM carefully.
   There are plenty of helper function in the class to ease implementation.
3. Your solution must meet all the conditions specificed below.
4. Below are some do's and don'ts for this problem as well.

Conditions
-------------------
There are some conditions to be satisfied for an acceptable solution.
These may or may not be verified by the marking script.

1. The solution loop must not run for more that a certain number of random iterations
   (Specified by a class member called MAX_ITER). This is mainly a safety
   measure to avoid time-out-related issues and will be set generously.
2. The planning function must return a list of nodes that represent a collision-free path
   from start node to the goal node. The path states (path_x, path_y, path_yaw)
   specified by each node must define a Dubins-style path and traverse from node i-1 -> node i.
   (READ the documentation for the node class to understand the terminology)
3. The returned path should have the start node at index 0 and goal node at index -1,
   while the parent node for node i from the list should be node i-1 from the list, ie,
   the path should be a valid list of nodes.
   (READ the documentation of the node to understand the terminology)
4. The node locations must not lie outside the map boundaries specified by
   RRT_DUBINS_PROBLEM.map_area.

DO(s) and DONT(s)
-------------------
1. Do not rename the file rrt_planner.py for submission.
2. Do not change change the PLANNING function signature.
3. Do not import anything other than what is already imported in this file.
4. You can write more function in this file in order to reduce code repitition
   but these function can only be used inside the PLANNING function.
   (since only the planning function will be imported)
"""


def rrt_planner(rrt_dubins, display_map=False):
    """
        Execute RRT planning using Dubins-style paths. Make sure to populate the node_list.

        Inputs
        -------------
        rrt_dubins  - (RRT_DUBINS_PROBLEM) Class conatining the planning
                      problem specification
        display_map - (boolean) flag for animation on or off (OPTIONAL)

        Outputs
        --------------
        (list of nodes) This must be a valid list of connected nodes that form
                        a path from start to goal node

        NOTE: In order for rrt_dubins.draw_graph function to work properly, it is important
        to populate rrt_dubins.nodes_list with all valid RRT nodes.
    """
    p = 10  # sampling rate for random node generation
    
    # LOOP for max iterations
    i = 0
    while i < rrt_dubins.max_iter:
        i += 1

        # Generate a random vehicle state (x, y, yaw)
        if random.randint(0, 100) > p:
            x = random.uniform(rrt_dubins.x_lim[0], rrt_dubins.x_lim[1])
            y = random.uniform(rrt_dubins.y_lim[0], rrt_dubins.y_lim[1])
            yaw = random.uniform(-math.pi, math.pi)
            random_node = rrt_dubins.Node(x, y, yaw)
        else:
            random_node = rrt_dubins.Node(rrt_dubins.goal.x, rrt_dubins.goal.y, rrt_dubins.goal.yaw)

        # Create new node from the valid random node and closest node
        new_node = create_new_node(rrt_dubins, random_node)

        # Check if the path between nearest node and random state has obstacle collision
        # Add the node to nodes_list if it is valid
        if rrt_dubins.check_collision(new_node):
            rrt_dubins.node_list.append(new_node)  # Storing all valid nodes

        # Draw current view of the map
        # PRESS ESCAPE TO EXIT
        if display_map:
            rrt_dubins.draw_graph()

        # Check if new_node is close to goal
        if new_node:
            last_index = find_last_index(rrt_dubins)
            if last_index:
                print("Iters:", i, ", number of nodes:", len(rrt_dubins.node_list))

                # Compile the list of nodes for path from start node to goal node
                final_path = [rrt_dubins.goal]
                node = rrt_dubins.node_list[last_index]
                while node.parent:
                    final_path.append(node)
                    node = node.parent
                final_path.append(rrt_dubins.start)

                return final_path[::-1]

    if i == rrt_dubins.max_iter:
        print('reached max iterations')

    # Return path, which is a list of nodes leading to the goal...
    return None


def create_new_node(rrt_dubins, random_node):
    """
    Create new node from the valid random node and closest node
    :param rrt_dubins: RRT object
    :param random_node: a randomly generated node inside the map
    :return: index of closest node
    """
    node_list = []
    for node in rrt_dubins.node_list:
        node_list.append(math.hypot(node.x - random_node.x, node.y - random_node.y) ** 2)

    # Find an existing node nearest to the random vehicle state
    closest_index = node_list.index(min(node_list))
    closest_node = rrt_dubins.node_list[closest_index]

    return rrt_dubins.propogate(closest_node, random_node)


def find_last_index(rrt_dubins):
    """
    Find the index for the goal node in the node_list
    :param rrt_dubins: RRT object
    :return: index with minimum cost
    """
    goal_candidates = []
    min_cost = 99999
    min_index = -1

    # Add candidates for goal indices if they are within a certain threshold
    for (i, node) in enumerate(rrt_dubins.node_list):
        # Adaptively change distance threshold by getting closer to the goal node
        dist_to_goal = rrt_dubins.calc_dist_to_goal(node.x, node.y)
        dist_threshold = 0.5 * math.exp(dist_to_goal)
        angle_threshold = np.deg2rad(1.0)

        # Check with distance and angle thresholds
        if dist_to_goal <= dist_threshold and \
                abs(rrt_dubins.node_list[i].yaw - rrt_dubins.goal.yaw) <= angle_threshold:
            goal_candidates.append(i)

    if not goal_candidates:
        return None

    # Find the index for the node with the minimum cost
    for i in goal_candidates:
        if rrt_dubins.node_list[i].cost < min_cost:
            min_cost = rrt_dubins.node_list[i].cost
            min_index = i

    if min_cost < 99999:
        return min_index
    else:
        return None