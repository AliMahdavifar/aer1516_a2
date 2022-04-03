"""
Assignment #2 Template file
"""
import random
import math
import numpy as np

"""
Problem Statement
--------------------
Implement the planning algorithm called Rapidly-Exploring Random Trees* (RRT)
for the problem setup given by the rrt_star_dubins_PROMLEM class.

INSTRUCTIONS
--------------------
1. The only file to be submitted is this file rrt_star_planner.py. Your
   implementation can be tested by running rrt_star_dubins_PROBLEM.PY (check the 
   main function).
2. Read all class and function documentation in rrt_star_dubins_PROBLEM carefully.
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
   rrt_star_dubins_PROBLEM.map_area.

DO(s) and DONT(s)
-------------------
1. Do not rename the file rrt_star_planner.py for submission.
2. Do not change change the PLANNING function signature.
3. Do not import anything other than what is already imported in this file.
4. You can write more function in this file in order to reduce code repitition
   but these function can only be used inside the PLANNING function.
   (since only the planning function will be imported)
"""


def rrt_star_planner(rrt_star_dubins, display_map=False):
    """
        Execute RRT* planning using Dubins-style paths. Make sure to populate the node_list.

        Inputs
        -------------
        rrt_star_dubins  - (rrt_star_dubins_PROBLEM) Class conatining the planning
                      problem specification
        display_map - (boolean) flag for animation on or off (OPTIONAL)

        Outputs
        --------------
        (list of nodes) This must be a valid list of connected nodes that form
                        a path from start to goal node

        NOTE: In order for rrt_star_dubins.draw_graph function to work properly, it is important
        to populate rrt_star_dubins.nodes_list with all valid RRT nodes.
    """
    p = 10  # sampling rate for random node generation

    # LOOP for max iterations
    i = 0
    while i < rrt_star_dubins.max_iter:
        i += 1

        # Generate a random vehicle state (x, y, yaw)
        if random.randint(0, 100) > p:
            x = random.uniform(rrt_star_dubins.x_lim[0], rrt_star_dubins.x_lim[1])
            y = random.uniform(rrt_star_dubins.y_lim[0], rrt_star_dubins.y_lim[1])
            yaw = random.uniform(-math.pi, math.pi)
            random_node = rrt_star_dubins.Node(x, y, yaw)
        else:
            random_node = rrt_star_dubins.Node(rrt_star_dubins.goal.x, rrt_star_dubins.goal.y, rrt_star_dubins.goal.yaw)

        # Create new node from the valid random node and closest node
        new_node = create_new_node(rrt_star_dubins, random_node)

        # Check if the path between nearest node and random state has obstacle collision
        # Add the node to nodes_list if it is valid
        if rrt_star_dubins.check_collision(new_node):
            # Find best parent
            closest_nodes = find_closest_nodes(rrt_star_dubins, new_node)
            best_parent_node, min_cost = choose_cheapest_parent(rrt_star_dubins, new_node, closest_nodes)

            # Update new node with best parent
            if min_cost:
                new_node = rrt_star_dubins.propogate(best_parent_node, new_node)
                new_node.cost = min_cost
            else:
                new_node = None

            if new_node:
                rrt_star_dubins.node_list.append(new_node)
                update_tree(rrt_star_dubins, new_node, closest_nodes)

        # Draw current view of the map
        # PRESS ESCAPE TO EXIT
        if display_map:
            rrt_star_dubins.draw_graph()

        # Check if new_node is close to goal
        if new_node:
            last_index = find_last_index(rrt_star_dubins)
            if last_index:
                print("Iters:", i, ", number of nodes:", len(rrt_star_dubins.node_list))
                
                # Compile the list of nodes for path from start node to goal node
                final_path = [rrt_star_dubins.goal]
                node = rrt_star_dubins.node_list[last_index]
                while node.parent:
                    final_path.append(node)
                    node = node.parent
                final_path.append(rrt_star_dubins.start)

                return final_path[::-1]

    if i == rrt_star_dubins.max_iter:
        print('reached max iterations')

    # Return path, which is a list of nodes leading to the goal...
    return None


def create_new_node(rrt_star_dubins, random_node):
    """
    Create new node from the valid random node and closest node
    :param rrt_star_dubins: RRT* object
    :param random_node: a randomly generated node inside the map
    :return: index of closest node
    """
    node_list = []
    for node in rrt_star_dubins.node_list:
        node_list.append(math.hypot(node.x - random_node.x, node.y - random_node.y) ** 2)

    # Find an existing node nearest to the random vehicle state
    closest_index = node_list.index(min(node_list))
    closest_node = rrt_star_dubins.node_list[closest_index]

    return rrt_star_dubins.propogate(closest_node, random_node)


def find_closest_nodes(rrt_star_dubins, new_node):
    """
    Return all the nodes that are a maximum of certain radius away from new_node
    :param rrt_star_dubins: RRT* object
    :param new_node: newly created random node
    :return: list of closest nodes within a certain threshold
    """
    max_circle_dist = 50
    expansion_radius = 30
    dist_list = []
    closest_nodes = []

    # Set the search radius
    node_list_len = len(rrt_star_dubins.node_list) + 1
    radius = min(max_circle_dist * math.sqrt((math.log(node_list_len) / node_list_len)), expansion_radius)

    # Create list of distances for all nodes up to new_node

    for node in rrt_star_dubins.node_list:
        dist_list.append(math.hypot(node.x - new_node.x, node.y - new_node.y) ** 2)

    # Find all closest nodes with the circle
    for i in dist_list:
        if i <= radius ** 2:
            closest_nodes.append(dist_list.index(i))
    return closest_nodes


def choose_cheapest_parent(rrt_star_dubins, new_node, closest_nodes):
    """
    Choose the cheapest nodes within the circle around new node as its parent
    :param rrt_star_dubins: RRT* object
    :param new_node: newly generated random node
    :param closest_nodes: list of closest nodes to new_node within a circle
    :return: Parent node to new_node with cheapest cost
    """
    if not closest_nodes:
        print("New node has no neighbour within the chosen radius")
        return None

    inf_cost = float("inf")
    costs = []

    # Find the cost for all the closest indices
    for i in closest_nodes:
        closest_node = rrt_star_dubins.node_list[i]
        parent_candid = rrt_star_dubins.propogate(closest_node, new_node)
        if parent_candid and rrt_star_dubins.check_collision(parent_candid):
            costs.append(rrt_star_dubins.calc_new_cost(closest_node, new_node))
        else:  # Collision occurs
            costs.append(inf_cost)
    min_cost = min(costs)

    if min_cost == inf_cost:
        print("No path found!")
        return None, None

    # Find new parent
    best_parent_index = closest_nodes[costs.index(min_cost)]
    best_parent_node = rrt_star_dubins.node_list[best_parent_index]

    return best_parent_node, min_cost


def update_tree(rrt_star_dubins, new_node, closest_nodes):
    """
    Reset the parent of nodes in closest_nodes to new_node,
    if new_node is a cheaper node to travel from
    :param rrt_star_dubins: RRT* object
    :param new_node: newly generated random tree
    :param closest_nodes: list of nodes closest to new_node within a certain radius
    """
    for i in closest_nodes:
        # Find the closest node having traversed from new_node
        closest_node = rrt_star_dubins.node_list[i]
        updated_closest_node = rrt_star_dubins.propogate(new_node, closest_node)

        if not updated_closest_node:
            continue

        # If the cost is better and the node has no collision, update the closest node
        updated_closest_node.cost = rrt_star_dubins.calc_new_cost(new_node, closest_node)
        if rrt_star_dubins.check_collision(updated_closest_node) and \
                closest_node.cost > updated_closest_node.cost:
            closest_node.x = updated_closest_node.x
            closest_node.path_x = updated_closest_node.path_x

            closest_node.y = updated_closest_node.y
            closest_node.path_y = updated_closest_node.path_y

            # Update the cost in the node list now that new_node is the parent
            closest_node.parent = updated_closest_node.parent
            closest_node.cost = updated_closest_node.cost

            update_nodes_cost(rrt_star_dubins, new_node)


def update_nodes_cost(rrt_star_dubins, parent_node):
    """
    Update the cost in the tree's node list with a new parent node, for
    those nodes with having parent_node as their predecessor
    :param rrt_star_dubins: RRt* object
    :param parent_node: newly set parent node
    """
    for node in rrt_star_dubins.node_list:
        if node.parent != parent_node:
            continue
        node.cost = rrt_star_dubins.calc_new_cost(parent_node, node)
        update_nodes_cost(rrt_star_dubins, node)


def find_last_index(rrt_star_dubins):
    """
    Find the index for the goal node in the node_list
    :param rrt_star_dubins: RRT* object
    :return: index with minimum cost
    """
    goal_candidates = []
    min_cost = 99999
    min_index = -1

    # Add candidates for goal indices if they are within a certain threshold
    for (i, node) in enumerate(rrt_star_dubins.node_list):
        # Adaptively change distance threshold by getting closer to the goal node
        dist_to_goal = rrt_star_dubins.calc_dist_to_goal(node.x, node.y)
        dist_threshold = 0.5 * math.exp(dist_to_goal)
        angle_threshold = np.deg2rad(1.0)

        # Check with distance and angle thresholds
        if dist_to_goal <= dist_threshold and \
                abs(rrt_star_dubins.node_list[i].yaw - rrt_star_dubins.goal.yaw) <= angle_threshold:
            goal_candidates.append(i)

    if not goal_candidates:
        return None

    # Find the index for the node with the minimum cost
    for i in goal_candidates:
        if rrt_star_dubins.node_list[i].cost < min_cost:
            min_cost = rrt_star_dubins.node_list[i].cost
            min_index = i

    if min_cost < 99999:
        return min_index
    else:
        return None