import numpy as np
import matplotlib.pyplot as plt
from helper_new import *

def generate_hexagonal_nodes(rows, cols, rod_length, precision=8):
    """
    Generate hexagonal grid nodes for a given number of rows and columns of hexagons.

    Args:
        rows (int): Number of hexagonal rows.
        cols (int): Number of hexagonal columns.
        rod_length (float): Length of each hexagon edge.
        precision (int): Decimal precision for rounding coordinates.

    Returns:
        nodes (dict): A dictionary mapping global node index to (x, y) coordinates.
    """
    nodes = {}  # Map global node index -> (x, y)
    unique_coords = set()  # Use a set to store unique rounded coordinates
    node_index = 0  # Global node index counter

    for i in range(rows):
        for j in range(cols):
            # Compute center position of the hexagon
            x = j * rod_length * 3 / 2
            y = i * rod_length * np.sqrt(3)
            if j % 2 == 1:  # Offset odd columns
                if i == (rows - 1):  # Skip invalid nodes in the last row for odd columns
                    continue
                else:
                    y += rod_length * np.sqrt(3) / 2

            # Generate six local nodes for the current hexagon
            for k in range(6):
                angle = np.pi / 3 * k
                node_x = x + rod_length * np.cos(angle)
                node_y = y + rod_length * np.sin(angle)

                # Round coordinates to avoid duplicates
                rounded_coord = (round(node_x, precision), round(node_y, precision))

                # Only add the node if it hasn't been added yet
                if rounded_coord not in unique_coords:
                    unique_coords.add(rounded_coord)
                    nodes[node_index] = rounded_coord
                    node_index += 1

    return nodes


def reassign_node_labels(nodes, cols):
    """
    Reassign node labels based on sorted coordinates (y first, then x).

    Args:
        nodes (dict): A dictionary of global node index -> (x, y) coordinates.

    Returns:
        sorted_nodes (dict): A dictionary of new labels -> (x, y) coordinates.
    """
    # Sort nodes by y coordinate first, then x coordinate
    sorted_coords = sorted(nodes.values(), key=lambda coord: (coord[1], coord[0]))

    # Create a new dictionary with row-column-based labels
    sorted_nodes = {}
    for idx, (x, y) in enumerate(sorted_coords):
        row = idx // (cols+1) + 1  # Compute row number (1-based)
        col = idx % (cols+1) + 1   # Compute column number (1-based)
        label = (row, col)  # New label as a tuple
        sorted_nodes[label] = (x, y)

    return sorted_nodes


def generate_rods(sorted_nodes, rows, cols):
    """
    Generate rods (edges) based on the given connection rules.

    Args:
        sorted_nodes (dict): A dictionary of (row, col) labels -> (x, y) coordinates.
        rows (int): Number of rows.
        cols (int): Number of columns.

    Returns:
        rods (list): A list of tuples representing rods (edges) between nodes.
    """
    rods = []
    nodes_dict = {node:idx for idx, node in enumerate(sorted_nodes)}

    # Generate rods
    for node in nodes_dict:
        if node[0] < (2*rows + 1):
            node1 = (node[0]+1, node[1])
            rods.append((nodes_dict[node], nodes_dict[node1]))
        if node[0] % 2 == node[1] % 2 and node[1] < (cols + 1):
            node2 = (node[0], node[1]+1)
            rods.append((nodes_dict[node], nodes_dict[node2]))

    return rods

def generate_flexible_nodes(rods, structure_nodes, nv_rod):
    """
    Generate flexible simulation nodes by dividing each rod into segments with uniform spacing.

    Args:
        rods (list): List of rods, where each rod is a tuple of two node labels (e.g., (node1, node2)).
        sorted_nodes (dict): Dictionary of node labels -> (x, y) coordinates.
        deltaL (float): Uniform spacing for dividing each rod.

    Returns:
        np.ndarray: Flattened array of all flexible nodes' coordinates in the order [x0, y0, x1, y1, ...].
    """
    flexible_nodes = []  # List to collect all nodes

    for rod in rods:
        # Get the start and end points of the rod
        node1, node2 = rod
        x1, y1 = structure_nodes[node1]
        x2, y2 = structure_nodes[node2]

        # Calculate the distance between the two nodes
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        num_segments = nv_rod + 1  # Number of segments along the rod

        # Generate intermediate nodes along the rod
        for i in range(1, num_segments):  # Not include endpoints
            t = i / num_segments  # Linear interpolation parameter (0 to 1)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            flexible_nodes.append((x, y))  # Collect the node's coordinates

    return np.array(flexible_nodes)


def calculate_curvature(connections, nodes):
    """
    Calculate the initial curvature (curvature0) for each main node in the honeycomb structure.

    Args:
        connections (dict): Dictionary where keys are main node indices, and values are lists of connected nodes.
        nodes (dict): Dictionary of node indices -> (x, y) coordinates.

    Returns:
        curvature_dict (dict): Dictionary of curvature values for each set of three points.
    """
    curvature_dict = {}

    for main_node, neighbors in connections.items():
        # Ensure there are at least two neighbors to form a triangle
        if len(neighbors) < 2:
            continue

        # Sort neighbors to ensure consistency in order
        sorted_neighbors = sorted(neighbors)

        # Generate all combinations of the main node and two neighbors
        for i in range(len(sorted_neighbors)):
            for j in range(i + 1, len(sorted_neighbors)):
                # Three points: main_node, neighbor1, neighbor2
                neighbor1 = sorted_neighbors[i]
                neighbor2 = sorted_neighbors[j]
                point1, point2, point3 = nodes[main_node], nodes[neighbor1], nodes[neighbor2]

                # Calculate vectors
                vec1 = np.array(point1) - np.array(point2)
                vec2 = np.array(point1) - np.array(point3)

                # Calculate the angle between vec1 and vec2
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Clip to avoid numerical issues

                # Calculate curvature0: 2 * tan((180° - angle) / 2) in radians
                curvature = 2 * np.tan((np.pi - angle) / 2)

                # Store in dictionary with main_node as the first element
                key = (main_node, neighbor1, neighbor2)
                curvature_dict[key] = curvature

    return curvature_dict

# Parameters
rows, cols, rod_length = 3, 3, 0.5  # Example with 4 rows and 5 columns

# Generate nodes
structure_nodes = generate_hexagonal_nodes(rows, cols, rod_length)

# Reassign node labels
sorted_structure_nodes = reassign_node_labels(structure_nodes, cols)
structure_nodes = {idx: coord for idx, coord in enumerate(sorted_structure_nodes.values())}

# Generate rods based on the given connection rules
rods = generate_rods(sorted_structure_nodes, rows, cols)
rods_dict = {idx: rod for idx, rod in enumerate(rods)}
rods_idx = {rod: idx for idx, rod in enumerate(rods)}

connection = {node:set() for node in structure_nodes} #Build a connection dictionary showing all nodes connected to each node using integer labels.
for node1, node2 in rods:
    connection[node1].add(node2)  # Add node2 to node1's connection list
    connection[node2].add(node1)  # Add node1 to node2's connection list

# Calculate global_curvature
curvature_dict = calculate_curvature(connection, structure_nodes)
for nodes in curvature_dict:
    if nodes[2]-nodes[1]==8 and (nodes[0]//(cols+1))%2 + nodes[0]%2==1:
        curvature_dict[nodes] = -curvature_dict[nodes]
    elif nodes[2]-nodes[1]!=8 and ((nodes[0]//(cols+1))+ nodes[0])%2==0:
        curvature_dict[nodes] = -curvature_dict[nodes]


# Define variables and initial conditions
npoint = (2 * rows + 1) * (cols + 1) # number of structure node points
nrod = len(rods) # number of rods
# Number of nodes for each rod
nv_rod = 8
nv = nv_rod * nrod + npoint
nv_top = (cols + 1) * (nv_rod + 2) / 2
ndof = 2 * nv # number of DOFs

flexible_nodes = generate_flexible_nodes(rods, structure_nodes, nv_rod)
structure_nodes_array = np.array(list(structure_nodes.values()))


# Plot original nodes and rods
plt.figure(figsize=(8, 8))
for rod in rods:
    start, end = rod
    x_values = [structure_nodes[start][0], structure_nodes[end][0]]
    y_values = [structure_nodes[start][1], structure_nodes[end][1]]
    plt.plot(x_values, y_values, 'b-', lw=1)  # Plot rods in blue

for label, coord in sorted_structure_nodes.items():
    x, y = coord
    plt.scatter(x, y, color='red', s=50, label="Original Nodes" if label == (1, 1) else None)
    plt.text(x, y, f"{label}", fontsize=8, ha='center', va='center')

# Plot flexible nodes
flexible_x = flexible_nodes[:, 0]
flexible_y = flexible_nodes[:, 1]
plt.scatter(flexible_x, flexible_y, color='green', s=20, label='Flexible Nodes')  # Flexible nodes in green

# Customize plot
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Original and Flexible Nodes Distribution')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()



all_nodes = np.concatenate((structure_nodes_array, flexible_nodes))
# Flatten the list of nodes into [x0, y0, x1, y1, ...] format
q0 = np.array(all_nodes).flatten()

# Time step
dt = 1e-4 # trial and error may be needed

# # Rod Length
# RodLength = 0.10 # meter

# Discrete Length
deltaL = rod_length / (nv_rod + 1)

# Radii of spheres
R = np.zeros(nv)
R[:] = 0.01 #deltaL / 10
# midNode = int ((nv+1)/2)
# R[midNode-1] = 0.025 # meter

# Densities
rho_metal = 7000 # kg/m^3
rho_air = 1 # kg/m^3
rho = rho_metal - rho_air

# Total mass
mass_total = 20 # kg

# Cross-sectional radius of rod
r0 = (mass_total / (nrod* np.pi * rod_length * rho_metal))**0.5 # meter

# Young's modulus
Y = 3e11 # Pascals

# Viscosity
visc = 1 # Pa-s

# Maximum number of iterations
maximum_iter = 200

# Total time
totalTime = 3 # seconds

# Utility variables
ne = (nv_rod + 1) * nrod # Number of edges
EI = Y * np.pi * r0 ** 4 / 4 # Bending stiffness (Nm^2)
EA = Y * np.pi * r0 ** 2 # Stretching stiffness (N)

# tolerance on force
tol = EI / rod_length** 2 * 1e-4 # small enough force

# Computer Mass
m = np.zeros(ndof) # 2*nv = ndof
for k in range(nv):
  m[2*k] = 4.0/3.0 * rho_metal * np.pi * R[k] ** 3.0 # mass for x_k
  m[2*k+1] = m[2*k] # mass for y_k
mMat = np.diag(m)

# External Force and Gravity
Q = -2000 # N total load
Q_point = Q / nv_top
g = np.array([0, -9.8]) # m/s^2
W = np.zeros(ndof)
for k in range(nv):
  W[2*k] = 4.0/3.0 * np.pi * R[k]**3 * rho * g[0] # Weight for x_k
  W[2*k+1] = 4.0/3.0 * np.pi * R[k]**3 * rho * g[1] # Weight for y_k

# Apply loads to the top corners
W[2*(npoint-cols)-1] += Q
W[2*npoint-1] += Q

# # External Force (viscous damping): define the viscous damping matrix, C
# C = np.zeros((ndof, ndof))
# for k in range(nv):
#   C[2*k, 2*k] = 6 * np.pi * visc * R[k]
#   C[2*k+1, 2*k+1] = C[2*k, 2*k]

# # Initial conditions (positions and velocities)
# q0 = np.zeros(2*nv)
# for c in range(nv):
#   q0[2*c] = nodes[c, 0] # x coord of c-th node
#   q0[2*c+1] = nodes[c, 1] # y coord of c-th node

q = q0.copy()
u = (q - q0) / dt # all zeros

all_DOFs = np.arange(ndof)
# fixed_index = np.array([0, 1, ndof-1]) # If you need to add more fixed DOFs, just add them here. Be mindful of Python/MATLAB's indexing convention
fixed_index = all_DOFs[0:2*(cols+1)]
# free_index is the difference between all_DOFs and fixed_index
free_index = np.setdiff1d(all_DOFs, fixed_index)

# PART 2
# Time stepping scheme
#
Nsteps = round(totalTime / dt)+2
ctime = 0 # Current time

# Some arrays to store the results (not mandatory)
all_pos = np.zeros(Nsteps) # y coordinate of middle node
all_v = np.zeros(Nsteps) # y velocity of middle node
midAngle = np.zeros(Nsteps) # angle/bent shape at middle node (radians)

plotStep = 10 # Plot every 50 time steps
debugstep = 50

for timeStep in range(1, Nsteps):
    # print('t = %f' % ctime)

    q, flag = objfun(q0, q0, u, dt, tol, maximum_iter, m, mMat,  # inertia
                     EI, EA, # elastic stiffness
                     W, # external force
                     deltaL, npoint, nv_rod,
                     rods, structure_nodes, connection, curvature_dict,
                     free_index)

    if flag < 0:
        print('Could not converge')
        break

    u = (q - q0) / dt # New velocity

    # Update old position
    q0 = q.copy() # New position

    # nodes_pos = q0.reshape(nv//4, 4, 2)  # Reshape to a 2D array (n, 2)

    if (timeStep - 1) % debugstep == 0:
        # print(nodes_pos)
        x_arr = q[::2]  # x coordinates
        y_arr = q[1::2]  # y coordinates
        plt.clf()
        for idx, (x_arr, y_arr) in enumerate(zip(x_arr, y_arr)):
            plt.text(x_arr, y_arr, f'{idx}', color='red', fontsize=12, ha='center', va='center')

    # Plotting
    if (timeStep-1) % plotStep == 0:
        x_arr = q[::2]  # x coordinates
        y_arr = q[1::2]  # y coordinates
        plt.clf()
        # for idx, (x_arr, y_arr) in enumerate(zip(x_arr, y_arr)):
        #     plt.text(x_arr, y_arr, f'{idx}', color='red', fontsize=12, ha='center', va='center')
        plt.scatter(x_arr, y_arr, color='green', s=20)

        for rod in rods_dict:
            start, end = rods_dict[rod]
            x_values = ([q0[2 * start]] +
                      list(q0[2 * npoint + 2 * rod * nv_rod: 2 * npoint + 2 * (rod + 1) * nv_rod:2]) +
                      [q0[2 * end]])
            y_values = ([q0[2 * start + 1]] +
                      list(q0[2 * npoint + 2 * rod * nv_rod + 1: 2 * npoint + 2 * (rod + 1) * nv_rod + 1:2]) +
                      [q0[2 * end + 1]])
            plt.plot(x_values, y_values, 'b-', lw=1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.grid(True)
        plt.title('t=%f' % ctime)
        plt.show()
    print(ctime)
    ctime += dt  # Update time