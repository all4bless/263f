import numpy as np
import matplotlib.pyplot as plt
from helper_new import *
import subprocess
import os

# Parameters
rows, cols, rod_length = 5, 5, (1.5/5)  # Example with 4 rows and 5 columns

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
    if nodes[2]-nodes[1]==2*(cols+1) and ((nodes[0]//(cols+1)) + nodes[0])%2==1:
        curvature_dict[nodes] = -curvature_dict[nodes]
    elif nodes[2]-nodes[1]!=2*(cols+1) and ((nodes[0]//(cols+1))+ nodes[0])%2==0:
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

all_nodes = np.concatenate((structure_nodes_array, flexible_nodes))
# Flatten the list of nodes into [x0, y0, x1, y1, ...] format
q0 = np.array(all_nodes).flatten()


# Plot original nodes and rods
plt.figure(figsize=(8, 8))

# Plot rods as blue lines
for rod in rods:
    start, end = rod
    x_values = [structure_nodes[start][0], structure_nodes[end][0]]
    y_values = [structure_nodes[start][1], structure_nodes[end][1]]
    plt.plot(x_values, y_values, 'b-', lw=1)

# Plot structural nodes as red dots
for label, coord in sorted_structure_nodes.items():
    x, y = coord
    plt.scatter(x, y, color='red', s=50, label="Original Nodes" if label == (1, 1) else None)

# Plot flexible nodes as green dots
flexible_x = flexible_nodes[:, 0]
flexible_y = flexible_nodes[:, 1]
plt.scatter(flexible_x, flexible_y, color='green', s=20, label='Flexible Nodes')

# Add loading arrows (green downward arrows)
arrow_y = max(q0[1::2]) + 0.4  # Arrow starting y-coordinate
for x in q0[2*(npoint-cols-1) : 2*npoint : 2]:  # Arrow x-coordinates
    plt.arrow(x, arrow_y, 0, -0.2, head_width=0.1, head_length=0.1,
             fc='green', ec='green')

# Add boundary conditions (blue triangles)
triangle_y = min(q0[1::2]) - 0.1
for x in q0[0 : 2*(cols+1) : 2]:  # Triangle x-coordinates
    plt.plot([x-0.06, x+0.06, x, x-0.06],
            [triangle_y, triangle_y, triangle_y+0.1, triangle_y], 'b-')

# Customize plot appearance
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Model Setup')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()


# Time step
dt = 1e-3 # trial and error may be needed

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
totalTime = 1 # seconds

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
Q_t = -20000 # N total load
Q_point = Q_t / (cols+1)
g = np.array([0, -9.8]) # m/s^2
W = np.zeros(ndof)
Q = np.zeros((int(totalTime/dt), ndof))
for k in range(nv):
  W[2*k] = 4.0/3.0 * np.pi * R[k]**3 * rho * g[0] # Weight for x_k
  W[2*k+1] = 4.0/3.0 * np.pi * R[k]**3 * rho * g[1] # Weight for y_k

# Apply loads to the top layer
ty = list(range(2*(npoint-cols-1)+1,2*npoint,2)) #+ list(range(ndof-(cols+1)*nv_rod+1,ndof,2))
for n in range(int(totalTime/dt)):
    Q[n][ty] += Q_point * n / (totalTime//dt)

# # Apply loads to the top corners
# W[2*(npoint-cols)-1] -= 5000
# W[2*npoint-1] -= 5000

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
Nsteps = int(totalTime / dt)
ctime = 0 # Current time

# Some arrays to store the results (not mandatory)
all_pos = np.zeros(Nsteps) # y coordinate of middle node
all_v = np.zeros(Nsteps) # y velocity of middle node
midAngle = np.zeros(Nsteps) # angle/bent shape at middle node (radians)

# Create an output directory for frames
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

# Set parameters for plotting and simulation
plotStep = 20  # Save a frame every 50 time steps


# Calculate axis limits with some padding
x_min, x_max = min(q0[::2]), max(q0[::2])
y_min, y_max = min(q0[1::2]), max(q0[1::2])
x_padding, y_padding = 0.3, 0.5

x_limits = (x_min - x_padding, x_max + x_padding)
y_limits = (y_min - 0.3, y_max + y_padding)

ctime = 0  # Initialize simulation time
h_initial = max(q0[1::2]) - min(q0[1::2]) # Initial height

for timeStep in range(Nsteps):
    # Simulation logic (update positions, velocities, etc.)
    q, flag = objfun(q0, q0, u, dt, tol, maximum_iter, m, mMat,  # inertia
                     EI, EA,  # elastic stiffness
                     W, Q[timeStep],  # external force
                     deltaL, npoint, nv_rod,
                     rods, structure_nodes, connection, curvature_dict,
                     free_index)
    if flag < 0:
        print("Could not converge")
        break
    u = (q - q0) / dt  # Update velocity
    q0 = q.copy()      # Update position

    # Update the simulation for plotStep iterations
    if timeStep % plotStep == 0:
        print(ctime)

        # Plot the current state and save it as a PNG
        x_arr = q[::2]  # Extract x-coordinates
        y_arr = q[1::2]  # Extract y-coordinates
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.scatter(x_arr, y_arr, color='green', s=3)

        # Plot each rod as a line connecting its nodes
        for rod in rods_dict:
            start, end = rods_dict[rod]
            x_values = ([q0[2 * start]] +
                        list(q0[2 * npoint + 2 * rod * nv_rod: 2 * npoint + 2 * (rod + 1) * nv_rod:2]) +
                        [q0[2 * end]])
            y_values = ([q0[2 * start + 1]] +
                        list(q0[2 * npoint + 2 * rod * nv_rod + 1: 2 * npoint + 2 * (rod + 1) * nv_rod + 1:2]) +
                        [q0[2 * end + 1]])
            plt.plot(x_values, y_values, 'b-', lw=3)

        # Add loading arrows (green downward arrows)
        arrow_y = max(q0[1::2]) + 0.4  # Arrow starting y-coordinate
        for x in q0[2 * (npoint - cols - 1): 2 * npoint: 2]:  # Arrow x-coordinates
            plt.arrow(x, arrow_y, 0, -0.2, head_width=0.1, head_length=0.1,
                      fc='green', ec='green')

        # Add boundary conditions (blue triangles)
        triangle_y = min(q0[1::2]) - 0.1
        for x in q0[0: 2 * (cols + 1): 2]:  # Triangle x-coordinates
            plt.plot([x - 0.06, x + 0.06, x, x - 0.06],
                     [triangle_y, triangle_y, triangle_y + 0.1, triangle_y], 'b-')

        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        # Set consistent axis limits
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)

        plt.title(f"Simulation Time: {ctime:.3f}s")  # Add a title with simulation time
        plt.xlabel("x")  # Label x-axis
        plt.ylabel("y")  # Label y-axis
        # plt.axis('auto')  # Set equal scaling for x and y axes
        plt.grid(True)  # Add gridlines
        # Save the plot as a PNG file
        plt.savefig(os.path.join(output_dir, f"frame_{timeStep//plotStep:06d}.png"))
        plt.close()  # Close the plot to free memory

    ctime += dt  # Update simulation time

h_final = max(q0[1::2]) - min(q0[1::2]) # Final height

# Record CompressedRatio
CompressedRatio = h_final / h_initial

with open('simulation_results.txt', 'a') as f:
    f.write(f"h_final = {h_final:.4f}\n")
    f.write(f"CompressedRatio = {CompressedRatio:.4f}\n")
