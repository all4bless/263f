import numpy as np

def objfun(q_guess, q_old, u_old,
           dt, tol, maximum_iter,
           m, mMat, # inertia
           EI, EA, # elastic stiffness
           W, Q, # external force
           deltaL, npoint, nv_rod,
           rods, structure_nodes, connection, curvature_dict,
           free_index): # free_index indicates the DOFs that evolve under Equations of Motion w/o boundary conditions

    # q_guess: Guess solution for the DOF vector at the new timestep
    # q_old: Old position (DOF) from the old time step
    # u_old: Old velocity (derivative of DOFs) from the old time step
    # Need to compute q_new

    q_new = q_guess.copy()
    q_new_special = q_new[:2*npoint]
    q_new_common = q_new[2*npoint:]

    # Newton-Raphson scheme
    error = 10 * tol
    iter_count = 0 # number of iterations
    flag = 1 # start with a positive ("good") flag

    rods_dict = {idx: rod for idx, rod in enumerate(rods)}
    rods_idx = {rod: idx for idx, rod in enumerate(rods)}

    while error > tol:
        # f = np.zeros_like(q_new)
        # J = np.zeros((f.shape[0], f.shape[0]))
        f = m / dt * ((q_new - q_old) / dt - u_old)
        J = mMat / dt ** 2

        for rod in rods_dict:
            s_flexible = list(range((2*npoint + 2*rod*nv_rod),(2*npoint + 2*(rod+1)*nv_rod)))
            s1 = [2*rods_dict[rod][0], 2*rods_dict[rod][0]+1] + s_flexible + [2*rods_dict[rod][1], 2*rods_dict[rod][1]+1]
            q_new_local = q_new[s1]

            # Calculate the elastic forces: Fb, Fs (-gradient of Eb and -gradient of Es)
            Fb, Jb = getFb(q_new_local, EI, deltaL)
            Fs, Js = getFs(q_new_local, EA, deltaL)

            # # Calculate the viscous force: Fv
            # Fv = - C @ (q_new - q_old) / dt
            # Jv = - C / dt
            # Calculate the "force" (LHS of equations of motion) and the Jacobian
            # f[s1] = f[s1] + m[s1] / dt * ( (q_new[s1] - q_old[s1])/dt - u_old[s1]) - (Fb+Fs+W[s1])
            # J[np.ix_(s1, s1)] = J[np.ix_(s1, s1)] + mMat[np.ix_(s1, s1)] / dt**2 - (Jb+Js)
            f[s1] = f[s1] - (Fb + Fs + W[s1] + Q[s1])
            J[np.ix_(s1, s1)] = J[np.ix_(s1, s1)] - (Jb + Js)
            # print(s1)
        # print(np.round(f,1))
        for nodes in curvature_dict:
            main_node = nodes[0]
            s_main = [2 * main_node, 2 * main_node + 1]
            if main_node < nodes[1]:
                s_neighbour1_x = 2 * npoint + 2 * rods_idx[(main_node, nodes[1])] * nv_rod
                s_neighbour2_x = 2 * npoint + 2 * rods_idx[(main_node, nodes[2])] * nv_rod
            elif nodes[1] < main_node < nodes[2]:
                s_neighbour1_x = 2 * npoint + 2 * (rods_idx[(nodes[1], main_node)]+1) * nv_rod - 2
                s_neighbour2_x = 2 * npoint + 2 * rods_idx[(main_node, nodes[2])] * nv_rod
            else:
                s_neighbour1_x = 2 * npoint + 2 * (rods_idx[(nodes[1], main_node)] + 1) * nv_rod - 2
                s_neighbour2_x = 2 * npoint + 2 * (rods_idx[(nodes[2], main_node)] + 1) * nv_rod - 2
            s_neighbour1 = [s_neighbour1_x, s_neighbour1_x + 1]
            s_neighbour2 = [s_neighbour2_x, s_neighbour2_x + 1]

            xk, yk = q_new[s_main]
            xkm1, ykm1 = q_new[s_neighbour1]
            xkp1, ykp1 = q_new[s_neighbour2]

            s2 = s_neighbour1+s_main+s_neighbour2
            # print(s2)
            # Compute the force due to E_b
            geb,flag1 = gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature_dict[nodes], deltaL, EI)
            if flag1 == -1:
                # print(s2)
                print(nodes)
                # print(len(q_new))
                print(curvature_dict[nodes])
                break
            f[s2] += geb
            # f[s2] += gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature_dict[nodes], deltaL, EI)  # Size is 6
            J[np.ix_(s2, s2)] += hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature_dict[nodes], deltaL, EI)  # Size is 6x6

            # # Compute the force due to E_s^k
            # Fs = -gradEs(xk, yk, xkp1, ykp1, deltaL, EA)  # Size is 4
            # Js = -hessEs(xk, yk, xkp1, ykp1, deltaL, EA)  # Size is 4x4
        # print(np.round(f,1))

        # We have to separate the "free" parts of f and J
        f_free = f[free_index]
        J_free = J[np.ix_(free_index, free_index)]

        # Newton's update
        # q_new = q_new - np.linalg.solve(J, f)
        # Only update the free DOFs
        dq_free = np.linalg.solve(J_free, f_free)
        q_new[free_index] = q_new[free_index] - dq_free

        # q_new[fixed_index] = q_fixed : not necessary here bc boundary conditions do not change with time

        # Calculate the error
        error = np.linalg.norm(f_free)

        # Update iteration number
        iter_count += 1
        # print("Iteration: %d" % iter_count)
        # print("Error: %f" % error)
        # if iter_count in [1,3,5,10,50,100]:
            # print(np.round(f_free,2))
            # print(np.round(J_free,2))
            # print(error)
            # print(np.round(dq_free,2))
            # print(np.round(q_new,2))
        if iter_count > maximum_iter:
          flag = -1
          print("Maximum number of iterations reached.")
          print(error)
          return q_new, flag

    return q_new, flag

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

def crossMat(a):
    """
    Returns the cross product matrix of vector 'a'.

    Parameters:
    a : np.ndarray
        A 3-element array representing a vector.

    Returns:
    A : np.ndarray
        The cross product matrix corresponding to vector 'a'.
    """
    A = np.array([[0, -a[2], a[1]],
                  [a[2], 0, -a[0]],
                  [-a[1], a[0], 0]])

    return A

def gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, EI):
    """
    Returns the derivative of bending energy E_k^b with respect to
    x_{k-1}, y_{k-1}, x_k, y_k, x_{k+1}, and y_{k+1}.

    Parameters:
    xkm1, ykm1 : float
        Coordinates of the previous node (x_{k-1}, y_{k-1}).
    xk, yk : float
        Coordinates of the current node (x_k, y_k).
    xkp1, ykp1 : float
        Coordinates of the next node (x_{k+1}, y_{k+1}).
    curvature0 : float
        Discrete natural curvature at node (xk, yk).
    l_k : float
        Voronoi length of node (xk, yk).
    EI : float
        Bending stiffness.

    Returns:
    dF : np.ndarray
        Derivative of bending energy.
    """

    # Nodes in 3D
    node0 = np.array([xkm1, ykm1, 0.0])
    node1 = np.array([xk, yk, 0])
    node2 = np.array([xkp1, ykp1, 0])

    # Unit vectors along z-axis
    m2e = np.array([0, 0, 1])
    m2f = np.array([0, 0, 1])

    kappaBar = curvature0

    # Initialize gradient of curvature
    gradKappa = np.zeros(6)

    # Edge vectors
    ee = node1 - node0
    ef = node2 - node1

    # Norms of edge vectors
    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)

    # Unit tangents
    te = ee / norm_e
    tf = ef / norm_f

    flag1 = 1

    # Curvature binormal
    chi = 1.0 + np.dot(te, tf)
    if abs(chi) < 1e-10:  # Add small threshold
        flag1 = -1
        chi = 1e-10  # Prevent division by zero

    kb = 2.0 * np.cross(te, tf) / chi

    # kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf))
    # chi = 1.0 + np.dot(te, tf)
    tilde_t = (te + tf) / chi
    tilde_d2 = (m2e + m2f) / chi

    # Curvature
    kappa1 = kb[2]

    # Gradient of kappa1 with respect to edge vectors
    Dkappa1De = 1.0 / norm_e * (-kappa1 * tilde_t + np.cross(tf, tilde_d2))
    Dkappa1Df = 1.0 / norm_f * (-kappa1 * tilde_t - np.cross(te, tilde_d2))

    # Populate the gradient of kappa
    gradKappa[0:2] = -Dkappa1De[0:2]
    gradKappa[2:4] = Dkappa1De[0:2] - Dkappa1Df[0:2]
    gradKappa[4:6] = Dkappa1Df[0:2]

    # Gradient of bending energy
    dkappa = kappa1 - kappaBar
    dF = gradKappa * EI * dkappa / l_k

    return dF, flag1

def hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, EI):
    """
    Returns the Hessian (second derivative) of bending energy E_k^b
    with respect to x_{k-1}, y_{k-1}, x_k, y_k, x_{k+1}, and y_{k+1}.

    Parameters:
    xkm1, ykm1 : float
        Coordinates of the previous node (x_{k-1}, y_{k-1}).
    xk, yk : float
        Coordinates of the current node (x_k, y_k).
    xkp1, ykp1 : float
        Coordinates of the next node (x_{k+1}, y_{k+1}).
    curvature0 : float
        Discrete natural curvature at node (xk, yk).
    l_k : float
        Voronoi length of node (xk, yk).
    EI : float
        Bending stiffness.

    Returns:
    dJ : np.ndarray
        Hessian of bending energy.
    """

    # Nodes in 3D
    node0 = np.array([xkm1, ykm1, 0])
    node1 = np.array([xk, yk, 0])
    node2 = np.array([xkp1, ykp1, 0])

    # Unit vectors along z-axis
    m2e = np.array([0, 0, 1])
    m2f = np.array([0, 0, 1])

    kappaBar = curvature0

    # Initialize gradient of curvature
    gradKappa = np.zeros(6)

    # Edge vectors
    ee = node1 - node0
    ef = node2 - node1

    # Norms of edge vectors
    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)

    # Unit tangents
    te = ee / norm_e
    tf = ef / norm_f

    # Curvature binormal
    kb = 2.0 * np.cross(te, tf) / (1.0 + np.dot(te, tf))

    chi = 1.0 + np.dot(te, tf)
    tilde_t = (te + tf) / chi
    tilde_d2 = (m2e + m2f) / chi

    # Curvature
    kappa1 = kb[2]

    # Gradient of kappa1 with respect to edge vectors
    Dkappa1De = 1.0 / norm_e * (-kappa1 * tilde_t + np.cross(tf, tilde_d2))
    Dkappa1Df = 1.0 / norm_f * (-kappa1 * tilde_t - np.cross(te, tilde_d2))

    # Populate the gradient of kappa
    gradKappa[0:2] = -Dkappa1De[0:2]
    gradKappa[2:4] = Dkappa1De[0:2] - Dkappa1Df[0:2]
    gradKappa[4:6] = Dkappa1Df[0:2]

    # Compute the Hessian (second derivative of kappa)
    DDkappa1 = np.zeros((6, 6))

    norm2_e = norm_e**2
    norm2_f = norm_f**2

    Id3 = np.eye(3)

    # Helper matrices for second derivatives
    tt_o_tt = np.outer(tilde_t, tilde_t)
    tmp = np.cross(tf, tilde_d2)
    tf_c_d2t_o_tt = np.outer(tmp, tilde_t)
    kb_o_d2e = np.outer(kb, m2e)

    D2kappa1De2 = (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt - tf_c_d2t_o_tt.T) / norm2_e - \
                  kappa1 / (chi * norm2_e) * (Id3 - np.outer(te, te)) + \
                  (kb_o_d2e + kb_o_d2e.T) / (4 * norm2_e)

    tmp = np.cross(te, tilde_d2)
    te_c_d2t_o_tt = np.outer(tmp, tilde_t)
    tt_o_te_c_d2t = te_c_d2t_o_tt.T
    kb_o_d2f = np.outer(kb, m2f)

    D2kappa1Df2 = (2 * kappa1 * tt_o_tt + te_c_d2t_o_tt + te_c_d2t_o_tt.T) / norm2_f - \
                  kappa1 / (chi * norm2_f) * (Id3 - np.outer(tf, tf)) + \
                  (kb_o_d2f + kb_o_d2f.T) / (4 * norm2_f)
    D2kappa1DeDf = -kappa1 / (chi * norm_e * norm_f) * (Id3 + np.outer(te, tf)) \
                  + 1.0 / (norm_e * norm_f) * (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt + \
                  tt_o_te_c_d2t - crossMat(tilde_d2))
    D2kappa1DfDe = D2kappa1DeDf.T

    # Populate the Hessian of kappa
    DDkappa1[0:2, 0:2] = D2kappa1De2[0:2, 0:2]
    DDkappa1[0:2, 2:4] = -D2kappa1De2[0:2, 0:2] + D2kappa1DeDf[0:2, 0:2]
    DDkappa1[0:2, 4:6] = -D2kappa1DeDf[0:2, 0:2]
    DDkappa1[2:4, 0:2] = -D2kappa1De2[0:2, 0:2] + D2kappa1DfDe[0:2, 0:2]
    DDkappa1[2:4, 2:4] = D2kappa1De2[0:2, 0:2] - D2kappa1DeDf[0:2, 0:2] - \
                         D2kappa1DfDe[0:2, 0:2] + D2kappa1Df2[0:2, 0:2]
    DDkappa1[2:4, 4:6] = D2kappa1DeDf[0:2, 0:2] - D2kappa1Df2[0:2, 0:2]
    DDkappa1[4:6, 0:2] = -D2kappa1DfDe[0:2, 0:2]
    DDkappa1[4:6, 2:4] = D2kappa1DfDe[0:2, 0:2] - D2kappa1Df2[0:2, 0:2]
    DDkappa1[4:6, 4:6] = D2kappa1Df2[0:2, 0:2]

    # Hessian of bending energy
    dkappa = kappa1 - kappaBar
    dJ = 1.0 / l_k * EI * np.outer(gradKappa, gradKappa)
    dJ += 1.0 / l_k * dkappa * EI * DDkappa1

    return dJ

def gradEs(xk, yk, xkp1, ykp1, l_k, EA):
    """
    Calculate the gradient of the stretching energy with respect to the coordinates.

    Args:
    - xk (float): x coordinate of the current point
    - yk (float): y coordinate of the current point
    - xkp1 (float): x coordinate of the next point
    - ykp1 (float): y coordinate of the next point
    - l_k (float): reference length
    - EA (float): elastic modulus

    Returns:
    - F (np.array): Gradient array
    """
    F = np.zeros(4)
    F[0] = -(1.0 - np.sqrt((xkp1 - xk)**2.0 + (ykp1 - yk)**2.0) / l_k) * ((xkp1 - xk)**2.0 + (ykp1 - yk)**2.0)**(-0.5) / l_k * (-2.0 * xkp1 + 2.0 * xk)
    F[1] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1) / l_k * (-0.2e1 * ykp1 + 0.2e1 * yk)
    F[2] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1) / l_k * (0.2e1 * xkp1 - 0.2e1 * xk)
    F[3] = -(0.1e1 - np.sqrt((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k) * ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1) / l_k * (0.2e1 * ykp1 - 0.2e1 * yk)

    F = 0.5 * EA * l_k * F  # Scale by EA and l_k

    return F

def hessEs(xk, yk, xkp1, ykp1, l_k, EA):
    """
    This function returns the 4x4 Hessian of the stretching energy E_k^s with
    respect to x_k, y_k, x_{k+1}, and y_{k+1}.
    """
    J = np.zeros((4, 4))  # Initialize the Hessian matrix
    J11 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (-2 * xkp1 + 2 * xk) ** 2) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * ((-2 * xkp1 + 2 * xk) ** 2) / 0.2e1 - 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J12 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (-2 * ykp1 + 2 * yk) * (-2 * xkp1 + 2 * xk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * xkp1 + 2 * xk) * (-2 * ykp1 + 2 * yk) / 0.2e1
    J13 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * xkp1 - 2 * xk) * (-2 * xkp1 + 2 * xk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * xkp1 + 2 * xk) * (2 * xkp1 - 2 * xk) / 0.2e1 + 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J14 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) * (-2 * xkp1 + 2 * xk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * xkp1 + 2 * xk) * (2 * ykp1 - 2 * yk) / 0.2e1
    J22 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (-2 * ykp1 + 2 * yk) ** 2) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * ((-2 * ykp1 + 2 * yk) ** 2) / 0.2e1 - 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J23 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * xkp1 - 2 * xk) * (-2 * ykp1 + 2 * yk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * ykp1 + 2 * yk) * (2 * xkp1 - 2 * xk) / 0.2e1
    J24 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) * (-2 * ykp1 + 2 * yk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (-2 * ykp1 + 2 * yk) * (2 * ykp1 - 2 * yk) / 0.2e1 + 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J33 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * xkp1 - 2 * xk) ** 2) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * ((2 * xkp1 - 2 * xk) ** 2) / 0.2e1 - 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k
    J34 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) * (2 * xkp1 - 2 * xk)) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * (2 * xkp1 - 2 * xk) * (2 * ykp1 - 2 * yk) / 0.2e1
    J44 = (1 / ((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) / l_k ** 2 * (2 * ykp1 - 2 * yk) ** 2) / 0.2e1 + (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.3e1 / 0.2e1)) / l_k * ((2 * ykp1 - 2 * yk) ** 2) / 0.2e1 - 0.2e1 * (0.1e1 - np.sqrt(((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2)) / l_k) * (((xkp1 - xk) ** 2 + (ykp1 - yk) ** 2) ** (-0.1e1 / 0.2e1)) / l_k

    J = np.array([[J11, J12, J13, J14],
                   [J12, J22, J23, J24],
                   [J13, J23, J33, J34],
                   [J14, J24, J34, J44]])

    J *= 0.5 * EA * l_k

    return J

def getFs(q, EA, deltaL):
  # Calculate the elastic stretching force for DOF vector q
  # Size of the output Fs is the same as q
  # Size of the output Js is len(q) times len(q)

  Fs = np.zeros_like(q)
  Js = np.zeros((len(q), len(q)))

  ndof = len(q) # number of DOFs
  nv = int( ndof/2 )

  for k in range(0, nv-1): # loop over all the nodes except the last one
    # Get the coordinates of the current node
    xk = q[2*k]
    yk = q[2*k+1]
    xkp1 = q[2*k+2]
    ykp1 = q[2*k+3]
    ind = np.arange(2*k, 2*k+4) # [2*k, 2*k+1, 2*k+2, 2*k+3]

    # Compute the force due to E_s^k
    gradEnergy = gradEs(xk, yk, xkp1, ykp1, deltaL, EA) # Size is 4
    Fs[ind] = Fs[ind] - gradEnergy

    hessEnergy = hessEs(xk, yk, xkp1, ykp1, deltaL, EA) # Size is 4x4
    Js[np.ix_(ind, ind)] = Js[np.ix_(ind, ind)] - hessEnergy

  return Fs, Js

def getFb(q, EI, deltaL):
  # Calculate the elastic bending force for DOF vector q
  # Size of the output Fb is the same as q
  # Size of the output Jb is len(q) times len(q)

  Fb = np.zeros_like(q)
  Jb = np.zeros((len(q), len(q)))

  ndof = len(q) # number of DOFs
  nv = int( ndof/2 )

  for k in range(1, nv-1): # loop over all the nodes except the first and last one
    # Extract relevants DOFs from q
    xkm1 = q[2*k-2]
    ykm1 = q[2*k-1]
    xk = q[2*k]
    yk = q[2*k+1]
    xkp1 = q[2*k+2]
    ykp1 = q[2*k+3]
    ind = np.arange(2*k-2, 2*k+4) # [2*k-2, 2*k-1, 2*k, 2*k+1, 2*k+2, 2*k+3]

    # Compute the force due to E_b
    gradEnergy,flag1 = gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, 0, deltaL, EI) # Size is 6
    Fb[ind] = Fb[ind] - gradEnergy

    hessEnergy = hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, 0, deltaL, EI) # Size is 6x6
    Jb[np.ix_(ind,ind)] = Jb[np.ix_(ind,ind)] - hessEnergy

  return Fb, Jb
