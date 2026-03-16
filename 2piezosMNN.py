"""
Author: Francesco Ardrizzini
Research projct on Mechanical Neural Networks
"""


import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def fea_setup(n_elem, init_coordinates, init_connectivity_table):
    """ Build the FE mesh of the structure
    
    Input:
        n_elem: number of finite elements per beam
        init_coordinates: array of initial node coordinates
        init_connectivity_table: beam connectivity based on initial nodes
    
    Output:
        n_nodes: number of nodes per beam
        coordinates: array of FE nodal coordinates (renumbered)
        connectivity_table: FE element connectivity with updated node numbering
   """
        
    n_nodes = n_elem + 1 # number of nodes per beam
    nodes_coord = [] # global FE node coordinates
    connectivity_table = [] # FE element connectivity
    node_index = {} # map (x,y) -> node index
    
    # Beam suddivision
    for beam in init_connectivity_table:
        start = init_coordinates[beam[0]]
        end   = init_coordinates[beam[1]]
    
        dx = end[0] - start[0]
        dy = end[1] - start[1]
    
        local_nodes = []
    
        for i in range(n_nodes):
            x = start[0] + dx * i / n_elem
            y = start[1] + dy * i / n_elem
            new_node = (x, y)
    
            if new_node not in node_index:
                node_index[new_node] = len(nodes_coord)
                nodes_coord.append([x, y])
    
            local_nodes.append(node_index[new_node])
    
        # FE elements along the beam
        for i in range(n_elem):
            connectivity_table.append([local_nodes[i], local_nodes[i + 1]])
    
    coordinates = np.array(nodes_coord)
    connectivity_table = np.array(connectivity_table)
    
    
    # ---- Nodes renumbering according to convention --------------------------

    # Regroup nodes per y level
    levels = {}
    for i, (_, y) in enumerate(coordinates):
        levels.setdefault(y, []).append(i)
    
    # Order the levels: centre -> up -> down
    y0 = [y for y in levels.keys() if y == 0]
    y_pos = sorted([y for y in levels.keys() if y > 0])
    y_neg = sorted([y for y in levels.keys() if y < 0], reverse=True)
    y_levels = y0 + y_pos + y_neg
    
    # Build the new sorted list of nodes
    new_order = []
    for y in y_levels:
        new_order.extend(sorted(levels[y], key=lambda i: coordinates[i, 0]))
    
    # Map old -> new
    old_to_new = {old: new for new, old in enumerate(new_order)}
    
    # Sort coordinates
    coordinates = coordinates[new_order]
    
    # Update the connectivity table
    connectivity_table = np.array([[old_to_new[n1], old_to_new[n2]] for n1, n2 in connectivity_table]) 
    
    return n_nodes, coordinates, connectivity_table

def boundary_conditions(n_nodes, n_elem, coordinates, connectivity_table):
    """Define boundary conditions and active degrees of freedom
    
    Input:
        n_nodes: number of nodes per beam
        n_elem: number of elements per beam
        coordinates: FE nodal coordinates
        connectivity_table: FE element connectivity
    
    Output:
        n_elements: total number of FE elements
        n_nodes: total number of FE nodes
        n_dofs: total number of degrees of freedom
        dofs: array of all DoFs
        dofs_no_bcs: list of free DOFs
    """
    
    # first blocked dof of the extremity nodes to build the dofs_bcs array
    dof_1 = 3 * (8 * (n_elem - 1) + 3 * n_nodes - 1)
    dof_2 = dof_1 + n_elem * 3
    dof_3 = dof_1 * 2
    dof_4 = dof_3 + n_elem * 3
    
    dofs_bcs = [dof_1, dof_1 + 1, dof_1 + 2, dof_2, dof_2 + 1, dof_2 + 2, dof_3, dof_3 + 1, dof_3 + 2, dof_4, dof_4 + 1, dof_4 + 2]  # Table of degrees of freedom with boundary conditions, in particular locked nodes 5, 6, 10, 11
    
    # FE model size
    n_elements = connectivity_table.shape[0] # Total number of elements
    n_nodes = coordinates.shape[0] # Total number of nodes
    n_dofs = 3 * n_nodes # Total number of DoFs
    
    # DoFs arrays
    dofs = np.arange(n_dofs)
    dofs_no_bcs = list(set(dofs) - set(dofs_bcs))
    
    return n_elements, n_nodes, n_dofs, dofs, dofs_no_bcs


def beam_2d_element(coord1, coord2, E, I, A):
    """ Compute the global stiffness matrix of a 2D Euler–Bernoulli beam element
    
    Input:
        coord1: coordinates of node 1 [x, y]
        coord2: coordinates of node 2 [x, y]
        E: Young's modulus
        I: second moment of area
        A: cross-sectional area
    
    Output:
        Kg: 6x6 element stiffness matrix in global coordinates
    """
    
    L= np.linalg.norm(coord2-coord1) # Element length
    
    # Node coordinates
    x1, y1= coord1
    x2, y2= coord2
    
    # Direction cosines
    c= (x2-x1)/L
    s= (y2-y1)/L 
    
    # Local stiffness matrix
    K = np.array([
        [(E*A)/L, 0, 0, -(E*A)/L, 0, 0],
        [0, (12*E*I)/L**3, (6*E*I)/L**2, 0, (-12*E*I)/L**3, (6*E*I)/L**2],
        [0, (6*E*I)/L**2, (4*E*I)/L, 0, (-6*E*I)/L**2, (2*E*I)/L],
        [-(E*A)/L, 0, 0, (E*A)/L, 0, 0],
        [0, -(12*E*I)/L**3, (-6*E*I)/L**2, 0, (12*E*I)/L**3, (-6*E*I)/L**2],
        [0, (6*E*I)/L**2, (2*E*I)/L, 0, (-6*E*I)/L**2, (4*E*I)/L]
    ])
    
    # Transformation matrix local to global
    T = np.array([
        [c, s, 0, 0, 0, 0],
        [-s, c, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, c, s, 0],
        [0, 0, 0, -s, c, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    
    # Global stiffness matrix
    Kg = T.T @ K @ T
    
    return Kg

def global_reduced_K(n_dofs, dofs_no_bcs, n_elements, connectivity_table, coordinates, E_eq, I_eq, A_eq):
    """Assemble the global stiffness matrix of the structure and return the reduced matrix
    
    Input:
        n_dofs: total number of degrees of freedom
        dofs_no_bcs: list of free degrees of freedom
        n_elements: total number of beam elements
        connectivity_table: element-to-node connectivity
        coordinates: nodal coordinates array
        E_eq: equivalent Young's modulus
        I_eq: equivalent second moment of area
        A_eq: equivalent cross-sectional area
    
    Output:
        K_red: reduced global stiffness matrix
    """
    
    # Global stiffness matrix
    K = np.zeros((n_dofs, n_dofs))
    
    for ii in range(n_elements):
        left_node, right_node = connectivity_table[ii] 
        K_element = beam_2d_element(coordinates[left_node], coordinates[right_node], E_eq , I_eq, A_eq) 
        
        # Global DoF indices
        dof_left = 3 * left_node
        dof_right = 3 * right_node
        
        # Assembly into global matrix
        K[dof_left:dof_left + 3, dof_left:dof_left + 3] += K_element[0:3, 0:3]
        K[dof_right:dof_right + 3, dof_right:dof_right + 3] += K_element[3:6, 3:6]
        K[dof_left:dof_left + 3, dof_right:dof_right + 3] += K_element[0:3, 3:6]
        K[dof_right:dof_right + 3, dof_left:dof_left + 3] += K_element[3:6, 0:3]
    
    # Apply boundary conditions (static condensation)
    K_red = K[np.ix_(dofs_no_bcs, dofs_no_bcs)]

    return K_red # reduced stiffness matrix


def beam_voltage_to_element_voltage(V_beams, n_elem):
    """Assign top and bottom piezo voltages per beam to all its finite elements

    Input:
        V_beams: array (N_beams, 2)
                  V_beams[b,0] = V_top of beam b
                  V_beams[b,1] = V_bottom of beam b
        n_elem: number of finite elements per beam

    Output:
        V_top_elem: array (N_elements,)
        V_bot_elem: array (N_elements,)
    """

    V_beams = np.asarray(V_beams)
    assert V_beams.ndim == 2 and V_beams.shape[1] == 2, \
        "V_beams must have shape (N_beams, 2)"

    N_beams = V_beams.shape[0]
    N_elements = N_beams * n_elem

    V_top_elem = np.zeros(N_elements)
    V_bot_elem = np.zeros(N_elements)

    for b in range(N_beams):
        start = b * n_elem
        end   = (b + 1) * n_elem

        V_top_elem[start:end] = V_beams[b, 0]
        V_bot_elem[start:end] = V_beams[b, 1]

    return V_top_elem, V_bot_elem

    
    
def piezo_element_load(coord1, coord2, V_top, V_bot, cst_axial, cst_bending):
    """Equivalent nodal load vector for a 2D Euler–Bernoulli beam element
    with two piezoelectric patches (top & bottom)

    Axial force  ~ (V_top + V_bot)
    Bending moment ~ (V_top - V_bot)
    """

    # Element geometry
    L = np.linalg.norm(coord2 - coord1)
    x1, y1 = coord1
    x2, y2 = coord2
    c = (x2 - x1) / L
    s = (y2 - y1) / L

    # Piezo-induced resultants
    N_p = cst_axial * (V_top + V_bot)       # axial force
    M_p = cst_bending * (V_top - V_bot)     # bending moment

    # Local equivalent nodal load vector
    # DOFs: [u1, v1, θ1, u2, v2, θ2]
    f_loc = np.array([
        -N_p / 2,        0.0,        -M_p / 2,
        +N_p / 2,        0.0,        +M_p / 2
    ])

    # Rotation matrix (local → global)
    T = np.array([
        [ c,  s, 0, 0, 0, 0],
        [-s,  c, 0, 0, 0, 0],
        [ 0,  0, 1, 0, 0, 0],
        [ 0,  0, 0, c, s, 0],
        [ 0,  0, 0,-s, c, 0],
        [ 0,  0, 0, 0, 0, 1]
    ])

    return T.T @ f_loc


def reduced_global_f(f, dofs_no_bcs, width, E_pzt, d31, t_pzt, y_pzt, coordinates, connectivity_table, n_elements, V_top_elem, V_bot_elem):
    """Assemble the global load vector including piezoelectric contributions
    (two patches per beam element: top and bottom) and return the reduced
    force vector after applying boundary conditions

    INPUT:
        f               : global mechanical load vector (size n_dofs)
        dofs_no_bcs     : list of free degrees of freedom (after BCs)
        width           : beam width [mm]
        E_pzt           : Young modulus of piezoelectric material [MPa]
        d31             : piezoelectric strain constant [mm/V]
        t_pzt           : thickness of one piezo patch [mm]
        y_pzt           : distance of piezo centroid from beam neutral axis [mm]
        coordinates     : nodal coordinates array (n_nodes x 2)
        connectivity_table : element connectivity table (n_elements x 2)
        n_elements      : total number of finite elements
        V_top_elem      : voltage applied to the TOP piezo patch of each element
                          (array of size n_elements)
        V_bot_elem      : voltage applied to the BOTTOM piezo patch of each element
                          (array of size n_elements)

    OUTPUT:
        f_red           : reduced global force vector (free DOFs only)
    """

    # Axial force constant (extension / contraction)
    cst_axial = width * E_pzt * d31 / t_pzt

    # Bending moment constant (curvature)
    cst_bending = width * E_pzt * d31 * y_pzt

    # Assembly of piezoelectric equivalent nodal loads
    for e in range(n_elements):
        n1, n2 = connectivity_table[e]

        # Element-level piezoelectric equivalent load
        f_e = piezo_element_load(
            coordinates[n1],
            coordinates[n2],
            V_top_elem[e],
            V_bot_elem[e],
            cst_axial,
            cst_bending
        )

        # Global DOF indices
        dof1 = 3 * n1
        dof2 = 3 * n2

        # Assembly into global force vector
        f[dof1:dof1 + 3] += f_e[0:3]
        f[dof2:dof2 + 3] += f_e[3:6]

    # Reduction of force vector (boundary conditions applied)
    f_red = f[dofs_no_bcs]

    return f_red




def solve_fem(n_elem, coordinates, connectivity_table, n_elements, n_dofs, dofs_no_bcs, K_red, width, E_pzt, d31, t_pzt, y_pzt, V_beams_top, V_beams_bot, F1_applied, F2_applied, dof_f1, dof_f2):
    """Solve the full FEM system including mechanical loads and piezoelectric actuation
    (two piezo patches per beam: top and bottom).

    INPUT:
        n_elem              : number of finite elements per beam
        coordinates         : nodal coordinates array (n_nodes x 2)
        connectivity_table  : element connectivity table (n_elements x 2)
        n_elements          : total number of finite elements
        n_dofs              : total DOFs in the model
        dofs_no_bcs         : list of free DOFs after boundary conditions
        K_red               : reduced stiffness matrix
        width               : beam width [mm]
        E_pzt               : Young modulus of piezoelectric material [MPa]
        d31                 : piezoelectric strain constant [mm/V]
        t_pzt               : thickness of one piezo patch [mm]
        y_pzt               : distance of piezo centroid from neutral axis [mm]
        V_beams_top         : voltage applied to TOP piezo patch (one per beam)
        V_beams_bot         : voltage applied to BOTTOM piezo patch (one per beam)
        F1_applied          : external force applied at DOF dof_f1
        F2_applied          : external force applied at DOF dof_f2
        dof_f1              : DOF index of first applied force
        dof_f2              : DOF index of second applied force

    OUTPUT:
        q                   : full displacement vector (size n_dofs)
    """

    # Global mechanical load vector
    f = np.zeros(n_dofs)

    # Apply external mechanical forces
    f[dof_f1] += F1_applied
    f[dof_f2] += F2_applied

    
    # Convert beam voltages to element voltages
    V_beams = np.array([V_beams_top, V_beams_bot]).T
    V_top_elem, V_bot_elem = beam_voltage_to_element_voltage(V_beams, n_elem)

    # Reduced force vector including piezoelectric actuation
    f_red = reduced_global_f(
        f,
        dofs_no_bcs,
        width,
        E_pzt,
        d31,
        t_pzt,
        y_pzt,
        coordinates,
        connectivity_table,
        n_elements,
        V_top_elem,
        V_bot_elem
    )

    # Solve reduced FEM system
    q_red = np.linalg.solve(K_red, f_red)

    # Reconstruction of full displacement vector
    q = np.zeros(n_dofs)
    q[dofs_no_bcs] = q_red

    return q

def twoNodes_target(V_opt, target1, target2, n_elem, coordinates, connectivity_table, n_elements, n_dofs, dofs_no_bcs, K_red, width, E_pzt, d31, t_pzt, y_pzt, F1, F2, dof_f1, dof_f2):
    """
    Objective function for optimization:
    minimize the squared error between FEM displacements and prescribed targets
    at two output nodes, using piezoelectric actuation (top + bottom patches).

    INPUT:
        V_opt               : optimization vector [V_top_1,...,V_top_N, V_bot_1,...,V_bot_N]
        target1, target2    : target displacements [ux, uy] for output nodes
        n_elem              : number of finite elements per beam
        coordinates         : nodal coordinates (N x 2)
        connectivity_table  : element connectivity (Ne x 2)
        n_elements          : total number of elements
        n_dofs              : total DOFs
        dofs_no_bcs         : free DOFs after BCs
        K_red               : reduced stiffness matrix
        width, E_pzt, d31,
        t_pzt, y_pzt        : piezoelectric parameters
        F1, F2              : external forces
        dof_f1, dof_f2      : DOFs where forces are applied

    OUTPUT:
        err                 : scalar objective function value
    """

    # Split optimization vector into top and bottom voltages
    N_beams = len(V_opt) // 2
    V_beams_top = V_opt[:N_beams]
    V_beams_bot = V_opt[N_beams:]

    # Solve FEM with current piezo voltages
    q = solve_fem(
        n_elem,
        coordinates,
        connectivity_table,
        n_elements,
        n_dofs,
        dofs_no_bcs,
        K_red,
        width,
        E_pzt,
        d31,
        t_pzt,
        y_pzt,
        V_beams_top,
        V_beams_bot,
        F1,
        F2,
        dof_f1,
        dof_f2
    )

    # DOFs of output nodes (x,y translations)
    dof_u1 = (7 * n_elem - 3) * 3
    dof_u2 = (18 * n_elem - 9) * 3

    # Extract displacements
    scale_opt = 1e3  # numerical conditioning

    u1 = scale_opt * np.array([q[dof_u1],     q[dof_u1 + 1]])
    u2 = scale_opt * np.array([q[dof_u2],     q[dof_u2 + 1]])

    t1 = scale_opt * target1
    t2 = scale_opt * target2

    # Objective function (MSE)
    err1 = np.sum((u1 - t1)**2)
    err2 = np.sum((u2 - t2)**2)

    return err1 + err2


def two_target(V_opt, t1_1, t1_2, t2_1, t2_2, n_elem, coordinates, connectivity_table, n_elements, n_dofs, dofs_no_bcs, K_red, width, E_pzt, d31, t_pzt, y_pzt, F1_1, F2_1, F1_2, F2_2, dof_f1_1, dof_f2_1, dof_f1_2, dof_f2_2):
    """
    Objective function for two independent load cases using piezoelectric actuation
    with top and bottom patches. For each load case, prescribed target displacements 
    (ux, uy) are defined at two output nodes, with different targets between nodes 
    and between load cases. The objective function is defined as the sum of squared 
    errors between computed and target displacements.

    INPUT:
        V_opt               : optimization vector [V_top_1,...,V_top_N, V_bot_1,...,V_bot_N]
        t1_1, t1_2          : target displacements in load case 1 [ux, uy] for output nodes
        t2_1, t2_2          : target displacements in load case 2 [ux, uy] for output nodes
        n_elem              : number of finite elements per beam
        coordinates         : nodal coordinates (N x 2)
        connectivity_table  : element connectivity (Ne x 2)
        n_elements          : total number of elements
        n_dofs              : total DOFs
        dofs_no_bcs         : free DOFs after BCs
        K_red               : reduced stiffness matrix
        width, E_pzt, d31,
        t_pzt, y_pzt        : piezoelectric parameters
        F1_1, F2_1          : external forces (load case 1)
        F1_2, F2_2          : external forces (load case 2)
        dof_f1_1, dof_f2_1  : DOFs of forces (case 1)
        dof_f1_2, dof_f2_2  : DOFs of forces (case 2)

    OUTPUT:
        err                 : scalar objective function value
    """

    # Split optimization vector into top and bottom voltages
    N_beams = len(V_opt) // 2
    V_beams_top = V_opt[:N_beams]
    V_beams_bot = V_opt[N_beams:]

    # Solve FEM – load case 1
    q1 = solve_fem(
        n_elem,
        coordinates,
        connectivity_table,
        n_elements,
        n_dofs,
        dofs_no_bcs,
        K_red,
        width,
        E_pzt,
        d31,
        t_pzt,
        y_pzt,
        V_beams_top,
        V_beams_bot,
        F1_1,
        F2_1,
        dof_f1_1,
        dof_f2_1
    )

    # Solve FEM – load case 2
    q2 = solve_fem(
        n_elem,
        coordinates,
        connectivity_table,
        n_elements,
        n_dofs,
        dofs_no_bcs,
        K_red,
        width,
        E_pzt,
        d31,
        t_pzt,
        y_pzt,
        V_beams_top,
        V_beams_bot,
        F1_2,
        F2_2,
        dof_f1_2,
        dof_f2_2
    )

    # DOFs of output nodes
    dof_u1 = (7 * n_elem - 3) * 3
    dof_u2 = (18 * n_elem - 9) * 3

    scale_opt = 1e3

    # Extract displacements
    u1_case1 = scale_opt * np.array([q1[dof_u1], q1[dof_u1 + 1]])
    u2_case1 = scale_opt * np.array([q1[dof_u2], q1[dof_u2 + 1]])

    u1_case2 = scale_opt * np.array([q2[dof_u1], q2[dof_u1 + 1]])
    u2_case2 = scale_opt * np.array([q2[dof_u2], q2[dof_u2 + 1]])

    t1_1 = scale_opt * t1_1
    t1_2 = scale_opt * t1_2
    t2_1 = scale_opt * t2_1
    t2_2 = scale_opt * t2_2
    # Objective function
    err = (
        np.sum((u1_case1 - t1_1)**2) +
        np.sum((u2_case1 - t1_2)**2) +
        np.sum((u1_case2 - t2_1)**2) +
        np.sum((u2_case2 - t2_2)**2)
    )

    return err

def plot_fea_structure(connectivity_table, coordinates):
    """Plot the undeformed FE mesh
    
    Input:
        connectivity_table: FE element connectivity
        coordinates: FE nodal coordinates
    
    Output:
        Graphical visualization
    """
    plt.figure(figsize=(6, 6))
    
    # Draw the elements as lines between nodes
    for elem in connectivity_table:
        n1, n2 = elem
        plt.plot(
            [coordinates[n1, 0], coordinates[n2, 0]],
            [coordinates[n1, 1], coordinates[n2, 1]],
            'k-', linewidth=1.2
        )
    
    # Plot the nodes
    plt.scatter(coordinates[:, 0], coordinates[:, 1], s=40, color='royalblue', zorder=3)
    
    # label of the nodes
    for i, (x, y) in enumerate(coordinates):
        plt.text(x, y + 30, str(i), fontsize=8, ha='center', color='darkred')
    
    plt.axis('equal')
    plt.title("FE mesh")
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()

def plot_deformed_structure(coordinates, connectivity_table, q, V_elem_top, V_elem_bot, scale,
                            target1=None, target2=None, node_target_1=None, node_target_2=None):
    """
    Plot undeformed and deformed 2D beam FEM structure with two piezo patches
    (top and bottom) per beam element.

    Input:
        coordinates: nodal coordinates (N x 2)
        connectivity_table: element connectivity (Ne x 2)
        q: global displacement vector
        V_elem_top: voltage per element (top piezo)
        V_elem_bot: voltage per element (bottom piezo)
        scale: deformation amplification factor
        target1, target2: target displacements [ux, uy]
        node_target_1, node_target_2: target node indices
    """

    # Extract translational displacements
    u = q[0::3]
    v = q[1::3]

    coord_def = np.zeros_like(coordinates)
    coord_def[:, 0] = coordinates[:, 0] + scale * u
    coord_def[:, 1] = coordinates[:, 1] + scale * v

    plt.figure(figsize=(7, 7))

    # Undeformed structure
    for n1, n2 in connectivity_table:
        plt.plot(
            [coordinates[n1, 0], coordinates[n2, 0]],
            [coordinates[n1, 1], coordinates[n2, 1]],
            color='gray', linewidth=1.0, zorder=1
        )

    # Deformed structure + piezo patches
    offset = 10  # distance of piezo strip from beam (visual only)

    for e, (n1, n2) in enumerate(connectivity_table):

        x1, y1 = coord_def[n1]
        x2, y2 = coord_def[n2]

        # Beam direction
        dx = x2 - x1
        dy = y2 - y1
        L = np.hypot(dx, dy)

        # Normal direction
        nx = -dy / L
        ny =  dx / L

        # -------- Beam line (neutral axis)
        plt.plot([x1, x2], [y1, y2],
                 color='black', linewidth=1.0, zorder=2)

        # -------- Top piezo patch
        if V_elem_top[e] > 0:
            col_top = 'red'
        elif V_elem_top[e] < 0:
            col_top = 'blue'
        else:
            col_top = 'darkgrey'

        plt.plot(
            [x1 + offset * nx, x2 + offset * nx],
            [y1 + offset * ny, y2 + offset * ny],
            color=col_top, linewidth=1.0, zorder=3
        )

        # -------- Bottom piezo patch
        if V_elem_bot[e] > 0:
            col_bot = 'red'
        elif V_elem_bot[e] < 0:
            col_bot = 'blue'
        else:
            col_bot = 'darkgrey'

        plt.plot(
            [x1 - offset * nx, x2 - offset * nx],
            [y1 - offset * ny, y2 - offset * ny],
            color=col_bot, linewidth=1.0, zorder=3
        )

    # Target points
    if target1 is not None and node_target_1 is not None:
        x0, y0 = coordinates[node_target_1]
        plt.scatter(
            x0 + target1[0] * scale,
            y0 + target1[1] * scale,
            marker='x', s=120, linewidths=2,
            color='black', zorder=5
        )

    if target2 is not None and node_target_2 is not None:
        x0, y0 = coordinates[node_target_2]
        plt.scatter(
            x0 + target2[0] * scale,
            y0 + target2[1] * scale,
            marker='x', s=120, linewidths=2,
            color='black', zorder=5, label='Targets'
        )
        plt.legend()

    plt.axis('equal')
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.title(f"Deformed structure with piezo patches (scale = {scale})")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()


# ========================PROBLEM TO SOLVE ====================================

if __name__ == '__main__':

# ========= Material, geometrical properties and applied forces ===============
    l = 400 #[mm] beam length    

    # Beam materiel properties
    E_beam = 70e3  # [MPa] 90
    t_beam = 2 # [mm] 10
    width = 20   # [mm]
    A_beam = t_beam * width # [mm^2]
    
    I_beam = width * t_beam**3 / 12 # [mm^4]
    
    # Piezoelectric materiel properties
    E_pzt = 62e3 # [MPa]
    t_pzt = 0.5 # [mm]
    A_pzt = t_pzt * width # [mm^2]
    d31 = -274e-9 # [mm/V]
    
    y_pzt = (t_beam + t_pzt) / 2
    I_pzt = 2 * (width * t_pzt**3 / 12 + width * t_pzt * y_pzt**2) # [mm^4] # Inertia wrt neutral axis of the beam
    
    # Equivalent section properties (beam + piezo)
    EA_eq = E_beam * A_beam + 2* (E_pzt * A_pzt)
    EI_eq = E_beam * I_beam + E_pzt * I_pzt
    
    A_eq = A_beam + 2 * A_pzt
    E_eq = EA_eq / A_eq
    I_eq = EI_eq / E_eq
    
    # Applied forces
    F1 = 100 # N
    F2 = 200 # N
    
# ========================== PROBLEM GEOMETRY =================================
    
    # Initial nodes cooridnates -> need to devide each beam in more elements -> done in a next step
    init_coordinates = np.array([
        [0.0, 0.0],  #0
        [l, 0.0],  #1
        [-0.5*l, 0.866*l],  #2
        [0.5*l, 0.866*l], #3 
        [1.5*l, 0.866*l], #4 
        [0, 2*0.866*l], #5 
        [l, 2*0.866*l],  #6
        [-0.5*l, -0.866*l], #7
        [0.5*l, -0.866*l], #8
        [1.5*l, -0.866*l], #9
        [0, -2*0.866*l], #10 
        [l, -2*0.866*l],  #11
    ])
    
# Connectivity table, each line corresponds to an element and contains the indices of the two nodes that make it up
    init_connectivity_table = np.array([
        [0, 1], #0
        [0, 2], #1
        [0, 3], #2
        [1, 3], #3
        [1, 4], #4
        [2, 3], #5
        [3, 4], #6
        [2, 5], #7
        [3, 5], #8
        [3, 6], #9
        [4, 6], #10
        [5, 6], #11
        [0, 7], #12
        [0, 8], #13
        [1, 8], #14
        [1, 9], #15
        [7, 8], #16
        [8, 9], #17
        [7, 10], #18
        [8, 10], #19
        [8, 11], #20
        [9, 11], #21
        [10, 11], #22
    ])
    
# ==================== FEA SETUP ==========================================
    
    # Creation of the partition of the structure 
    n_elem = 10 # Number of elements per beam to perform the FEM
    n_node, coordinates, connectivity_table = fea_setup(n_elem, init_coordinates, init_connectivity_table)
    
    # Plot of the fractioned structure
    plot_fea_structure(connectivity_table, coordinates)
    
# =================== BONDARY CONDITIONS ======================================
    
    n_elements, n_nodes, n_dofs, dofs, dofs_no_bcs = boundary_conditions(n_node, n_elem, coordinates, connectivity_table)
     
# ================ Finite element modelling ===================================
    
    K_red = global_reduced_K(n_dofs, dofs_no_bcs, n_elements, connectivity_table, coordinates, E_eq, I_eq, A_eq)
    
# ======= CONSTRUCTION OF THE LOAD VECTOR =====================================
    
    f = np.zeros(n_dofs)
    
# ==== External load applied to prescribed nodes
    
    dof_f1 = (5 * n_elem - 3) * 3# first dof of node number 2 (in original numbering)
    dof_f2 = (16 * n_elem - 9) * 3 # first dof of node number 7 (in original numbering)
    
    f[dof_f1]  += F2
    f[dof_f2] += F2
    
    
# # ===== Random voltage assignment:

    V_top = [-134.7,  82.3, -191.6,  47.9, 163.2, -29.5,  98.1, -176.4,  12.8, 154.6, -67.2, 201.0, -143.9,  36.4, -118.7, 189.3,  74.1, -52.6, 166.8, -91.4, 23.7, -159.2, 108.5]
    V_bottom = [45.6, -210.3,  132.8, -67.4,  98.9, 174.2, -143.5,  21.7, -189.6,  64.1, 212.4, -35.8,   9.6, -156.7,  187.9, -92.3,  118.5, -203.4,  56.2,  141.8, -74.9,  29.3, -165.1]
    V_beams = np.array([V_top, V_bottom]).T
   
# ====== From beam's piezo voltage to element voltage ========================= 
    
    Vel_top, Vel_bottom  = beam_voltage_to_element_voltage(V_beams, n_elem)

# ====== Reduced force vectors considering the boundary conditions ============
    
    f_red = reduced_global_f(f, dofs_no_bcs, width, E_pzt, d31, t_pzt, y_pzt, coordinates, connectivity_table, n_elements, Vel_top, Vel_bottom)
    
# ====== RESOLUTION OF THE SYSTEM Ku = F ======================================
    # Resolution of the reduced system
    q_red = np.linalg.solve(K_red, f_red) 
    
    # Reconstruct full displacement vector
    q = np.zeros(n_dofs)
    q[dofs_no_bcs] = q_red
    

# ====== VISUALIZATION OF THE RESULTS =========================================
    print("\n\n\n=========== Results for arbitrary voltage case ====================")
    dof_u1 = (7 * n_elem - 3) * 3 # first dof of the node number 4 in original numbering
    dof_u2 = (18 * n_elem - 9) * 3 # first dof of the node number 9 in original numbering 
    print(f"\nDisplacement of node 4 : {q[dof_u1]}mm along x and {q[dof_u1 + 1]}mm along y \n")
    print(f"Displacement of node 9 : {q[dof_u2]}mm along x and {q[dof_u2 + 1]}mm along y \n")
    
    plot_deformed_structure(coordinates, connectivity_table, q, Vel_top, Vel_bottom, scale = 3e2)
    
    
# ====== OPTIMIZATION FOR ONE LOAD CASE  ==========================================================
    print("\n\n\n=========== Results for optimisation of one load case ==============")
    # Targets (mm)
    target1 = np.array([5e-3, 9e-3])
    target2 = np.array([7e-3, 7e-3])
    
    # Initial guess
    V0_top = np.ones(23)
    V0_bot = np.ones(23)
    V0 = np.concatenate((V0_top, V0_bot))
    
    # Bounds on voltages
    bnds = [(-200, 200) for _ in range(23*2)]
    
    res = minimize(
        twoNodes_target,
        V0,
        args=(
            target1, target2,
            n_elem, coordinates, connectivity_table,
            n_elements, n_dofs, dofs_no_bcs,
            K_red, width, E_pzt, d31, t_pzt, y_pzt,
            F1, F1, dof_f1, dof_f2
        ),
        method='SLSQP',
        bounds=bnds,
        options={'ftol': 1e-10, 'disp': True}
    )
    
    print("\nOptimization success:", res.success)
    print("Final error:", res.fun)
    print("Optimal voltages (first 23 values -> top patches, latter 23 values -> bottom patches):", res.x)

    print("\n\nThis is the optimized structure results with prescribed displacement of the two right node")
    
    # ================== POST-PROCESSING: OPTIMIZED SOLUTION =====================
    V_top, V_bot = res.x[:23], res.x[23:]
    #V_top, V_bot = np.zeros(23), np.zeros(23)
    # Solve FEM with optimal voltages
    q_opt = solve_fem(n_elem, coordinates, connectivity_table, n_elements, n_dofs, dofs_no_bcs, K_red, width, E_pzt, d31, t_pzt, y_pzt, V_top, V_bot, F1, F1, dof_f1, dof_f2)
    
    # Element voltages
    Vel_opt_top, Vel_opt_bot = beam_voltage_to_element_voltage(np.array([V_top, V_bot]).T, n_elem)
    
    # Target node indices (from DOFs)
    node_target_1 = (7 * n_elem - 3)
    node_target_2 = (18 * n_elem - 9)
    
    # Plot optimized structure with targets
    plot_deformed_structure(coordinates, connectivity_table, q_opt, Vel_opt_top, Vel_opt_bot, scale = 1e4, target1=target1, target2=target2, node_target_1=node_target_1, node_target_2=node_target_2)
    
    print("\nAchieved displacement node 1:", q_opt[3*node_target_1:3*node_target_1+2])
    print("Target displacement node 1:", target1)
    
    print("\nAchieved displacement node 2:", q_opt[3*node_target_2:3*node_target_2+2])
    print("Target displacement node 2:", target2)








# ====== OPTIMIZATION FOR TWO LOAD CASES  ==========================================================
    print("\n\n\n=========== Results for optimisation of two load cases (MNN) ==============")
    # Targets (in mm)
    target1_1 = np.array([0.005, 0.009]) 
    target1_2 = np.array([0.007, 0.007])
    target2_1 = np.array([0.003, 0.005]) 
    target2_2 = np.array([0.008, 0.001])
    
    # Initial guess
    V0_top = np.ones(23)
    V0_bot = np.ones(23)
    V0 = np.concatenate((V0_top, V0_bot))
    
    # Bounds on voltages
    bnds = [(-200, 200) for _ in range(23 * 2)]
    
    res = minimize(
        two_target,
        V0,
        args=(
            target1_1, target1_2, target2_1, target2_2,
            n_elem, coordinates, connectivity_table,
            n_elements, n_dofs, dofs_no_bcs,
            K_red, width, E_pzt, d31, t_pzt, y_pzt,
            F1, F1, F2, F2, dof_f1, dof_f2, dof_f1, dof_f2,
        ),
        method='SLSQP',
        bounds=bnds,
        options={'ftol': 1e-10, 'disp': True}
    )
    
    print("\nOptimization success:", res.success)
    print("Final error:", res.fun)
    print("Optimal voltages (first 23 values -> top patches, latter 23 values -> bottom patches):", res.x)

    print("\n\nThis is the optimized structure results with prescribed displacement of the two right node")
    
    # ================== POST-PROCESSING: OPTIMIZED SOLUTION =====================
    
    V_top, V_bot = res.x[:23], res.x[23:]
    # Solve FEM with optimal voltages
    q_opt = solve_fem(n_elem, coordinates, connectivity_table, n_elements, n_dofs, dofs_no_bcs, K_red, width, E_pzt, d31, t_pzt, y_pzt, V_top, V_bot, F1, F1, dof_f1, dof_f2)
    
    # Element voltages for coloring
    Vel_opt_top, Vel_opt_bot = beam_voltage_to_element_voltage(np.array([V_top, V_bot]).T, n_elem)
    
    # Target node indices (from DOFs)
    node_target_1 = (7 * n_elem - 3)
    node_target_2 = (18 * n_elem - 9)
    
    # Plot optimized structure with targets
    plot_deformed_structure(coordinates, connectivity_table, q_opt, Vel_opt_top, Vel_opt_bot, scale = 1e4, target1=target1_1, target2=target1_2, node_target_1=node_target_1, node_target_2=node_target_2)
    
    print("\nAchieved displacement node 1:", q_opt[3*node_target_1:3*node_target_1+2])
    print("Target displacement node 1:", target1_1)
    
    print("\nAchieved displacement node 2:", q_opt[3*node_target_2:3*node_target_2+2])
    print("Target displacement node 2:", target1_2)

