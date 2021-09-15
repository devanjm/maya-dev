import numpy as np

from maya import cmds
from maya.api import OpenMaya, OpenMayaAnim


def run(mesh,
        influences,
        ids=None,
        weights_pp=None,
        falloff_distance=0.5,
        interp_type='linear',
        ):
    """
    Generate the weights, and apply them in one step.

    See generate_weights and apply_weights.

    Args:
        mesh (str): Name of the mesh.
        influences (list/str): List of influences to generate weights for.
        ids (list/int): List of vertex indices to modify weights for, or None for all.
        weights_pp (list/float): List of per vertex weights_pp used to
                                blend the generated weights with the original.
        falloff_distance (float): Value between 0-1 indicating how far into the next/prev
                                  influence the weights for the closest influence will effect.
        interp_type (str): Weight curve interp type.  Valid types are "linear" and "smooth".

    """

    weights = generate_weights(mesh,
                               influences,
                               ids=ids,
                               falloff_distance=falloff_distance,
                               interp_type=interp_type)

    apply_weights(mesh, weights, influences, ids=ids, weights_pp=weights_pp)


def generate_weights(mesh,
                     influences,
                     ids=None,
                     falloff_distance=0.5,
                     interp_type='linear',
                     ):
    """
    Generate weights for a sequence of influences that will ramp up/down as points
    approach the next influence in the sequence.

    Finds the closest point from the point to the vector between influences,
    and uses that position to apply weights according to a linear or smooth
    step curve.

    Useful to weight tubular objects.  Fingers, tails, etc.

    Args:
        mesh (str): Name of the mesh.
        influences (list/str): List of influences to generate weights for.
        ids (list/int): List of vertex indices to modify weights for, or None for all.
        falloff_distance (float): Value between 0-1 indicating how far into the next/prev
                                  influence the weights for the closest influence will effect.
        interp_type (str): Weight curve interp type.  Valid types are "linear" and "smooth".

    Returns (np.ndarray): The weights array.

    """

    # Get point position data
    points = get_mfn_mesh(mesh).getPoints(OpenMaya.MSpace.kWorld)
    points = np.array([[i.x, i.y, i.z] for i in points])
    if ids:
        points = points[ids]

    num_points = points.shape[0]

    # Since the arg is expressed as distance, convert it to the start point in the falloff curve.
    edge0 = 1 - falloff_distance

    # Get influence position data
    joint_positions = np.array([cmds.xform(i, q=1, ws=1, t=1) for i in influences])

    # This is an array starting at 0 - num_points for indexing relative to the number of points passed in
    v_idx_arr = np.arange(num_points)

    # Get the closest line segment index, and the closest point on that line
    closest_index, closest_point, line_vectors = get_closest_line_points(points, joint_positions)
    num_lines = line_vectors.shape[0]

    # Create empty weight array of zeros
    weights = np.zeros(num_points * num_lines).reshape(num_points, num_lines)

    # If the falloff distance is 0, there is no transition between influences at all,
    # so just assign full weights to the closest joint.
    if falloff_distance == 0:
        weights[v_idx_arr, closest_index] = 1
        return weights

    # Get line vector lengths
    line_lengths = np.linalg.norm(line_vectors[closest_index], axis=1)

    # get line start to closest point lengths
    length_closest_point = np.linalg.norm(closest_point - joint_positions[closest_index], axis=1)

    # divide to get our normalized length along the line for the closest point
    wgts_x = length_closest_point / line_lengths

    # First joint weights
    first_joint_mask = closest_index == 0

    # Generate the weights based on lookup curve using normalized distance along joint as x
    closest_joint_weights = weights_lookup_max_to_min(wgts_x[first_joint_mask],
                                                      edge0,
                                                      interp_type)
    weights[v_idx_arr[first_joint_mask], closest_index[first_joint_mask]] = closest_joint_weights
    weights[v_idx_arr[first_joint_mask], closest_index[first_joint_mask] + 1] = 1 - closest_joint_weights

    # Last joint weights
    last_joint_mask = closest_index == np.max(closest_index)
    closest_joint_weights = weights_lookup_min_to_max(wgts_x[last_joint_mask],
                                                      edge0,
                                                      interp_type)
    weights[v_idx_arr[last_joint_mask], closest_index[last_joint_mask]] = closest_joint_weights
    weights[v_idx_arr[last_joint_mask], closest_index[last_joint_mask] - 1] = 1 - closest_joint_weights

    # Middle joint weights
    middle_segment_lines = np.sort(np.unique(closest_index))[1:-1]
    if middle_segment_lines.size > 0:

        # Create a mask for all the points which have a joint on either side,
        # indicating a link in the middle of the chain
        middle_segment_mask = np.logical_not(first_joint_mask | last_joint_mask)

        # Apply that mask to the wgts_x for reuse
        wgts_x_middle = wgts_x[middle_segment_mask]

        # Create a tuple for indexing the middle segment weights arrays
        index_tuple = (v_idx_arr[middle_segment_mask], closest_index[middle_segment_mask])

        # Create weights for the middle joints, forming a peak in the middle of the joint 0.5 -> 1.0 -> 0.5 : /\
        sec0_weights = weights_lookup_min_to_max(wgts_x_middle, edge0, interp_type)
        sec1_weights = weights_lookup_max_to_min(wgts_x_middle, edge0, interp_type)

        # Get the inverse of each section to assign
        sec0_wgts_inv = 1 - sec0_weights
        sec1_wgts_inv = 1 - sec1_weights

        # The combined weights form the "peak" and are assigned to the closest influence for middle points
        combined_wgts = np.minimum(sec0_weights, sec1_weights)

        stacked_middle_wgts = np.column_stack((sec0_wgts_inv, combined_wgts, sec1_wgts_inv))

        if edge0 < 0.5:
            num_middle_pnts = combined_wgts.shape[0]
            falloff_index = int(num_middle_pnts * (1 - edge0))

            # Mult the sec0 (previous) influence weights
            sec0_inv_mult = np.zeros(num_middle_pnts)
            sec0_inv_mult[:falloff_index] = np.linspace(1, 0, falloff_index)
            sec0_order = np.argsort(wgts_x_middle)
            sec0_wgts_inv[sec0_order] *= sec0_inv_mult

            # Mult the sec1 (next) influence weights
            sec1_inv_mult = np.zeros(num_middle_pnts)
            sec1_inv_mult[int(num_middle_pnts - falloff_index):] = np.linspace(0, 1, falloff_index)
            sec1_order = np.argsort(wgts_x_middle)
            sec1_wgts_inv[sec1_order] *= sec1_inv_mult

            # Need to normalize if the falloff < 0.5 because simple linear blend of weights
            # will not add to 1 if we still want a "peak" in the middle influence.
            # For example if the falloff was 0, IE the weights from the previous and next influence
            # blended with the middle influence the entire length of the line, then that would sum to 0.5
            # leaving the middle joints weights flat along it's length.
            # I prefer the peaked shape, so doing the normalization because the "peak" is baked into
            # the weights_lookup functions.
            stacked_middle_wgts = np.column_stack((sec0_wgts_inv, combined_wgts, sec1_wgts_inv))
            stacked_middle_wgts = stacked_middle_wgts / np.sum(stacked_middle_wgts, axis=1).reshape(-1, 1)

        # Assign to the closest joint
        weights[index_tuple] = stacked_middle_wgts[:, 1]

        # Now have to assign the inverse to either the parent or child for the first half or second half of points
        # Just bump up/down the index on the columns (influence) index array
        weights[index_tuple[0], index_tuple[1] - 1] = stacked_middle_wgts[:, 0]
        weights[index_tuple[0], index_tuple[1] + 1] = stacked_middle_wgts[:, 2]

    return weights


def apply_weights(mesh, weights, affect_influences, ids=None, weights_pp=None):
    """
    Integrate the tube weight generated array into the existing mesh
    skinCluster weight array.

    Args:
        mesh (str): Name of the mesh.
        weights (np.ndarray): Numpy weight array returned by generate_weights.
        affect_influences (list/str): List of influences to apply weights on.
        ids (list/int): List of vertex indices to modify weights for, or None for all.
        weights_pp (list/float): List of per vertex weights used to
                                blend the generated weights with the original.

    Returns:

    """
    # Get weight data
    skin_data = get_skin_data(mesh, ids=ids)
    skinCl, influences, locked_influences, num_influences, object_weights = skin_data['skinCluster'], \
                                                                             skin_data['influences'], \
                                                                             skin_data['locked_influences'], \
                                                                             skin_data['num_influences'], \
                                                                             skin_data['n_weights']

    influence_indices = [influences.index(j) for j in affect_influences[:-1]]
    use_opacity = np.any(weights_pp != 1)

    if use_opacity:
        weights = blend_weight_arrays(object_weights[:, influence_indices],
                                      weights,
                                      weights_pp=weights_pp)

    # Assign the "tube" generated weights to those indices
    object_weights[:, influence_indices] = weights

    # Generate a mask for the inverse so we can apply 0 to any influences
    # other than the ones we are working with for the set of points we are affecting.
    other_influence_indices = np.ones(num_influences, np.bool)
    other_influence_indices[influence_indices] = 0

    # Assign all other influence weights to 0, or normalize them if we had selection weights_pp
    if use_opacity:
        object_weights = normalize_weights(object_weights, hold_influences=influence_indices)
    else:
        object_weights[:, other_influence_indices] = 0

    ids = ids or list(range(object_weights.shape[0]))

    set_weights(mesh, ids, object_weights)


def get_closest_line_points(points, line_points, include_last=False):
    """
    Find the closest point on a line segment for given points.

    Lines are created as vectors between the line_points.

    Returns the index of the closest line, the position of the point along that line,
    and the vectors of all the lines.

    Args:
        points (list/float3): List of xyz values for each point.
        line_points (list/float3): List of each point of the lines to check against, IE joint positions.
        include_last (bool):  Whether to consider tha last point as a valid result.

    Returns (tuple): Per point array of:
                        Array of closest line index (int)
                        Array of closest line points (float3)
                        Array of line vectors (float3)

    """
    points = np.array(points)
    line_points = np.array(line_points)

    num_lines = len(line_points)

    if not include_last:
        num_lines -= 1

    num_points = len(points)

    # Create line vectors
    lines = []
    for i in range(num_lines):
        if include_last:
            if i == (num_lines - 1):
                continue
        lines.append(
            [line_points[i + 1][0] - line_points[i][0],
             line_points[i + 1][1] - line_points[i][1],
             line_points[i + 1][2] - line_points[i][2]])

    if include_last:
        # make tiny line for the last point
        lines.append(
                [(line_points[-1][0] - line_points[-2][0]) * -0.0001,
                 (line_points[-1][1] - line_points[-2][1]) * -0.0001,
                 (line_points[-1][2] - line_points[-2][2]) * -0.0001]
            )

    if not include_last:
        line_points = line_points[:-1]

    # Arrange data for per point distance comparison to each line point ( nP, nJ )
    # This means we need an array of size num_points * num_lines
    points_repeated = np.repeat(points, num_lines, axis=0).reshape(num_points, num_lines, 3)

    line_vectors = np.array(lines)
    line_vectors_repeated = np.repeat(line_vectors[np.newaxis, :], num_points, axis=0)
    line_points_repeated = np.repeat(line_points[np.newaxis, :], num_points, axis=0)

    # the equation to find closest point on lines
    start_to_point = points_repeated - line_points_repeated
    line_lengths = np.linalg.norm(line_vectors_repeated, axis=2)
    atb2 = line_lengths ** 2

    dot = np.sum(line_vectors_repeated * start_to_point, axis=2)

    t = dot / atb2

    # clip the t value to remain inside the line segment.
    # if it was unbounded, the line would continue indefinitely
    t_normalized = np.clip(t.reshape(num_points, num_lines, -1), 0, 1)

    closest_line_points = line_points_repeated + (line_vectors_repeated * t_normalized)

    # Length of vector from our points to the calculated closest line points
    dist_to_lines = np.linalg.norm(points_repeated - closest_line_points, axis=2)

    # Closest line index will be the one that has the
    # shortest length from the point to the closest line point.
    closest_line_indices = np.where(np.min(dist_to_lines, axis=1).reshape(-1, 1) == dist_to_lines)

    # Since we could have 2 lines return as closest, remove duplicate indices.
    # Roll with -1 basically means prefer children when equal.  A value of 1 would mean prefer parent
    dups = np.where(closest_line_indices[0] == np.roll(closest_line_indices[0], -1))[0]

    # To determine which line is closest if we had equal distance,
    # check distance against a point nudged slightly along each of the 2 lines.
    # The shorter distance means this point is on the smaller side made by splitting the angle between the two lines
    dup_line_pairs = np.c_[closest_line_indices[1][dups], closest_line_indices[1][dups + 1]]
    dup_line_start_points = line_points[dup_line_pairs]
    dup_line_vectors = line_vectors[dup_line_pairs]

    # copy the line vectors where 2 points pointed to it
    nudged_line_points = dup_line_vectors.copy()

    # Nudge the first index (the parent)
    nudged_line_points[:, 0] = dup_line_start_points[:, 0] + (dup_line_vectors[:, 0] * 0.9)

    # Nudge the second index (the child)
    nudged_line_points[:, 1] = dup_line_start_points[:, 1] + (dup_line_vectors[:, 1] * 0.1)

    # Arrange an array of the points in pairs matching the shape of the nudged_line_points
    point_pairs = np.tile(points[dups - np.arange(dups.size)], 2).reshape(-1, 2, 3)

    # Calculate the distance to both nudged points, and pick the shorter
    dist_to_nudged = np.linalg.norm(nudged_line_points - point_pairs, axis=2)
    choice_of_pairs = np.where(dist_to_nudged == np.min(dist_to_nudged, axis=1).reshape(-1, 1))

    # closest_rows is just an array from 0 - num_points because every point has a closest line point
    closest_rows = np.delete(closest_line_indices[0], dups)

    # This is the array of the closest line segment per point
    closest_cols = np.delete(closest_line_indices[1], dups)

    # Insert the shorter of the two pairs we figured out above
    closest_cols[dups - np.arange(dups.size)] = dup_line_pairs[choice_of_pairs]

    return closest_cols, closest_line_points[closest_rows, closest_cols], line_vectors


def weights_lookup_max_to_min(x, edge0, interp='linear'):
    """
    Get the Y values progressing from low to high.

    Args:
        x (list/float): List of x axis values in 0-1 range.
        edge0 (float): Position in X where falloff begins.
        interp (str): "linear" or "smooth" falloff.

    Returns (list/float): List of Y values corresponding to the X values.

    """
    y_max = 0.5
    if interp == 'smooth':
        y = 1 - smoothstep(x, edge0, 1.0) * y_max
    else:
        y = 1 - linstep(x, edge0, 1.0) * y_max
    y -= 1 - (y_max * 2)
    return y


def weights_lookup_min_to_max(x, edge0, interp='linear'):
    """
    Get the Y values progressing from high to low.

    Args:
        x (list/float): List of x axis values in 0-1 range.
        edge0 (float): Position in X where falloff begins.
        interp (str): "linear" or "smooth" falloff.

    Returns (list/float): List of Y values corresponding to the X values.

    """
    y_max = 0.5
    if interp == 'smooth':
        y = y_max + smoothstep(x, 0.0, 1 - edge0) * y_max
    else:
        y = y_max + linstep(x, 0.0, 1 - edge0) * y_max
    return y

# -------------------------------------------------------------------------------------------------------------------- #
# Misc. supporting functions
# -------------------------------------------------------------------------------------------------------------------- #
def get_dag_path(node):
    """
    Return the MDagPath for a node.

    Args:
        node: (str) Object to retrieve the MDagPath for.

    Returns: OpenMaya.MDagPath

    """

    m_sel_list = OpenMaya.MSelectionList()
    m_sel_list.add(node)
    return m_sel_list.getDagPath(0)


def get_mfn_mesh(node):

    return OpenMaya.MFnMesh(get_dag_path(node))


def where_in(arrA, arrB):
    """
    Find the indices of where nArrayA occur in nArrayB.
    Maybe unnecessary but I find myself doing this all the time and looking it up :)

    Args:
        arrA (np.array): List of values to search for location of nArrayA values.
        arrB (np.array): List of values to search for occurrences of.

    Returns (np.array): Indices of where nArrayA Values are found in nArrayB.

    """

    orig_order = arrA.argsort()
    b_order = orig_order[np.searchsorted(arrA[orig_order], arrB)]
    return b_order


def linstep(x, edge0=0, edge1=1):
    if edge1 - edge0 == 0:
        return np.repeat(edge1, x.shape[0])
    return np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)


def smoothstep(x, edge0=0, edge1=1):
    x = linstep(x, edge0, edge1)
    return x * x * (3 - 2 * x)


def get_skin_data(mesh, ids=None):
    """
    Get the data required for weight manipulation operations.

    Args:
        mesh (str): Name of the object which has the skinCluster.
        ids (list/int): List of ids to operate on.

    Returns (dict): SkinCluster data.

    """

    influences = get_skin_influences(get_skincluster_from_object(mesh))
    num_influences = len(influences)
    skinCl = get_skincluster_from_object(mesh)
    n_weights = np.array(get_weights(mesh, ids=ids))
    n_weights = n_weights.reshape(int(n_weights.size / num_influences), num_influences).clip(0.0, 1.0)
    locked_influences = [i for i in range(num_influences) if cmds.getAttr('%s.liw' % influences[i])]

    return dict(influences=influences,
                num_influences=num_influences,
                locked_influences=locked_influences,
                skinCluster=skinCl,
                n_weights=n_weights)


def get_skin_influences(skinCluster):
    """
    Get the names of the influences from a skinCluster.

    Args:
        skinCluster (str): Name of the skinCluster.

    Returns (list/str): List of influence names.

    """
    mfn_skin = get_mfn_skin(skinCluster)
    return [i.partialPathName() for i in mfn_skin.influenceObjects()]


def get_skincluster_from_object(mesh):
    """
    Get the skinCluster from an object.

    Args:
        mesh (str): Name of the object.

    Returns (str): Name of the associated skinCluster.

    """
    shapes = cmds.listRelatives(mesh, s=True)
    if shapes:
        skinCls = cmds.findType(shapes[0], e=True, type='skinCluster')
        if skinCls:
            return skinCls[0]


def get_weights(mesh, ids=None):
    """
    Get skin weights through api.

    """
    shapes = get_shapes(mesh)
    if not shapes:
        return

    m_sel = OpenMaya.MSelectionList()
    m_sel.add(get_skincluster_from_object(mesh))
    skinCls_mObject = m_sel.getDependNode(0)
    mfn_skin = OpenMayaAnim.MFnSkinCluster(skinCls_mObject)
    shape_dag_path = get_dag_path(shapes[0])

    # Mesh objects
    mfn_component = OpenMaya.MFnSingleIndexedComponent()
    mfn_vert_component = mfn_component.create(OpenMaya.MFn.kMeshVertComponent)
    if not ids:
        m_iter_vertex = OpenMaya.MItMeshVertex(shape_dag_path)
        mfn_component.addElements(list(range(m_iter_vertex.count())))
    else:
        mfn_component.addElements(ids)

    m_weights = mfn_skin.getWeights(shape_dag_path,
                                   mfn_vert_component,
                                   OpenMaya.MIntArray(list(range(len(mfn_skin.influenceObjects())))))
    return m_weights


def get_mfn_skin(skinCluster):

    m_sel = OpenMaya.MSelectionList()
    m_sel.add(skinCluster)
    m_skin_depend = m_sel.getDependNode(0)
    return OpenMayaAnim.MFnSkinCluster(m_skin_depend)


def blend_weight_arrays(n_weightsA, n_weightsB, value=1.0, weights_pp=None):
    """
    Blend two 2d weight arrays with a global mult factor, and per point weight values.
    The incoming weights_pp should be a 1d array, as it's reshaped for the number of influences.

    Args:
        n_weightsA (np.array): Weight array to blend towards n_weightsB.
        n_weightsB (np.array): Target weight array to move n_weightsA towards.
        value (float): Global mult factor.
        weights_pp (list/float): Per point weight values.  This should be a 1d array.

    Returns (numpy.ndarray): Blended weights array.

    """
    if n_weightsA.shape != n_weightsB.shape:
        raise ValueError('Shape of both arrays must match: {}, {}'.format(n_weightsA.shape, n_weightsB.shape))

    weights_pp = weights_pp or np.ones(n_weightsA.shape[0])
    weights_pp = np.repeat(weights_pp, n_weightsA.shape[1]).reshape(-1, n_weightsA.shape[1]) * value
    n_weights = np_interp_by_weight(n_weightsA, n_weightsB, weights_pp)
    return n_weights


def np_interp_by_weight(nArrayA, nArrayB, n_weights):
    """
    Move all items of nArrayA towards values in nArrayB by step n_weights.

    Args:
        nArrayA (np.array): List of values to shift.
        nArrayB (np.array): List of values to shift towards.
        n_weights (np.array): List of 0-1 values for percentage of shift.

    Returns (np.array): List of shifted values.

    """

    if not nArrayA.size == nArrayB.size:
        raise Exception('All arrays need to be of the same size')

    nDiff = abs((nArrayA - nArrayB))
    nMultAr = np.ones(nArrayA.size).reshape(nArrayA.shape)
    nMultAr[nArrayA > nArrayB] = -1
    return nArrayA + (nDiff * n_weights) * nMultAr


def normalize_weights(n_weights, hold_influences=None):
    """
    Normalize weights for influences other than iInfluence.

    Args:
        n_weights (np.array): The numpy weight array.
        hold_influences (list/int): Indexes of the influences to not modify.

    Returns (np.array): The normalized weight array.

    """
    nWgts = n_weights.copy()
    # No weights outside 0-1 range
    nWgts = np.clip(nWgts, 0, 1)

    if not hold_influences:
        nWgts /= nWgts.sum(axis=1, keepdims=True)
    else:
        hold_influences = to_list(hold_influences)

        # First need to normalize any points where the weights of the influences to hold total > 1
        rows_to_norm = np.where(np.sum(nWgts[:, hold_influences], axis=1, keepdims=1) > 1)[0]
        nWgts[rows_to_norm.reshape(-1, 1), hold_influences] /= np.sum(nWgts[rows_to_norm.reshape(-1, 1), hold_influences],
                                                                     axis=1, keepdims=1)

        # Now we can normalize the rest
        remainder_weight = np.ones((nWgts.shape[0], 1)) - nWgts[:, hold_influences].sum(axis=1, keepdims=True)
        other_infs = np.delete(np.arange(nWgts.shape[1]), hold_influences)

        # Now distribute the remaining weights to the other influences
        hold_wgts_sums = np.sum(nWgts[:, other_infs], axis=1, keepdims=1)
        non_zero_row_mask = np.where(hold_wgts_sums > 0.00001)[0]
        zero_row_mask = np.where(hold_wgts_sums == 0.0)[0]
        other_wgts_sums = np.sum(nWgts[:, other_infs], axis=1, keepdims=True)

        other_wgts_normd = nWgts[non_zero_row_mask][:, other_infs] / other_wgts_sums[non_zero_row_mask]
        other_wgts_normd *= remainder_weight[non_zero_row_mask]

        # Set back into array
        nWgts[non_zero_row_mask.reshape(-1, 1), other_infs] = other_wgts_normd

        # This step for cases where the other infl's weight sum was 0, so instead of dividing we just
        # even split the remaining weight among the other influences
        if any(zero_row_mask):
            split_remainder = remainder_weight[zero_row_mask] / other_infs.size
            nWgts[zero_row_mask.reshape(-1, 1), other_infs] = \
                split_remainder.repeat(other_infs.size).reshape(-1, other_infs.size)

    return nWgts


def set_weights(mesh, ids, weights):
    """
    Set skin weights with cmds.
    
    NOTE: There are much faster and more optimized ways to do this.


    """
    skin = get_skincluster_from_object(mesh)
    for i, v_id in enumerate(ids):
        for inf_id in range(weights.shape[1]):
            cmds.setAttr('{}.weightList[{}].weights[{}]'.format(skin, v_id, inf_id), weights[i, inf_id])


def to_list(input):
    if not type(input) in [list, tuple]:
        return [input]
    return input


def get_shapes(mesh, include_intermediate=False):
    shapes = cmds.listRelatives(mesh, s=True, ni=not include_intermediate)
    return shapes