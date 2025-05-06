
import numpy as np
from fswma import *
import vtkmodules.all as vtk
import whitematteranalysis as wma
from dipy.tracking.streamline import orient_by_streamline

# functions use in AFQ
def _calculate_line_indices(input_line_length, output_line_length):
    # this is the increment between output points
    step = (input_line_length - 1.0) / (output_line_length - 1.0)
    # these are the output point indices (0-based)
    ptlist = []
    for ptidx in range(0, output_line_length):
        ptlist.append(ptidx * step)
    # test
    if __debug__:
        # this tests we output the last point on the line
        # test = ((output_line_length - 1) * step == input_line_length - 1)
        test = (round(ptidx * step) == input_line_length - 1)
        if not test:
            print("<fibers.py> ERROR: fiber numbers don't add up.")
            print(step)
            print(input_line_length)
            print(output_line_length)
            print(test)
            raise AssertionError

    return ptlist


def _validate_vector(u, dtype=None):
    # XXX Is order='c' really necessary?
    u = np.asarray(u, dtype=dtype, order='c')
    if u.ndim == 1:
        return u
    raise ValueError("Input vector should be 1-D.")


def mahalanobis(u, v, VI):
    u = _validate_vector(u)
    v = _validate_vector(v)
    VI = np.atleast_2d(VI)
    delta = u - v
    m = np.dot(np.dot(delta, VI), delta)
    return np.sqrt(m)


def gaussian_weights(streamlines, num):

    streamlines = np.array(streamlines)
    w = np.zeros((len(streamlines), num))

    for i in range(num):
        node_coords = streamlines[:, i]
        # Reorganize as an upper diagonal matrix for expected Mahalanobis
        c = np.triu(np.cov(node_coords.T, ddof=0))
        # calculate the mean value
        m = np.mean(node_coords, 0)
        
        if np.linalg.det(c) == 0:
            for j in range(len(streamlines)):
                w[j, i] = 1 / mahalanobis(node_coords[j], m, np.linalg.pinv(c))
                
        else:
            for j in range(len(streamlines)):
                w[j, i] = 1 / mahalanobis(node_coords[j], m, np.linalg.inv(c))

    w = w / np.sum(w, 0)

    return w


def implement_afq_algorithm(vtp_file, center_line):
    inpd = wma.io.read_polydata(vtp_file)
    inpd.GetLines().InitTraversal()
    inpointdata = inpd.GetPointData()
    array = inpointdata.GetArray('color_index')
    line_ptids = vtk.vtkIdList()

    streamlines, vals = [], []
    for lidx in range(inpd.GetNumberOfLines()):
        inpd.GetLines().GetNextCell(line_ptids)
        line_length = line_ptids.GetNumberOfIds()

        streamline, val = np.zeros([0, 3]), np.zeros(0)
        for line_index in _calculate_line_indices(line_length, len(center_line)):
            ptidx = line_ptids.GetId(int(round(line_index)))
            streamline = np.vstack((streamline, np.array(inpd.GetPoint(ptidx))))
            val = np.append(val, array.GetValue(ptidx))

        streamlines.append(streamline)
        vals.append(val)

    rec_streamlines = orient_by_streamline(streamlines, center_line)
    rec_ids = []
    for s_i, s_j in zip(rec_streamlines, streamlines):
        if np.array_equal(s_i, s_j):
            rec_ids.append(1)  # the direction is same
        else:
            rec_ids.append(0)  # the direction is inverse

    # rewrite the direction of vals by rec_id
    for i, rec_id in enumerate(rec_ids):
        if rec_id == 0:
            vals[i] = vals[i][::-1]

    weights = gaussian_weights(rec_streamlines, len(center_line))
    weighted_val = np.sum(weights * vals, 0)

    return weighted_val


def divide_array(array, afq_significant_pts):
    # compute the length of each segment
    num_segments = len(afq_significant_pts)
    segment_length = len(array) // num_segments

    # initialize the segmented array
    segmented_array = np.zeros_like(array)

    # assign color parameters to each segment
    for i in range(num_segments):
        if afq_significant_pts[i] == 1:
            segmented_array[i*segment_length:(i+1)*segment_length] = 1
        else:
            segmented_array[i*segment_length:(i + 1)*segment_length] = 0

    # assign the remaining elements(if any) to the last segment
    if afq_significant_pts[-1] == 1:
        segmented_array[num_segments * segment_length:] = 1
    else:
        segmented_array[num_segments * segment_length:] = 0

    return segmented_array


# functions used in buan
def implement_buan_algorithm(vtp_file, center_line):
    inpd = wma.io.read_polydata(vtp_file)
    inpd.GetLines().InitTraversal()
    inpointdata = inpd.GetPointData()
    inpoints = inpd.GetPoints()
    array = inpointdata.GetArray('color_index')
    line_ptids = vtk.vtkIdList()

    pts, vals = [], []  # read all necessary information
    for lidx in range(inpd.GetNumberOfLines()):
        inpd.GetLines().GetNextCell(line_ptids)
        for pidx in range(line_ptids.GetNumberOfIds()):
            ptidx = line_ptids.GetId(pidx)
            pt = inpoints.GetPoint(ptidx)
            val = array.GetValue(ptidx)
            pts.append(pt)
            vals.append(val)

    # assign all points to the nearest centroid
    pt_num = len(pts)
    centroid_num = len(center_line)
    dis = np.zeros((centroid_num, pt_num))
    for i in range(centroid_num):
        for j in range(pt_num):
            dis[i, j] = LA.norm(pts[j] - center_line[i])

    assigned_id = np.argmin(dis, axis=0)  # assigned info
    grouped_vals = []
    for i in range(centroid_num):
        temp = []
        for group, val in zip(assigned_id, vals):
            if group == i:
                temp.append(val)
        grouped_vals.append(temp)

    return grouped_vals


# compute union set
def compute_intersection(A, B):
    # Convert A and B to set type
    set_A = set(tuple(point) for point in A)
    set_B = set(tuple(point) for point in B)
    # Compute intersection
    intersection = set_A & set_B
    # Convert intersection back to list type
    intersection_list = [list(point) for point in intersection]

    return intersection_list


