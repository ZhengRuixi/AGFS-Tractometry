import numpy as np
import os
from numpy import linalg as LA
import whitematteranalysis as wma
import vtkmodules.all as vtk
import networkx as nx
from scipy.sparse import lil_matrix
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from scipy.sparse import csr_matrix
from collections import defaultdict


def _fiber_distance_internal_use(fiber_r, fiber_a, fiber_s, fiber_array):
    """
    Compute the total fiber distance from one fiber to an array of many fibers.
    The number of points along every fiber must be the same.
    """

    fiber_array_r = fiber_array[:, :, 0]
    fiber_array_a = fiber_array[:, :, 1]
    fiber_array_s = fiber_array[:, :, 2]

    # compute the distance from this fiber to the array of other fibers
    dx = np.square(fiber_array_r - fiber_r)
    dy = np.square(fiber_array_a - fiber_a)
    dz = np.square(fiber_array_s - fiber_s)

    # sum dx dx dz at each point on the fiber and sqrt for threshold
    distance = np.sum(np.sqrt(dx + dy + dz), 1)

    # Remove effect of number of points along fiber (mean)
    npts = float(fiber_array.shape[1])
    distance = distance / npts

    return distance


def proceed_fiber_in_parallel(f_idx, x_array_orig, x_array_quiv):
    fiber_array = x_array_orig[f_idx, :]

    dis_orig = _fiber_distance_internal_use(
        fiber_array[:, 0], fiber_array[:, 1], fiber_array[:, 2], x_array_orig)
    dis_quiv = _fiber_distance_internal_use(
        fiber_array[:, 0], fiber_array[:, 1], fiber_array[:, 2], x_array_quiv)

    dis_tmp = np.stack((dis_orig, dis_quiv), axis=0)
    dis_min = np.min(dis_tmp, axis=0)
    dis_arg = np.argmin(dis_tmp, axis=0)

    return dis_arg, np.sum(dis_min)


def compute_centroid_in_cluster(fiber_array):
    tmp_r, tmp_s = np.shape(fiber_array.fiber_array_r)
    x_array_orig = np.zeros((tmp_r, tmp_s, 3))
    x_array_orig[:, :, 0] = fiber_array.fiber_array_r
    x_array_orig[:, :, 1] = fiber_array.fiber_array_a
    x_array_orig[:, :, 2] = fiber_array.fiber_array_s
    x_array_quiv = np.flip(x_array_orig, axis=1)
    num_fibers = x_array_orig.shape[0]

    print(f"Number of fibers: {num_fibers}")

    if num_fibers == 0:
        centroid = None
        reordered_fibers = np.array([])

    elif num_fibers == 1:
        centroid = x_array_orig
        reordered_fibers = np.array([0.0])
        print("Only one fiber, centroid is the fiber itself.")

    else:
        dis_sum = np.zeros(num_fibers)
        dis_arg_list = np.zeros((num_fibers, num_fibers))

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda f_idx: proceed_fiber_in_parallel(f_idx, x_array_orig, x_array_quiv),
                                        range(num_fibers)))

        for f_idx, (dis_arg, dis) in enumerate(results):
            dis_arg_list[f_idx, :] = dis_arg
            dis_sum[f_idx] = dis

        # reordered_fibers represents the direction for each streamline
        center_fiber_idx = np.argmin(dis_sum)
        print(f"Center fiber index: {center_fiber_idx} with min distance: {dis_sum[center_fiber_idx]}")

        reordered_fibers = dis_arg_list[center_fiber_idx, :]

        x_array_orig_ = x_array_orig[np.where(reordered_fibers == 0)]
        x_array_quiv_ = x_array_quiv[np.where(reordered_fibers == 1)]

        # save streamlines in the same direction
        x_array_reodered = np.concatenate((x_array_orig_, x_array_quiv_))
        centroid = np.mean(x_array_reodered, axis=0)

        print("Centroid calculated")

    return centroid


def assign_points_to_nearest_centroid(points, centroid):  # optimized
    dists = np.linalg.norm(points[:, np.newaxis] - centroid, axis=2)
    assigned_id = np.argmin(dists, axis=1)
    return assigned_id


def remove_outliers_and_compute_mean_distance(assigned_id, centroid, points):
    radius = []
    for i in range(len(centroid)):
        i_centroid = centroid[i]
        indices = [row for row, value in enumerate(assigned_id) if value == i]
        i_points = points[indices]

        dis = LA.norm(i_points - i_centroid, axis=1)
        mean_dis = np.mean(dis)
        std_dis = np.std(dis)

        filtered_i_points = i_points[dis <= mean_dis + 2 * std_dis]
        i_radius = np.mean(LA.norm(filtered_i_points - i_centroid, axis=1))
        radius.append(i_radius)

    radius = np.array(radius)
    return radius


# functions used in the method we proposed
def get_info_from_vtk_or_vtp(inpd, info_type):
    points, vals = [], []
    if info_type == 'point':
        inpd.GetLines().InitTraversal()
        inpoints = inpd.GetPoints()
        line_ptids = vtk.vtkIdList()
        for lidx in range(inpd.GetNumberOfLines()):
            inpd.GetLines().GetNextCell(line_ptids)
            for pidx in range(line_ptids.GetNumberOfIds()):
                ptidx = line_ptids.GetId(pidx)
                point = inpoints.GetPoint(ptidx)
                points.append(point)
        return np.array(points)

    elif info_type == 'FA':
        inpd.GetLines().InitTraversal()
        inpointdata = inpd.GetPointData()
        array = inpointdata.GetArray('color_index')
        line_ptids = vtk.vtkIdList()
        for lidx in range(inpd.GetNumberOfLines()):
            inpd.GetLines().GetNextCell(line_ptids)
            for pidx in range(line_ptids.GetNumberOfIds()):
                ptidx = line_ptids.GetId(pidx)
                val = array.GetValue(ptidx)
                vals.append(val)
        return np.array(vals)

    else:
        raise ValueError('Unsupported info_type: {}'.format(info_type))


def mahalanobis(u, v, c_reverse):
    delta = u - v
    m = np.dot(np.dot(delta, c_reverse), delta)
    return np.sqrt(m)


def weighted_values_to_centroid(center_line, assigned_id, points, values):
    weighted_values = []
    for i in range(len(center_line)):
        points_to_centroid = points[np.where(assigned_id == i)]
        values_to_centroid = values[np.where(assigned_id == i)]

        if len(points_to_centroid) < 3:
            value = np.nan
        else:
            c = np.cov(points_to_centroid.T, ddof=1)

            if np.isclose(np.linalg.det(c), 0):
                value = np.nan
            else:
                w = np.zeros(len(points_to_centroid))
                for j in range(len(points_to_centroid)):
                    w[j] = 1 / mahalanobis(points_to_centroid[j], center_line[i], np.linalg.inv(c))
                    w = w / sum(w, 0)
                    value = np.dot(values_to_centroid, w)

        weighted_values.append(value)

    weighted_values = np.array(weighted_values)

    return weighted_values


def convert_matrix_to_graph(matrix):
   g = nx.Graph()
   matrix = np.array(matrix)
   rows, cols = np.where(matrix == 1)
   g.add_edges_from(zip(rows, cols))
   return g


def get_percolated_cliques(cliques, k):
   # Find all cliques >= k
   cliques = [frozenset(clique) for clique in nx.find_cliques(g) if len(clique) >= k]
   n = len(cliques)
   # Build overlap matrix
   overlap_matrix = lil_matrix((n, n), dtype=int)
   print(f'overlap_matrix: {overlap_matrix.shape}')
   for i in range(n):
       for j in range(i + 1, n):  # Start from i+1 to avoid duplicate calculations
           intersection = len(cliques[i].intersection(cliques[j]))
           if intersection >= 2:  # original:k-1
               overlap_matrix[i, j] = overlap_matrix[j, i] = 1

   # Record the community ID
   community_ids = list(range(n))
   for i in range(n):
       for j in range(i + 1, n):
           if overlap_matrix[i, j] == 1:
               community_ids[j] = community_ids[i]

   # Find unique community IDs
   unique_ids = list(set(community_ids))

   # Build communities
   communities = []
   for unique_id in unique_ids:
       community = set()
       for idx, id in enumerate(community_ids):
           if id == unique_id:
               community.update(cliques[idx])
       communities.append(frozenset(community))

   # Sort communities by size
   communities.sort(key=len, reverse=True)

   return [list(community) for community in communities]


def convert_to_polydata(fiber_array):
    outpd = vtk.vtkPolyData()
    outpoints = vtk.vtkPoints()
    outlines = vtk.vtkCellArray()

    fiber_array = np.array(fiber_array)

    fiber_array_r = fiber_array[:, :, 0]
    fiber_array_a = fiber_array[:, :, 1]
    fiber_array_s = fiber_array[:, :, 2]
    number_of_fibers = fiber_array.shape[0]
    points_per_fiber = fiber_array.shape[1]
    outlines.InitTraversal()

    for lidx in range(0, number_of_fibers):
        cellptids = vtk.vtkIdList()

        for pidx in range(0, points_per_fiber):
            idx = outpoints.InsertNextPoint(fiber_array_r[lidx, pidx],
                                            fiber_array_a[lidx, pidx],
                                            fiber_array_s[lidx, pidx])

            cellptids.InsertNextId(idx)

        outlines.InsertNextCell(cellptids)

    # put data into output polydata
    outpd.SetLines(outlines)
    outpd.SetPoints(outpoints)

    return outpd


def write_vtk_vtp(lines_points, output_filename, additional_array):
    outpd = vtk.vtkPolyData()
    outpoints = vtk.vtkPoints()
    outlines = vtk.vtkCellArray()

    new_pointdata_array = vtk.vtkFloatArray()
    new_pointdata_array.SetName('color_index')

    outlines.InitTraversal()  # Initialization before traversal operation

    for lidx in range(0, len(lines_points)):
        cellptids = vtk.vtkIdList()

        for pidx in range(0, len(lines_points[lidx])):
            idx = outpoints.InsertNextPoint(lines_points[lidx][pidx][0],
                                            lines_points[lidx][pidx][1],
                                            lines_points[lidx][pidx][2])

            cellptids.InsertNextId(idx)

            new_pointdata_array.InsertNextValue(additional_array[lidx][pidx])

        outlines.InsertNextCell(cellptids)

    # put data into output polydata
    outpd.SetLines(outlines)
    outpd.SetPoints(outpoints)
    outpd.GetPointData().AddArray(new_pointdata_array)

    basename, extension = os.path.splitext(output_filename)
    if extension == '.vtk':
        writer = vtk.vtkDataSetWriter()
    elif extension == '.vtp':
        writer = vtk.vtkXMLPolyDataWriter()
    else:
        ValueError('Output file name error.')
    writer.SetFileName(output_filename)
    writer.SetInputData(outpd)
    writer.Write()


def rewrite_vtk_vtp(vtk_file, outpath, white_noise=False, significant_difference=False, center=None, radius=None):
    inpd = wma.io.read_polydata(vtk_file)
    inpd.GetLines().InitTraversal()
    inpoints = inpd.GetPoints()
    inpointdata = inpd.GetPointData()
    line_ptids = vtk.vtkIdList()

    lines, new_array = [], []
    for lidx in range(inpd.GetNumberOfLines()):
        line, metric = [], []
        inpd.GetLines().GetNextCell(line_ptids)

        for pidx in range(line_ptids.GetNumberOfIds()):
            ptidx = line_ptids.GetId(pidx)
            point = inpoints.GetPoint(ptidx)
            line.append(point)

            tensor = inpointdata.GetArray("tensor1")
            val = tensor.GetTuple(ptidx)
            val_mat = np.array([[val[0], val[1], val[2]], [val[3], val[4], val[5]], [val[6], val[7], val[8]]])
            evals, _ = LA.eig(val_mat)
            # Make sure not to get nans
            all_zero = (evals == 0).all(axis=0)
            ev1, ev2, ev3 = evals
            fa = np.sqrt(0.5 * ((ev1 - ev2) ** 2 +
                                (ev2 - ev3) ** 2 +
                                (ev3 - ev1) ** 2) /
                         ((evals * evals).sum(0) + all_zero))

            if not white_noise and not significant_difference:
                metric.append(fa)
            elif white_noise and not significant_difference:  # add white noise, default: weight=0.01
                fa = fa + 0.01 * np.random.randn()
                metric.append(fa)
            elif not white_noise and significant_difference:  # add significant difference to points in the specific ROI
                if LA.norm(point - center) <= radius:
                    fa = 1.5 * fa
                # for i in range(len(center)):
                #     if LA.norm(point - center[i]) <= radius[i]:
                #         fa = fa + 0.3
                #         break
                metric.append(fa)

        lines.append(np.array(line))
        new_array.append(np.array(metric))

    write_vtk_vtp(lines_points=lines, output_filename=outpath, additional_array=new_array)






