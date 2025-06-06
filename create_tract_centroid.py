import numpy as np
import os
import whitematteranalysis as wma
import vtkmodules.all as vtk
import argparse
from concurrent.futures import ThreadPoolExecutor


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

    # print(f"Number of fibers: {num_fibers}")

    if num_fibers == 0:
        centroid = None
        reordered_fibers = np.array([])

    elif num_fibers == 1:
        centroid = x_array_orig
        reordered_fibers = np.array([0.0])
        # print("Only one fiber, centroid is the fiber itself.")

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
        # print(f"Center fiber index: {center_fiber_idx} with min distance: {dis_sum[center_fiber_idx]}")

        reordered_fibers = dis_arg_list[center_fiber_idx, :]

        x_array_orig_ = x_array_orig[np.where(reordered_fibers == 0)]
        x_array_quiv_ = x_array_quiv[np.where(reordered_fibers == 1)]

        # save streamlines in the same direction
        x_array_reodered = np.concatenate((x_array_orig_, x_array_quiv_))
        centroid = np.mean(x_array_reodered, axis=0)

    return centroid


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


if __name__ == '__main__':
    # ----------------
    # Parse arguments
    # ----------------
    parser = argparse.ArgumentParser(description="Compute centroid of input vtk file.")

    parser.add_argument('inputVTK', help='input tractography data as vtkPolyData file(s).')
    parser.add_argument('outputDir', help='The output directory should be a new empty directory. It will be created if needed.')
    parser.add_argument('-numPoints', action='store', type=int, default=100, help='Number of points per fiber to compute centroid.')

    args = parser.parse_args()
    script_name = '<create_tract_centroid>'
    if not os.path.exists(args.inputVTK):
        print(script_name, "Error: Input tractography ", args.inputVTK, "does not exist.")
        exit()

    if not os.path.exists(args.outputDir):
        print(script_name, "Output directory", args.outputDir, "does not exist, creating it.")
        os.makedirs(args.outputDir)

    print(script_name, 'Reading input tractography:', args.inputVTK)
    inpd = wma.io.read_polydata(args.inputVTK)
    fiber_array = wma.fibers.FiberArray()
    fiber_array.convert_from_polydata(inpd, args.numPoints)
    centroid = compute_centroid_in_cluster(fiber_array)
    pd_centroid = convert_to_polydata([centroid])
    wma.io.write_polydata(pd_centroid, os.path.join(args.outputDir, 'centroid_' + os.path.basename(args.inputVTK)))



