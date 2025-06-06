from create_tract_centroid import *
import argparse
from numpy import linalg as LA
import numpy as np
import os
import vtk
import whitematteranalysis as wma


def extract_vtk_data(inpd, type):
    if not isinstance(inpd, vtk.vtkPolyData):
        raise TypeError('Input data must be of type vtk.vtkPolyData')
    type = type.lower()
    if type not in ['point', 'fa']:
        raise ValueError('Unsupported vtk data: {}'.format(type))
    
    # loop over lines
    inpd.GetLines().InitTraversal()
    line_ptids = vtk.vtkIdList()

    # Initialize containers
    points, FA = [], []  

    for lidx in range(inpd.GetNumberOfLines()):
        inpd.GetLines().GetNextCell(line_ptids)
        for pidx in range(line_ptids.GetNumberOfIds()):
            ptidx = line_ptids.GetId(pidx)
            # extract vtk data
            if type == 'point':
                point = inpd.GetPoints().GetPoint(ptidx)
                points.append(point)

            elif type == 'fa':
                tensor = inpd.GetPointData().GetArray("tensor1")
                if tensor is None:
                    raise ValueError("Array 'tensor1' not found in vtk data")
                
                val = tensor.GetTuple(ptidx)
                val_mat = np.array([[val[0], val[1], val[2]], 
                                    [val[3], val[4], val[5]],
                                    [val[6], val[7], val[8]]])
                evals, _ = LA.eig(val_mat)
                # Make sure not to get nans
                all_zero = (evals == 0).all(axis=0)
                ev1, ev2, ev3 = evals
                fa = np.sqrt(0.5 * ((ev1 - ev2) ** 2 +
                                        (ev2 - ev3) ** 2 +
                                        (ev3 - ev1) ** 2) /
                                ((evals * evals).sum(0) + all_zero))
                FA.append(fa)
            
    if type == 'point':
        return np.array(points)
    elif type == 'fa':
        return np.array(FA)     


def assign_to_nearest_centroid(points, centroid):
    dists = np.linalg.norm(points[:, np.newaxis] - centroid, axis=2)
    assigned_ids = np.argmin(dists, axis=1)
    return assigned_ids


def calc_mean_dist_no_outliers(assigned_ids, centroid, points):
    radius = []
    for i in range(len(centroid)):
        node = centroid[i]  # central node
        indices = [row for row, value in enumerate(assigned_ids) if value == i]
        node_parcel = points[indices]

        dis = LA.norm(node_parcel - node, axis=1)
        mean_dis = np.mean(dis)
        std_dis = np.std(dis)

        # filter out outliers
        filtered_node_parcel = node_parcel[dis <= mean_dis + 2 * std_dis]
        filtered_node_parcel_radius = np.mean(LA.norm(filtered_node_parcel - node, axis=1))
        radius.append(filtered_node_parcel_radius)

    return np.array(radius)


if __name__ == '__main__':
    # ----------------
    # Parse arguments
    # ----------------
    parser = argparse.ArgumentParser(description="Compute neighborhood relationships of vtk file.")

    parser.add_argument('inputDir', help='input vtk files as folder.')
    parser.add_argument('outputDir', help='The output directory should be a new empty directory. It will be created if needed.')
    parser.add_argument('-numPoints', action='store', type=int, default=100, help='Number of points per fiber to compute centroid.')

    args = parser.parse_args()
    script_name = '<extract_tract_neighborhood>'
    if not os.path.exists(args.inputDir):
        print(script_name, "Error: Input directory ", args.inputDir, "does not exist.")
        exit()

    if not os.path.exists(args.outputDir):
        print(script_name, "Output directory", args.outputDir, "does not exist, creating it.")
        os.makedirs(args.outputDir)

    centroids, radii = [], []
    for cluster in os.listdir(args.inputDir):
        inputVTK = os.path.join(args.inputDir, cluster)
        inpd = wma.io.read_polydata(inputVTK)

        # compute centroid
        fiber_array = wma.fibers.FiberArray()
        fiber_array.convert_from_polydata(inpd, args.numPoints)
        centroid = compute_centroid_in_cluster(fiber_array)

        # compute node parcel radius
        points = extract_vtk_data(inpd, 'point')
        assigned_ids = assign_to_nearest_centroid(points, centroid)
        radius = calc_mean_dist_no_outliers(assigned_ids, centroid, points)

        centroids.extend(centroid)
        radii.extend(radius)

    centroids = np.array(centroids)
    radii = np.array(radii)
    # extract neighborhood matrix
    dists = LA.norm(centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=-1)
    neighborhood = (dists < (radii[:, np.newaxis] + radii[np.newaxis, :])).astype(int)
    # update intra-cluster neighborhood relationships
    count = 0
    while count < len(centroids):
        neighborhood[count:count+args.numPoints, count:count+args.numPoints] = \
            np.eye(args.numPoints, dtype=int, k=1) + \
            np.eye(args.numPoints, dtype=int, k=-1)
        count += args.numPoints
    np.fill_diagonal(neighborhood, 0)
    neighborhood[neighborhood > 1] = 1
    np.save(os.path.join(args.outputDir, 'Neighborhood_' + os.path.basename(args.inputDir)), neighborhood)


