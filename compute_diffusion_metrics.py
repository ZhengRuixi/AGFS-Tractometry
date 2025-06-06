from create_tract_centroid import *
from extract_tract_neighborhood import *
import argparse
from numpy import linalg as LA
import numpy as np
import os
import whitematteranalysis as wma


def mahalanobis(u, v, c_reverse):
    delta = u - v
    m = np.dot(np.dot(delta, c_reverse), delta)
    return np.sqrt(m)

def weighted_diffusion_metric(centroid, assigned_ids, points, vals):
    res_list = []
    for i in range(len(centroid)):
        points_to_node = points[assigned_ids == i]
        vals_to_node = vals[assigned_ids == i]
        if len(points_to_node) < 3:
            res = np.nan
        else:
            c = np.cov(points_to_node.T, ddof=1)
            if np.isclose(LA.det(c), 0):
                res = np.nan
            else:
                w = np.zeros(len(points_to_node))
                for j in range(len(points_to_node)):
                    w[j] = 1 / mahalanobis(points_to_node[j], centroid[i], np.linalg.inv(c))
                    w = w / sum(w, 0)
                    res = np.dot(vals_to_node, w)
        res_list.append(res)
    return np.array(res_list)
    

# Take one subject as an example
if __name__ == '__main__':
    # ----------------
    # Parse arguments
    # ----------------
    parser = argparse.ArgumentParser(description="Compute weighted diffusion metric for each subject.")

    parser.add_argument('inputDir', help='input vtk files as folder.')
    parser.add_argument('outputDir', help='The output directory should be a new empty directory. It will be created if needed.')
    parser.add_argument('-numPoints', action='store', type=int, default=100, help='Number of points per fiber to compute centroid.')

    args = parser.parse_args()
    script_name = '<compute_diffusion_metric>'
    if not os.path.exists(args.inputDir):
        print(script_name, "Error: Input directory ", args.inputDir, "does not exist.")
        exit()

    if not os.path.exists(args.outputDir):
        print(script_name, "Output directory", args.outputDir, "does not exist, creating it.")
        os.makedirs(args.outputDir)

    sub_res = []
    for cluster in os.listdir(args.inputDir):
        inpath = os.path.join(args.inputDir, cluster)
        if os.path.getsize(inpath) / 1024 > 5:
            inpd = wma.io.read_polydata(os.path.join(args.inputDir, cluster))
            # compute centroid
            fiber_array = wma.fibers.FiberArray()
            fiber_array.convert_from_polydata(inpd, args.numPoints)
            centroid = compute_centroid_in_cluster(fiber_array)

            # compute weighted diffusion metric
            points = extract_vtk_data(inpd, 'point')
            values = extract_vtk_data(inpd, 'FA')
            assigned_ids = assign_to_nearest_centroid(points, centroid)
            res = weighted_diffusion_metric(centroid, assigned_ids, points, values)
        else:
            res = np.full(len(centroid), np.nan)
        sub_res.extend(res)
    
    print(f'Take one subject as an example (FA):\n {sub_res}')

     
    
    



        


