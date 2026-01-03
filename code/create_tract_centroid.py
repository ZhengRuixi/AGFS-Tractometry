#!/usr/bin/env python3
# -*- coding: utf-8 --

import os
import numpy as np
import vtkmodules.all as vtk
from concurrent.futures import ThreadPoolExecutor
from dipy.tracking.streamline import orient_by_streamline
import whitematteranalysis as wma

# ==================== Relative Paths ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        # Code directory
ROOT = os.path.dirname(SCRIPT_DIR)                             #  root directory
DATA_DIR = os.path.join(ROOT, "data", "ORG-atlas")

INPUT_BASE = os.path.join(DATA_DIR, "AnatomicalTracts_Separated")
OUTPUT_BASE = os.path.join(DATA_DIR, "ORG-atlas_centerline")

# Specify the tract to process here
TRACT_NAME = "AF_left"

input_vtk_path = os.path.join(INPUT_BASE, TRACT_NAME)
outdir_path    = os.path.join(OUTPUT_BASE, TRACT_NAME)
num_points = 100   
# ============================================================

os.makedirs(outdir_path, exist_ok=True)

def _fiber_distance_internal_use(fiber_r, fiber_a, fiber_s, fiber_array):
    fiber_array_r = fiber_array[:, :, 0]
    fiber_array_a = fiber_array[:, :, 1]
    fiber_array_s = fiber_array[:, :, 2]
    dx = np.square(fiber_array_r - fiber_r)
    dy = np.square(fiber_array_a - fiber_a)
    dz = np.square(fiber_array_s - fiber_s)
    distance = np.sum(np.sqrt(dx + dy + dz), 1)
    npts = float(fiber_array.shape[1])
    return distance / npts

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

def compute_centerline_in_cluster(fiber_array):
    tmp_r, tmp_s = np.shape(fiber_array.fiber_array_r)
    x_array_orig = np.zeros((tmp_r, tmp_s, 3))
    x_array_orig[:, :, 0] = fiber_array.fiber_array_r
    x_array_orig[:, :, 1] = fiber_array.fiber_array_a
    x_array_orig[:, :, 2] = fiber_array.fiber_array_s
    x_array_quiv = np.flip(x_array_orig, axis=1)
    num_fibers = x_array_orig.shape[0]

    if num_fibers == 0:
        return None
    elif num_fibers == 1:
        return x_array_orig[0]

    dis_sum = np.zeros(num_fibers)
    dis_arg_list = np.zeros((num_fibers, num_fibers))

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            lambda idx: proceed_fiber_in_parallel(idx, x_array_orig, x_array_quiv),
            range(num_fibers)
        ))

    for idx, (dis_arg, dis) in enumerate(results):
        dis_arg_list[idx, :] = dis_arg
        dis_sum[idx] = dis

    center_idx = np.argmin(dis_sum)
    reordered = dis_arg_list[center_idx, :]
    x_orig = x_array_orig[np.where(reordered == 0)]
    x_quiv = x_array_quiv[np.where(reordered == 1)]
    x_all = np.concatenate((x_orig, x_quiv))
    return np.mean(x_all, axis=0)   

def convert_to_polydata(fiber_array):
    outpd = vtk.vtkPolyData()
    outpoints = vtk.vtkPoints()
    outlines = vtk.vtkCellArray()
    fiber_array = np.array(fiber_array)
    number_of_fibers = fiber_array.shape[0]
    points_per_fiber = fiber_array.shape[1]
    for lidx in range(number_of_fibers):
        cellptids = vtk.vtkIdList()
        for pidx in range(points_per_fiber):
            idx = outpoints.InsertNextPoint(
                fiber_array[lidx, pidx, 0],
                fiber_array[lidx, pidx, 1],
                fiber_array[lidx, pidx, 2]
            )
            cellptids.InsertNextId(idx)
        outlines.InsertNextCell(cellptids)
    outpd.SetLines(outlines)
    outpd.SetPoints(outpoints)
    return outpd

# ============================= Main Process =============================
print(f"\n=== Processing tract: {TRACT_NAME} ===")

centerlines, cluster_names = [], []

for cluster_name in os.listdir(input_vtk_path):
    cluster_path = os.path.join(input_vtk_path, cluster_name)
    if not os.path.isfile(cluster_path):
        continue
    print(f"  -> Cluster: {cluster_name}")
    try:
        inpd = wma.io.read_polydata(cluster_path)
    except Exception as e:
        print(f"     Cannot read {cluster_name}, skipping ({e})")
        continue

    fiber_array = wma.fibers.FiberArray()
    fiber_array.convert_from_polydata(inpd, num_points)

    centerline = compute_centerline_in_cluster(fiber_array)
    if centerline is not None:
        centerlines.append(centerline)
        cluster_names.append(cluster_name)

if not centerlines:
    print("No valid centerlines generated!")
else:
    print("Orienting all centerlines consistently...")
    oriented_centerlines = orient_by_streamline(centerlines, centerlines[0])

    for idx, centerline in enumerate(oriented_centerlines):
        pd_centerline = convert_to_polydata([centerline])
        out_file = os.path.join(outdir_path, 'center_' + cluster_names[idx])
        wma.io.write_polydata(pd_centerline, out_file)

    print(f"Done: {TRACT_NAME} → {len(oriented_centerlines)} centerlines saved to")
    print(f"{outdir_path}")
