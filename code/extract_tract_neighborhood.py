#!/usr/bin/env python3
# -*- coding: utf-8 --


import os
import re
import numpy as np
from numpy import linalg as LA
import vtkmodules.all as vtk
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import whitematteranalysis as wma

# ==================== Relative Paths ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        # Code directory
ROOT = os.path.dirname(SCRIPT_DIR)                             # root directory
DATA_DIR = os.path.join(ROOT, "data", "ORG-atlas")

indir1 = os.path.join(DATA_DIR, "AnatomicalTracts_Separated")  # Cluster directory
indir2 = os.path.join(DATA_DIR, "ORG-atlas_centerline")        # Center directory
outdir = os.path.join(ROOT, "neighborhood")                     # Output directory
# ============================================================

# Only process AF_left
TRACT_NAME = "AF_left"

def extract_vtk_data(inpd, data_type):
    """Extract point coordinates or FA values"""
    if data_type == 'pt':
        inpd.GetLines().InitTraversal()
        inpoints = inpd.GetPoints()
        line_ptids = vtk.vtkIdList()
        pts = []
        while inpd.GetLines().GetNextCell(line_ptids):
            for i in range(line_ptids.GetNumberOfIds()):
                pt = inpoints.GetPoint(line_ptids.GetId(i))
                pts.append(pt)
        return np.array(pts)
   
    elif data_type == 'fa':
        fa_values = []
        inpoints = inpd.GetPoints()
        for i in range(inpoints.GetNumberOfPoints()):
            fa = inpd.GetPointData().GetArray('FA1').GetValue(i)
            fa_values.append(fa)
        return np.array(fa_values)
   
    else:
        raise ValueError(f"Unknown data type: {data_type}")

def assign_pts_to_nearest_node(pts, nodes):
    if len(pts) == 0 or len(nodes) == 0:
        return np.array([], dtype=int)
    dists = LA.norm(pts[:, np.newaxis] - nodes, axis=2)
    return np.argmin(dists, axis=1)

def calc_mean_dist_no_outliers(assignIds, nodes, pts):
    radius = []
    for i in range(len(nodes)):
        mask = (assignIds == i)
        parcel = pts[mask]
        if len(parcel) == 0:
            radius.append(0.0)
            continue
        d = LA.norm(parcel - nodes[i], axis=1)
        mean_d, std_d = np.mean(d), np.std(d)
        filtered = parcel[d <= mean_d + 2*std_d]
        r = np.mean(LA.norm(filtered - nodes[i], axis=1)) if len(filtered) > 0 else 0.0
        radius.append(r)
    return np.array(radius)

def construct_tract_neighborhood(cluster_dir, center_dir):
    center_nodes = []
    radii = []
    cluster_node_counts = []

    cluster_files = [
        f for f in os.listdir(cluster_dir)
        if f.endswith('.vtp') and f.startswith('cluster_')
    ]
    cluster_files = sorted(
        cluster_files,
        key=lambda f: int(''.join(filter(str.isdigit, f.split('_')[-1])))  
    )
    print(f"  Processing {len(cluster_files)} clusters in {os.path.basename(cluster_dir)}")

    for cluster_file in cluster_files:
        cluster_path = os.path.join(cluster_dir, cluster_file)
        inpd = wma.io.read_polydata(cluster_path)
        pts = extract_vtk_data(inpd, 'pt')
        if pts.size == 0:
            continue
        center_file = "center_" + cluster_file
        center_path = os.path.join(center_dir, center_file)
        if not os.path.exists(center_path):
            print(f"    Missing: {center_path}")
            continue
        centerline = wma.io.read_polydata(center_path)
        node_num = centerline.GetNumberOfPoints()
        if node_num == 0:
            continue
        nodes = np.array([centerline.GetPoint(i) for i in range(node_num)])
        assignIds = assign_pts_to_nearest_node(pts, nodes)
        radius = calc_mean_dist_no_outliers(assignIds, nodes, pts)
        center_nodes.extend(nodes)
        radii.extend(radius)
        cluster_node_counts.append(node_num)

    center_nodes = np.array(center_nodes)
    radii = np.array(radii)
    if center_nodes.size == 0:
        return np.array([])

    dists = LA.norm(center_nodes[:, np.newaxis, :] - center_nodes[np.newaxis, :, :], axis=-1)
    neighborhood = (dists < (radii[:, np.newaxis] + radii[np.newaxis, :])).astype(int)

    
    count = 0
    for node_num in cluster_node_counts:
        if node_num < 2:
            count += node_num
            continue
        neighborhood[count:count+node_num, count:count+node_num] = \
            np.eye(node_num, dtype=int, k=1) + np.eye(node_num, dtype=int, k=-1)
        count += node_num

    np.fill_diagonal(neighborhood, 0)
    neighborhood[neighborhood > 1] = 1
    print(f"  Final neighborhood shape: {neighborhood.shape}")
    return neighborhood

def save_heatmap(matrix, outpath):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap='binary', vmin=0, vmax=1, cbar=True, square=True)
    plt.title(os.path.basename(outpath).replace('.png', ''))
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# ============================= Main Process (only process AF_left) ==============================
os.makedirs(outdir, exist_ok=True)

tract_dir  = os.path.join(indir1, TRACT_NAME)
center_dir = os.path.join(indir2, TRACT_NAME)
npy_path   = os.path.join(outdir, f'{TRACT_NAME}.npy')
png_path   = os.path.join(outdir, f'{TRACT_NAME}.png')

print(f"\nProcessing: {TRACT_NAME}")
matrix = construct_tract_neighborhood(tract_dir, center_dir)

if matrix.size == 0:
    print(f"  Skip {TRACT_NAME}: no data")
else:
    np.save(npy_path, matrix)
    save_heatmap(matrix, png_path)
    print(f"  Saved: {npy_path}")
    print(f"  Saved: {png_path}")

print(f"\nAll done! Only processed {TRACT_NAME}")
print(f"Output directory: {outdir}")
