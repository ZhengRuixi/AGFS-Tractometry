#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import vtkmodules.all as vtk
from vtk.util.numpy_support import vtk_to_numpy
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import whitematteranalysis as wma

# =============== Relative Paths ===============
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
MDD_DIR = DATA_DIR / "MDD"
HC_DIR = DATA_DIR / "HC"
CENTERLINE_DIR = DATA_DIR / "ORG-atlas" / "ORG-atlas_centerline"

METRIC = "FA"
TRACT_TO_PROCESS = "AF_left"
NUM_WORKERS = min(32, os.cpu_count() or 1)
# =====================================================

def extract_vtk_data(polydata, dtype):
    if dtype == 'pt':
        pts = vtk_to_numpy(polydata.GetPoints().GetData())
        return pts if pts is not None else np.empty((0, 3))
    if dtype == 'fa':
        fa_array = polydata.GetPointData().GetArray('FA1')
        if fa_array is None:
            return np.zeros(polydata.GetNumberOfPoints())
        return np.array([fa_array.GetValue(i) for i in range(fa_array.GetNumberOfTuples())])
    return np.array([])

def assign_pts_to_nearest_node(points, nodes):
    if points.size == 0 or nodes.size == 0:
        return np.array([], dtype=int)
    tree = KDTree(nodes)
    _, idx = tree.query(points)
    return idx

def calc_weighted_diffusion_value(nodes, assign_ids, pts, vals):
    res = np.full(len(nodes), np.nan, dtype=float)
    for i in range(len(nodes)):
        mask = (assign_ids == i)
        pts_i = pts[mask]
        vals_i = vals[mask]
        if pts_i.shape[0] < 3:
            continue
        c = np.cov(pts_i.T, ddof=1)
        if np.isclose(np.linalg.det(c), 0):
            continue
        try:
            c_inv = np.linalg.inv(c)
            dists = cdist(pts_i, nodes[i:i+1], metric='mahalanobis', VI=c_inv).ravel()
            weights = 1.0 / (dists + 1e-12)
            weights /= weights.sum()
            res[i] = np.dot(vals_i, weights)
        except np.linalg.LinAlgError:
            continue
    return res

def process_subject(group_dir, subject_id):
    tract_base = Path(group_dir) / subject_id / 'AnatomicalTracts-Separated' / TRACT_TO_PROCESS
    if not tract_base.exists():
        return
    center_dir = CENTERLINE_DIR / TRACT_TO_PROCESS
    results = []
    for cluster_file in sorted(os.listdir(tract_base)):
        cluster_path = tract_base / cluster_file
        center_path = center_dir / f'center_{cluster_file}'
        if not center_path.exists():
            continue
        inpd = wma.io.read_polydata(str(cluster_path))
        pts = extract_vtk_data(inpd, 'pt')
        fas = extract_vtk_data(inpd, 'fa')
        center_pd = wma.io.read_polydata(str(center_path))
        nodes = np.array([center_pd.GetPoint(i) for i in range(center_pd.GetNumberOfPoints())])
        if nodes.size == 0:
            continue
        assign_ids = assign_pts_to_nearest_node(pts, nodes)
        cluster_vals = calc_weighted_diffusion_value(nodes, assign_ids, pts, fas)
        results.extend(cluster_vals)

    outdir = Path(group_dir) / subject_id / 'diffusion_measurements' / TRACT_TO_PROCESS
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / f'{METRIC}.npy', np.array(results, dtype=np.float32))

def main():
    subjects = [(MDD_DIR, sid) for sid in os.listdir(MDD_DIR)] + \
               [(HC_DIR, sid) for sid in os.listdir(HC_DIR)]
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_subject, group, sid) for group, sid in subjects]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing subjects"):
            pass

if __name__ == '__main__':
    main()
