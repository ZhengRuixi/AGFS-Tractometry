#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import ttest_ind
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==================== Paths (Kept as is) ====================
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
tract_name = 'AF_left'

MDD_dir = '/data05/xiluwang/GitHub/data/MDD'      # MDD group
HC_dir  = '/data05/xiluwang/GitHub/data/HC'       # HC group

type = 'FA'

perm_results_root = REPO_ROOT / 'perm_results'
outdir_p = perm_results_root / 'p' / type
outdir_r = perm_results_root / 'r' / type
neighbor_dir = '/data05/xiluwang/neighborhood'
final_outdir = perm_results_root / 'res'

for d in [outdir_p, outdir_r, final_outdir]:
    os.makedirs(d, exist_ok=True)

SKIP_SUBS = set()


# ==================== Part 1: Generate p and r  ====================
def concatenate(tract_name, indir, type):
    mat = []
    sub_dirs = [d for d in sorted(os.listdir(indir)) if d not in SKIP_SUBS and os.path.isdir(os.path.join(indir, d))]
    for sub_dir in sub_dirs:
        file = os.path.join(indir, sub_dir, 'diffusion_measurements', tract_name, f'{type}.npy')
        if os.path.exists(file):
            data = np.load(file)
            mat.append(data)
    return np.array(mat)


def permutation_run(tract_name, MDD_dir, HC_dir, outdir, type='FA', threshold=0.01, n_perm=1000):
    mat1 = concatenate(tract_name, MDD_dir, type)   # MDD
    mat2 = concatenate(tract_name, HC_dir, type)    # HC
    
    if mat1.size == 0 and mat2.size == 0:
        print(f"[Skip] {tract_name}: no data in both groups")
        return
    

    mat = np.concatenate([mat2, mat1], axis=0)       
    G1_num = mat2.shape[0]   
    G2_num = mat1.shape[0]
    total_num = G1_num + G2_num
    if total_num == 0:
        return
    
    pval_mat = np.full((n_perm, mat.shape[1]), np.nan)
    for i in tqdm(range(n_perm), desc=f'{tract_name} permutation'):
        perm = np.random.permutation(total_num)
        G1_idx = perm[:G1_num]
        G2_idx = perm[G1_num:]
        G1_tests = mat[G1_idx, :]
        G2_tests = mat[G2_idx, :]
        for feat in range(mat.shape[1]):
            G1_test = G1_tests[:, feat][~np.isnan(G1_tests[:, feat])]
            G2_test = G2_tests[:, feat][~np.isnan(G2_tests[:, feat])]
            if len(G1_test) >= 2 and len(G2_test) >= 2:
                _, p_val = ttest_ind(G1_test, G2_test)
                pval_mat[i, feat] = p_val
    outpath = os.path.join(outdir, f'{tract_name}.npy')
    np.save(outpath, pval_mat)


def real_run(tract_name, MDD_dir, HC_dir, outdir, type='FA'):
    mat1 = concatenate(tract_name, MDD_dir, type)   # MDD
    mat2 = concatenate(tract_name, HC_dir, type)    # HC
    pval_mat = np.zeros(mat2.shape[1])
    for feat in tqdm(range(mat2.shape[1]), desc=f'{tract_name} real'):
        G1_test = mat1[:, feat][~np.isnan(mat1[:, feat])]
        G2_test = mat2[:, feat][~np.isnan(mat2[:, feat])]
        if len(G1_test) >= 2 and len(G2_test) >= 2:
            _, p_val = ttest_ind(G1_test, G2_test)
            pval_mat[feat] = p_val
    outpath = os.path.join(outdir, f'{tract_name}.npy')
    np.save(outpath, pval_mat)


# ==================== Part 2: Post-processing  ====================
def perm_row_to_graph(p_row, neighbor_matrix, p_thresh=0.01):
    p_row = np.asarray(p_row)
    vec = (p_row < p_thresh)
    indices = np.nonzero(vec)[0]
    n_nodes = neighbor_matrix.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    if indices.size <= 1:
        return g
    sub = neighbor_matrix[np.ix_(indices, indices)]
    sub_upper = np.triu(sub, k=1)
    rows, cols = np.nonzero(sub_upper)
    edges = [(int(indices[r]), int(indices[c])) for r, c in zip(rows, cols)]
    if edges:
        g.add_edges_from(edges)
    return g


def postprocess_permutation_results(tract_name, permuted_npy_dir, real_npy_dir, neighbor_npy_dir, p_thresh=0.01):
    permuted_npy = os.path.join(permuted_npy_dir, f'{tract_name}.npy')
    real_npy = os.path.join(real_npy_dir, f'{tract_name}.npy')
    neighbor_npy = os.path.join(neighbor_npy_dir, f'{tract_name}.npy')
    real_mat = np.load(real_npy)
    permuted_mat = np.load(permuted_npy)
    neighborhood = np.load(neighbor_npy)
    neighborhood = (neighborhood != 0)
    process_mat = np.vstack((real_mat.reshape(1, -1), permuted_mat))
    graphs = []
    for i in range(process_mat.shape[0]):
        g = perm_row_to_graph(process_mat[i], neighborhood, p_thresh=p_thresh)
        graphs.append(g)
    return graphs


def merge_cliques_by_union_find(g, k=3, h=3):
    cliques = [frozenset(clique) for clique in nx.find_cliques(g) if len(clique) >= k]
    parent = list(range(len(cliques)))
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    for i in range(len(cliques)):
        for j in range(i + 1, len(cliques)):
            if len(cliques[i] & cliques[j]) >= h:
                root_i = find(i)
                root_j = find(j)
                if root_i != root_j:
                    parent[root_j] = root_i
    communities = {}
    for idx in range(len(cliques)):
        root = find(idx)
        if root not in communities:
            communities[root] = set()
        communities[root].update(cliques[idx])
    communities = sorted((list(community) for community in communities.values()), key=len, reverse=True)
    return communities


def detect_significant_difference(outdir, tract_name, graphs, threshold=95):
    num = len(graphs)
    results = []
    for g in graphs:
        results.append(merge_cliques_by_union_find(g))
    max_sizes = []
    for result in results[1:]:
        if len(result) > 0:
            max_sizes.append(len(result[0]))
        else:
            max_sizes.append(0)
    max_sizes_arr = np.array(max_sizes, dtype=float)
    node = int(np.percentile(max_sizes_arr, threshold)) if max_sizes_arr.size > 0 else 0
    if len(results[0]) > 0:
        real_size = len(results[0][0])
    else:
        real_size = 0

    plt.figure(figsize=(10, 5))
    if max_sizes_arr.size > 0:
        plt.hist(max_sizes_arr, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(node, color='red', linestyle='--', linewidth=2, label=f'95% threshold = {node}')
    plt.axvline(real_size, color='green', linestyle='-', linewidth=2, label=f'real = {real_size}')
    plt.xlabel('Community Size')
    plt.ylabel('Frequency')
    plt.title(f'{tract_name} Permutation Test Histogram')
    plt.legend()
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f'{tract_name}.png'))
    plt.close()

    passed_coms = [com for com in results[0] if len(com) >= node]
    if len(passed_coms) > 0:
        max_len = max(len(com) for com in passed_coms)
        detected_coms = np.full((len(passed_coms), max_len), -1, dtype=int)
        for i, com in enumerate(passed_coms):
            detected_coms[i, :len(com)] = com  
        detected_coms = np.array([])
    np.save(os.path.join(outdir, f'{tract_name}.npy'), detected_coms)
    print(f'Significant communities found: {len(passed_coms)}. Saved.')


# ==================== Main Process ====================
if __name__ == '__main__':
    print(f"Starting processing {tract_name} ")

    real_run(tract_name, MDD_dir, HC_dir, outdir_r, type='FA')
    permutation_run(tract_name, MDD_dir, HC_dir, outdir_p, type='FA', n_perm=1000)

    graphs = postprocess_permutation_results(tract_name, outdir_p, outdir_r, neighbor_dir, p_thresh=0.01)
    detect_significant_difference(final_outdir, tract_name, graphs, threshold=95)

    print("All done!")
    print(f"Results are at: {final_outdir}/AF_left.npy and AF_left.png")
