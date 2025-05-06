import argparse
from joblib import delayed, Parallel
from methods import *
from fswma import *
import numpy as np
import os
import re
import whitematteranalysis as wma
from tqdm import tqdm
from scipy.stats import ttest_ind
from collections import Counter
import pandas as pd
import subprocess
import concurrent.futures
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', dest='inputDirectory',
    help='Contains fiber clusters of dataset as vtkPolyData file(s).'
)
parser.add_argument(
    '-t', dest='template',
    help='Contains fiber clusters of ORG-atlas.'
)
parser.add_argument(
    '-l', dest='lst',
    help='The list of Subject-Classification.'
)
parser.add_argument(
    '-o', dest='outputDirectory',
    help='The output directory should be a new empty directory. It will be created if needed.'
)
parser.add_argument(
    '-r', dest='Result',
    help='The file of saving experimental results.'
)
parser.add_argument(
    '-c', dest='centroid',
    help='The folder of saving centroid as vtkPolyData file(s)'
)
parser.add_argument('-par1', dest='start')
parser.add_argument('-par2', dest='end')

args = parser.parse_args()
os.makedirs(args.outputDirectory, exist_ok=True)

# 1-ASD, 2-Healthy; 1-male, 2-female
sub_classification = []
with open(args.lst, 'r') as file:
    for line in file:
        info = line.strip().split(',')
        sub_classification.append(info[1].strip())
G1 = [index for index, element in enumerate(sub_classification) if element == '1']
G2 = [index for index, element in enumerate(sub_classification) if element == '2']
num1 = len(G1)
num2 = len(G2)

# subs and tracts to be addressed
subs = sorted(os.listdir(args.inputDirectory))
print(subs)
tracts = sorted(os.listdir(os.path.join(args.inputDirectory, subs[0], 'AnatomicalTracts_separated')))
tracts = [tract for tract in tracts if not tract.startswith('Sup-')]
print(len(tracts))


def process_vtp_or_vtk_file(tract, cluster_id, vtp_file):
    print(f'---{vtp_file}(template)---')
    inpd = wma.io.read_polydata(os.path.join(args.template, 'AnatomicalTracts_separated', tract, vtp_file))

    # calculate the center-line for these template clusters
    fiber_array = wma.fibers.FiberArray()
    fiber_array.convert_from_polydata(inpd, 100)
    center_line = compute_centroid_in_cluster(fiber_array)
    pd_center_line = convert_to_polydata([center_line])

    os.makedirs(os.path.join(args.centroid, tract), exist_ok=True)
    wma.io.write_polydata(pd_center_line, os.path.join(args.centroid, tract, 'centroid_' + vtp_file))

    points = get_info_from_vtk_or_vtp(inpd, 'point')
    assigned_id = assign_points_to_nearest_centroid(points, center_line)
    radius = remove_outliers_and_compute_mean_distance(assigned_id, center_line, points)

    # create _info1: DataFrame-(cluster_id, centroid_id, ras, radius) for center line
    _id1 = np.vstack((np.ones(len(center_line)) * cluster_id, np.arange(len(center_line)))).T
    _info1 = np.concatenate((_id1, center_line, radius.reshape(-1, 1)), axis=1)
    # create _info2: DataFrame-(cluster_id, assigned_id, ras) for all data points
    _id2 = np.vstack((np.ones(len(points)) * cluster_id, assigned_id)).T
    _info2 = np.hstack((_id2, points))

    # create info3
    def create_info3(sub):
        print(f'---{vtp_file}({sub})---')
        inpath = os.path.join(args.inputDirectory, sub, 'AnatomicalTracts_separated', tract, vtp_file)
        if os.path.getsize(inpath) / 1024 > 5:
            inpd = wma.io.read_polydata(inpath)
            sub_points = get_info_from_vtk_or_vtp(inpd, 'point')
            sub_vals = get_info_from_vtk_or_vtp(inpd, 'FA')
            sub_assigned_id = assign_points_to_nearest_centroid(sub_points, center_line)
            weighted_sub_vals = weighted_values_to_centroid(center_line, sub_assigned_id, sub_points, sub_vals)
        else:
            weighted_sub_vals = np.full(len(center_line), np.nan)

        return weighted_sub_vals

    _info3 = Parallel(n_jobs=30)(delayed(create_info3)(sub) for sub in subs)
    _info3 = np.column_stack(_info3)

    return _info1, _info2, _info3


print('======= proposed =======')
for tract in tracts[int(args.start):int(args.end)]:
    print(f'==== calculate {tract}... ====')
    template = os.path.join(args.template, 'AnatomicalTracts_separated', tract)
    vtp_files = sorted([file for file in os.listdir(template) if file.endswith('.vtp')])

    info1 = [None] * len(vtp_files)
    info2 = [None] * len(vtp_files)
    info3 = [None] * len(vtp_files)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_vtp_or_vtk_file, tract, cluster_id, vtp_file): cluster_id for
                   cluster_id, vtp_file in enumerate(vtp_files)}

        for future in concurrent.futures.as_completed(futures):
            cluster_id = futures[future]
            _info1, _info2, _info3 = future.result()
            info1[cluster_id] = _info1
            info2[cluster_id] = _info2
            info3[cluster_id] = _info3
    # combine info1, info2, info3 into single arrays
    info1 = np.vstack(info1)
    info2 = np.vstack(info2)
    info3 = np.vstack(info3)

    # save these files into appointed folder
    save_dir = os.path.join(args.outputDirectory, 'Proposed', tract)
    os.makedirs(save_dir, exist_ok=True)
    # np.save(os.path.join(save_dir, 'info1.npy'), info1)
    # np.save(os.path.join(save_dir, 'info2.npy'), info2)
    # np.save(os.path.join(save_dir, 'info3.npy'), info3)

    # compute neighbor matrix
    print('--- compute neighbor matrix ---')
    centroids = info1[:, 2:5]
    radii = info1[:, -1]
    # these distances among all points
    distance_matrix = np.linalg.norm(centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=-1)
    # create neighbor matrix
    neighbor = (distance_matrix < (radii[:, np.newaxis] + radii[np.newaxis, :])).astype(int)
    # count the number of centroids for each cluster
    cluster_id_count = Counter(info1[:, 0])
    sorted_cluster_id_count = sorted(cluster_id_count.items())
    # update neighbor
    count_sum = 0
    for _, count in sorted_cluster_id_count:
        neighbor[count_sum:count_sum + count, count_sum:count_sum + count] = np.eye(count, dtype=int, k=1) + np.eye(
            count, dtype=int, k=-1)
        count_sum += count
    # save neighbor
    np.fill_diagonal(neighbor, 0)
    neighbor[neighbor >= 1] = 1
    #    np.save(os.path.join(save_dir, 'neighbor.npy'), neighbor)

    # compute p matrix
    print('--- compute p-value matrix ---')
    real_plist = []
    G1_fas = info3[:, G1]
    G2_fas = info3[:, G2]
    for i in tqdm(range(len(info1))):
        G1_test = G1_fas[i]
        G1_test = G1_test[~np.isnan(G1_test)]

        G2_test = G2_fas[i]
        G2_test = G2_test[~np.isnan(G2_test)]
        _, p = ttest_ind(G1_test, G2_test)
        real_plist.append(p)
    real_p = np.array(real_plist)

    # Calculate the fake p value by permutation test
    print('--- permutation test ---')
    fake_plist = []
    idx = np.arange(len(subs))
    for i in tqdm(range(int(100))):
        fake_G1_idx = np.random.choice(idx, size=num1, replace=False)
        fake_G2_idx = np.delete(idx, fake_G1_idx)
        fake_G1_fas = info3[:, fake_G1_idx]
        fake_G2_fas = info3[:, fake_G2_idx]

        plist = []  # temp
        for j in range(len(info1)):
            fake_G1_test = fake_G1_fas[j]
            fake_G1_test = fake_G1_test[~np.isnan(fake_G1_test)]

            fake_G2_test = fake_G2_fas[j]
            fake_G2_test = fake_G2_test[~np.isnan(fake_G2_test)]

            _, p = ttest_ind(fake_G1_test, fake_G2_test)
            plist.append(p)
        fake_plist.append(plist)

    fake_p = np.vstack(fake_plist)
    all_p = np.vstack((real_p, fake_p))

    # combine p-value and neighbor matrix
    print('--- combine p-value and neighbor matrix ---')
    combined_matrix = np.zeros((10001, len(info1), len(info1)))
    for i in tqdm(range(10001)):
        p_line = all_p[i]
        indices = np.where(p_line <= 0.05)[0]
        # create boolean mask
        row_mask = np.zeros(len(info1), dtype=bool)
        row_mask[indices] = True
        col_mask = np.zeros(len(info1), dtype=bool)
        col_mask[indices] = True
        # combine with neighbor
        combined_matrix[i][row_mask[:, None] & col_mask[None, :]] = neighbor[row_mask[:, None] & col_mask[None, :]]

    # acquire supported pointset by k-cliques
    print('--- calculate supported pointset by k-cliques ---')
    set_size = []
    for i in tqdm(range(1, combined_matrix.shape[0])):
        g = convert_matrix_to_graph(combined_matrix[i])
        coms = get_percolated_cliques(cliques, k)
        if coms != None:
            if len(coms) == 0:
                set_size.append(0)
            else:
                set_size.append(len(coms[0]))

    # plot the result of permutation test
    print('--- plot histogram of permutation test ---')
    # count the frequency of different nums
    count = Counter(set_size)
    nums = list(count.keys())
    freqs = list(count.values())

    # Sort nums and freqs based on nums
    nums_sorted = sorted(nums)
    freqs_sorted = [freqs[nums.index(num)] for num in nums_sorted]
    np.save(os.path.join(save_dir, 'nums.npy'), nums_sorted)
    np.save(os.path.join(save_dir, 'freqs.npy'), freqs_sorted)

    # find the node of 95% for ASD and 95% for HCP
    value = int(sum(freqs_sorted) * 0.95) + 1
    cumulative_freq = 0
    for num, freq in zip(nums_sorted, freqs_sorted):
        cumulative_freq += freq
        if cumulative_freq >= value:
            node = num
            break
    print(f'The node of 95% is {node}')

    plt.figure()
    # use colors to set color and plot trend line
    colors = ['red' if num >= node else 'blue' for num in nums_sorted]
    plt.bar(nums_sorted, freqs_sorted, color=colors)
    plt.plot(nums_sorted, freqs_sorted, marker='o', linestyle='-', markersize=3, linewidth=1)
    # add label for set_size[0]
    for i in range(len(nums_sorted)):
        if nums_sorted[i] == node:
            plt.text(nums_sorted[i], freqs_sorted[i], str(nums_sorted[i]), ha='center', va='bottom')
    # the details of figure
    plt.xlabel('Num')
    plt.ylabel('Frequency')
    plt.title('{}'.format(tract))
    plt.savefig(os.path.join(save_dir, f'{tract}.png'))
    plt.clf()

    # find the satisfied supported pointsets and visualization
    print('--- find the satisfied supported pointsets and visualize ---')
    g = convert_matrix_to_graph(combined_matrix[0])
    coms = get_percolated_cliques(cliques,k)
    # count the real result
    supported_set = [com for com in coms if len(com) >= node]
    n = len(supported_set)
    print('{} has {} pointset(s)'.format(tract, n))
    # # write into the result.txt
    with open(args.Result, 'a') as file:
        file.write(f'{tract}: {n}\n')

    # find the id of centroid with detected significant difference
    color_index = np.zeros(len(info2))
    for i in range(len(supported_set)):
        print(f'The {i}th supported_set')
        print(supported_set[i])
        ids = info1[supported_set[i]][:, :2]
        print(ids)
        _ids = info2[:, :2]
        for j in range(len(ids)):
            id = ids[j]
            condition = np.all(_ids == id, axis=1)
            indices = np.where(condition)[0]
            color_index[indices] = i + 1

    start = 0
    for vtp_file in vtp_files:
        inpd = wma.io.read_polydata(os.path.join(args.template, 'AnatomicalTracts_separated', tract, vtp_file))
        lines, lines_colors = [], []

        for lidx in range(inpd.GetNumberOfCells()):
            pts = inpd.GetCell(lidx).GetPoints()

            line_points = np.array([np.array(pts.GetPoint(pidx)) for pidx in range(pts.GetNumberOfPoints())])
            line_colors = np.array([color_index[start + pidx] for pidx in range(pts.GetNumberOfPoints())])
            start += pts.GetNumberOfPoints()

            lines.append(line_points)
            lines_colors.append(line_colors)

        save_path = os.path.join(save_dir, 'vis')
        os.makedirs(save_path, exist_ok=True)

        outpath = os.path.join(save_path, vtp_file)
        write_vtk_vtp(lines_points=lines, output_filename=outpath, additional_array=lines_colors)

    parameters = [
        r'/home/ruixi/whitematteranalysis/bin/wm_append_clusters.py',
        save_path,
        save_dir,
        '-appendedTractName',
        tract,
    ]
    subprocess.run(parameters)







