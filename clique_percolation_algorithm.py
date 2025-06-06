from create_tract_centroid import *
from extract_tract_neighborhood import *
import networkx as nx
from scipy.sparse import lil_matrix
import numpy as np


def convert_matrix_to_graph(matrix):
   # For example, using adjacency matrix to create a graph
   g = nx.Graph()
   matrix = np.array(matrix)
   rows, cols = np.where(matrix == 1)
   g.add_edges_from(zip(rows, cols))
   return g


def get_percolated_cliques(g, k):
   # Find all cliques >= k
   cliques = [frozenset(clique) for clique in nx.find_cliques(g) if len(clique) >= k]
   num = len(cliques)
   # Build overlap matrix
   overlap_matrix = lil_matrix((num, num), dtype=int)
   # print(f'overlap_matrix: {overlap_matrix.shape}')
   for i in range(num):
       for j in range(i + 1, num):  # Start from i+1 to avoid duplicate calculations
           intersection = len(cliques[i].intersection(cliques[j]))
           if intersection >= k-1:  # original:k-1
               overlap_matrix[i, j] = overlap_matrix[j, i] = 1

   # Record the community ID
   community_ids = list(range(num))
   for i in range(num):
       for j in range(i + 1, num):
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


# Give a test
if __name__ == '__main__':
    # create an example
    G = nx.karate_club_graph()
    communities = get_percolated_cliques(G, 3)
    for i, community in enumerate(communities):
        print(f"Community {i+1}: {community}")