# AGFS-Tractometry
This repository releases the source code, pre-created template and testing sample (sub-100307) for the work related to our ongoing study on a novel approach for finer-scale tractometry using dMRI.
[![image.png](https://i.postimg.cc/52jbKZ7h/image.png)](https://postimg.cc/K1X6Rsyf)

## Function
- **create_tract_centroid**: This script is designed to calculate the central line of WM tracts.
  - _Example_: 
    ```python
    ./create_tract_centroid <input_file> <output_dir> -numPoints 100
    ```
[![image.png](https://i.postimg.cc/Fzp5KZz5/image.png)](https://postimg.cc/H8rPZbbB)

- **extract_tract_neighborhood**: This function extracts the neighborhood of a given tract.
  - _Example_: 
    ```python
    ./extract_tract_neighborhood <input_dir> <output_dir> -numPoints 100 
    ```

- **compute_diffusion_metrics**: Computes various diffusion metrics for tractometry analysis.
  - _Example_: 
    ```python
    ./compute_diffusion_metrics <input_dir> -numPoints 100
    ```

- **clique_percolation_algorithm**: Implements the clique percolation algorithm for community detection in tractometry.
  - _Example_: 
    ```python
    ./clique_percolation_algorithm
    ```

## Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/AGFS-Tractometry.git
   cd AGFS-Tractometry