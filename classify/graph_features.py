import numpy as np
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
import mne
import networkx as nx

def reshape(data):
    data = data.reshape(-1, data.shape[1]).T
    return data

def emd(x, nIMF = 8, stoplim = .001):
    """Perform empirical mode decomposition to extract 'niMF' components out of the signal 'x'."""
    
    r = x
    t = np.arange(len(r))
    imfs = np.zeros(nIMF,dtype=object)
    for i in range(nIMF):
        r_t = r
        is_imf = False
        
        while is_imf == False:
            # Identify peaks and troughs
            pks = sp.signal.argrelmax(r_t)[0]
            trs = sp.signal.argrelmin(r_t)[0]
            
            # Interpolate extrema
            pks_r = r_t[pks]
            fip = sp.interpolate.InterpolatedUnivariateSpline(pks,pks_r,k=3)
            pks_t = fip(t)
            
            trs_r = r_t[trs]
            fitr = sp.interpolate.InterpolatedUnivariateSpline(trs,trs_r,k=3)
            trs_t = fitr(t)
            
            # Calculate mean
            mean_t = (pks_t + trs_t) / 2
            mean_t = _emd_complim(mean_t, pks, trs)
            
            # Assess if this is an IMF (only look in time between peaks and troughs)
            sdk = _emd_comperror(r_t, mean_t, pks, trs)
            
            # if not imf, update r_t and is_imf
            if sdk < stoplim:
                is_imf = True
            else:
                r_t = r_t - mean_t
                
        
        imfs[i] = r_t
        r = r - imfs[i] 
        
    return imfs



def _emd_comperror(h, mean, pks, trs):
    """Calculate the normalized error of the current component"""
    samp_start = np.max((np.min(pks),np.min(trs)))
    samp_end = np.min((np.max(pks),np.max(trs))) + 1
    return np.sum(np.abs(mean[samp_start:samp_end]**2)) / np.sum(np.abs(h[samp_start:samp_end]**2))


def _emd_complim(mean_t, pks, trs):
    """Discard the mean extrema envelope past the first and last extrema"""
    samp_start = np.max((np.min(pks),np.min(trs)))
    samp_end = np.min((np.max(pks),np.max(trs))) + 1
    mean_t[:samp_start] = mean_t[samp_start]
    mean_t[samp_end:] = mean_t[samp_end]
    return mean_t

def get_imfs(data):
    all_imfs = []
    for signal in data:
        channel_data = signal
        imfs = emd(channel_data)
        all_imfs.append(imfs)

    imfs_array = np.array(all_imfs, )
    imfs_list = []
    for channel in range(imfs_array.shape[0]):
        imfs_list_channel = []
        for imf in range(imfs_array.shape[1]):
            imf_data = imfs_array[channel][imf]
            imfs_list_channel.append(imf_data)
        imfs_list.append(imfs_list_channel)

    imfs_3d = np.array(imfs_list)

    return imfs_3d


def apply_fft(data):
    fft_output = np.fft.fft(data, axis=1)
    magnitude = np.abs(fft_output)
    phase = np.angle(fft_output)
    return magnitude, phase

def calculate_plv(phase1, phase2):
    phase_diff = phase1 = phase2
    plv_complex = np.exp(1j * phase_diff)
    plv = np.abs(np.mean(plv_complex))
    return plv

def calculate_plv_matrix(phase_data, num_channels): # iterate through columns, and caulculate plv sfor each combo of channels
    plv_matrix = np.zeros((num_channels, num_channels))
    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            plv_matrix[i,j]= calculate_plv(phase_data[i], phase_data[j])
            plv_matrix[j,i] = plv_matrix[i, j]
    return plv_matrix


def find_all_matrices(phase_array):
    phase_matrices = []
    for data in range(phase_array.shape[1]):
        matrix = calculate_plv_matrix(phase_array[:,data,:], phase_array.shape[0])
        phase_matrices.append(matrix)
    phase_matrices = np.array(phase_matrices) 
    return phase_matrices

def imf_connectivity_matrices(data):
    data = reshape(data)
    imfs_arrays = get_imfs(data)
    magnitude_array = np.zeros_like(imfs_arrays, dtype=np.float64)
    phase_array = np.zeros_like(imfs_arrays, dtype=np.float64)
    for imf in range(imfs_arrays.shape[1]):
        magnitude, phase = apply_fft(imfs_arrays[:, imf, :])
        magnitude_array[:, imf, :] = magnitude
        phase_array[:, imf,:] = phase
    plv_matrices = find_all_matrices(phase_array)
    return plv_matrices


def calculate_node_strength(data):
    # Sum along rows to get node strengths (undirected case)
    matrices = imf_connectivity_matrices(data)
    array = []
    for i in range(matrices.shape[0]):
        node_strengths = np.sum(matrices[i], axis=1)
        array.append(node_strengths)
    return np.vstack(array)


def clustering_coefficient(data):
    matrices = imf_connectivity_matrices(data)
    array = []
    for i in range(matrices.shape[0]):
        num_nodes = matrices[i].shape[0]
        clustering_coeffs = np.zeros(num_nodes)

        for i in range(num_nodes):
            # Get neighbors of node i (non-zero connections)
            neighbors = np.where(matrices[i][i] > 0)[0]
            num_neighbors = len(neighbors)

            if num_neighbors < 2:
                continue  # No triangles if less than 2 neighbors

            # Sum of weights between neighbors
            subgraph = matrices[i][np.ix_(neighbors, neighbors)]
            triangle_weight_sum = np.sum(subgraph) / 2

            # Strength of node (sum of connections)
            strength = np.sum(matrices[i][i])

            # Compute clustering coefficient
            clustering_coeffs[i] = (2 * triangle_weight_sum) / (strength * (num_neighbors - 1))
        array.append(clustering_coeffs)
    return np.vstack(array)

            



def local_efficiency(data):
    matrices = imf_connectivity_matrices(data)
    array = []
    for i in range(matrices.shape[0]):
        num_nodes = matrices[i].shape[0]
        efficiency = np.zeros(num_nodes)

        for i in range(num_nodes):
            # Get neighbors of node i (non-zero connections)
            neighbors = np.where(matrices[i][i] > 0)[0]
            num_neighbors = len(neighbors)

            # Local efficiency is zero if fewer than 2 neighbors
            if num_neighbors < 2:
                continue

            # Extract the subgraph of neighbors
            subgraph = matrices[i][np.ix_(neighbors, neighbors)]

            # Build graph from the subgraph (for shortest paths)
            G = nx.from_numpy_array(subgraph)

            # Sum of inverse shortest paths between neighbors
            path_sum = 0
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    try:
                        path_length = nx.shortest_path_length(G, source=j, target=k, weight='weight')
                        path_sum += 1 / path_length if path_length > 0 else 0
                    except nx.NetworkXNoPath:
                        continue

            # Calculate local efficiency for node i
            efficiency[i] = (2 * path_sum) / (num_neighbors * (num_neighbors - 1))

        array.append(efficiency)
    return np.vstack(array)

def betweenness_centrality(data):
    """
    Computes betweenness centrality for a weighted graph.
    
    Parameters:
    - connectivity_matrix: 2D numpy array (e.g., PLV matrix) where entry (i, j) is the edge weight.
    
    Returns:
    - centrality: 1D numpy array of betweenness centrality values for each node.
    """
    matrices = imf_connectivity_matrices(data)
    array = []
    for i in range(matrices.shape[0]):
        # Ensure the matrix is symmetric and has zero diagonals
        np.fill_diagonal(matrices[i], 0)

        # Create a graph from the weighted matrix
        G = nx.from_numpy_array(matrices[i])

        # Compute betweenness centrality (weight = inverse of connection for shortest path calculation)
        centrality = nx.betweenness_centrality(G, weight='weight', normalized=True)

        # Convert centrality dictionary to numpy array
        array.append(np.array(list(centrality.values())))
    return np.vstack(array)











