import sys
import copy
import math
import itertools
from collections import Counter
import warnings
import multiprocessing as mp
from typing import List, Tuple

import scipy.signal
import sklearn.cluster
import numpy as np
import sklearn.decomposition
import networkx as nx
import scipy.stats
from scipy.ndimage.interpolation import shift
from skimage.util.shape import view_as_windows

import fastconv.corr1d as corr1d # compiled dependency wueric, compiled already




def align_group(X, sample_len = 30, window = 10, res = 2):
    """
    Align a set of set of signals to approximatetly minimize global l2 distance
    X: np.array
    window: max acceptable shift. int, default = 10
    res: resolution of cubic interpolation. int, default = 2
    sample_len: length of signals to be aligned. int, default = 2


    output: tuple<np.array>

    """
    X = X.reshape((-1,sample_len))

    # Special case: Only one signal in signal set
    if len(X) == 1:
        return X[0]
    else:
        
        # Upsample data for better signal alignment
        x = np.arange(0, len(X[0]))
        interpolant = scipy.interpolate.interp1d(x,X, kind='cubic')
        domain = np.linspace(0, len(x)-1, len(x)*res)
        high_res_X = interpolant(domain)
        
        min_index = np.argmin(X,axis = 1)
        
        # Find signals with median sample of either depolarization or hyperpolarization hump (if there is a cell firing) i.e. find an average signal
        c_i = np.argsort(min_index)[len(min_index)//2]
        
        # Store median signal to align all others to
        x = high_res_X[c_i:c_i + 1,::-1]
        
        one_window = np.zeros_like(x)
        one_window[:,::res] = 1
        one_window = one_window[:,::-1]
        
        # Compute l2 difference between reference and all other signals using convolution
        _, signal_len = x.shape
        conv_Xx = scipy.signal.convolve(high_res_X, x*one_window, mode = 'full')[:,(2*signal_len-1)//2-window*res:(2*signal_len-1)//2+window*res]
        conv_X21 = scipy.signal.convolve(high_res_X**2, one_window, mode = 'full')[:,(2*signal_len-1)//2-window*res:(2*signal_len-1)//2+window*res]
        conv_x21 = scipy.signal.convolve(x**2, one_window, mode = 'full')[:,(2*signal_len-1)//2-window*res:(2*signal_len-1)//2+window*res]
        
        # Indentify and shift signals to approximately minimize global l2 distance   
        amin_Xx = -(np.argmin( conv_X21 + conv_x21 - 2*conv_Xx, axis = 1) - window*res)
        high_res_X = strided_indexing_roll(high_res_X, amin_Xx)
        np.nan_to_num(high_res_X, copy = False)
        
        return np.mean(high_res_X[:,::res], axis = 0)

def all_pairwise(X_, Y_):
    """
    Compute pairwise difference signals between matrices with different number of rows
    X_: np.array
    Y_: np.array

    output: np.array
    """
    _, num_electrodes, num_samples = X_.shape
    X = X_[:,None,:,:]
    Y = Y_[None,:,:,:]
    
    return (X-Y).reshape((-1,num_electrodes,num_samples))

def convert_cliques_to_labels(cluster_cliques, num_trials):
    """
    Purpose: Convert list of list of sets (clique list from multiple cells) to an array of cluster ids
    """

    # Seed final list of cliques with the list of cliques from one cell
    recluster = cluster_cliques[0]

    # Iterate the list of cliques from every other cell
    for cc in cluster_cliques:
        tmp = []

        # Iterate through "reclustered" clique list
        for u in recluster:

            # Iterate through remaining clique lists
            for v in cc:

                # Find the trials that overlap in the cliques and add them to a temporary list
                w = set.intersection(*[u,v])
                if len(w) > 0:
                    tmp += [w]
        
        # Reseed final list of cliques 
        recluster = tmp

    # Assign labels to new merge clique list
    event_labels = np.zeros(num_trials, dtype = int)
    for i, cl in enumerate(recluster):
        event_labels[np.array(list(cl))]=i
    return event_labels

def compute_cosine_error(mask, edge_to_matched_signals, n, bad_edges_, similarity_threshold = 0.7):
    """
    This method computes the cosine similarity between the template (EI) assigned to a difference signal and the difference signal itself. If they have low cosine similarity, this indicates they don't "rise" and "fall" around the same time points. This is particularly helpful at filtering assignments when the algorithm attempts to "shove" the template to the very end of the possible window, effectively not using the information in the template. 
    """
    labelled_edges = [k for k, v in edge_to_matched_signals.items() if k[1]==n]
    
    if len(labelled_edges):
        cosine_similarity = [compute_cosine_similarity(mask*edge_to_matched_signals[e][0],edge_to_matched_signals[e][1]) for e in labelled_edges]
        low_power = [sim<=similarity_threshold for sim in cosine_similarity]
        bad_edges = [labelled_edges[i][0] for i, c in enumerate(low_power) if c]
        bad_edges_ += [e for e in bad_edges if tuple(e) not in [tuple(b) for b in bad_edges_]]
    return bad_edges_

def compute_cosine_similarity(u, v):
    #return np.sum(u * v)/np.linalg.norm(u)/np.linalg.norm(v)
    """Returns the cosine similarity between two tensors."""
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    if u_norm == 0 or v_norm == 0:
        return 0.0
    return u.ravel().dot(v.ravel()) / (u_norm * v_norm)    



def compute_latency_error(edge_to_matched_signals, n, bad_edges_, electrode, offset = 50, too_early = 5, too_late = 25):
    """
    Purpose: Find edges wherein the assigned template fall outside of the accepted window
    """
    labelled_edges = [k for k, v in edge_to_matched_signals.items() if k[1]==n]

    if len(labelled_edges):
        too_early_or_late = [(offset-edge_to_matched_signals[e][-1][electrode] < too_early) or (offset-edge_to_matched_signals[e][-1][electrode] > too_late ) for e in labelled_edges]
        bad_edges = [labelled_edges[i][0] for i, c in enumerate(too_early_or_late) if c]
        bad_edges_ += [e for e in bad_edges if tuple(e) not in [tuple(b) for b in bad_edges_]]
    return bad_edges_

def compute_small_signal_error(edge_to_matched_signals, n, bad_edges_, electrode,too_small = 1):
    """
    Purpose: Find edges wherein the assigned template is too small
    """
    labelled_edges = [k for k, v in edge_to_matched_signals.items() if k[1]==n]

    if len(labelled_edges): 
        too_small = [(np.max(np.abs(edge_to_matched_signals[e][1])[electrode])<=too_small) for e in labelled_edges]
        bad_edges = [labelled_edges[i][0] for i, c in enumerate(too_small) if c]
        bad_edges_ += [e for e in bad_edges if tuple(e) not in [tuple(b) for b in bad_edges_]]
    return bad_edges_

def compute_probability_error(G, clustering, bad_edges):
    """
    Purpose: Compute probability contributed by problematic edges
    """
    error = 0 
    error += sum([sum(clustering == e[1])/len(clustering) for e in bad_edges])
    error += sum([sum(clustering == n)/len(clustering) for e in bad_edges for n in nx.descendants(G, e[1])])
    return error

def compute_difference_signals(signals, event_labels):
    """
    Compute all potential difference signals between event cliques and store the subtraction ordering
    signals: np.array
    event_labels: list<int>

    output: np.array, np.array
    """
    _, num_electrodes, num_samples = signals.shape
    
    event_set = list(set(event_labels))
    num_event_signals = len(event_set)
    
    pairwise_signals = []
    signal_ordering = []
    

    for pair_of_event_cliques in list(itertools.combinations(np.arange(num_event_signals), 2 )):
        # Compute set of pairwise difference signals between event cliques
        X = signals[event_labels == event_set[pair_of_event_cliques[0]]]
        Y = signals[event_labels == event_set[pair_of_event_cliques[1]]]
        Z = np.einsum('ijk->jik',all_pairwise(X,Y)).reshape((num_electrodes, -1))

        # Align and take the average as the exemplar difference signal between the two event cliques
        diff_signal = np.array([align_group(Z[i], sample_len = num_samples) for i in range(num_electrodes)])
       
        pairwise_signals += [diff_signal]
        pairwise_signals += [-diff_signal]
        signal_ordering += [(event_set[pair_of_event_cliques[0]], event_set[pair_of_event_cliques[1]])]
        signal_ordering += [(event_set[pair_of_event_cliques[1]], event_set[pair_of_event_cliques[0]])]
        
    pairwise_signals = np.array(pairwise_signals)
    signal_ordering = np.array(signal_ordering)
    
    return pairwise_signals, signal_ordering

def cluster_each_cell(signals,mask, cell_to_electrode_list, electrode_list, noise, note, cluster_delay = 7, damping=0.5, random_state=0):
    """
    Purpose: To get signal clustering across the relevant electrodes for a list of cells
    """
    all_event_cliques = []
    for cell in cell_to_electrode_list.keys():

        # Get relevant electrodes for cell
        cell_electrode_list = cell_to_electrode_list[cell]

        # Find relevant indices in multi-cell electrode list
        electrode_list_ind = np.array([j for j, e in enumerate(electrode_list) if e in cell_electrode_list])

        # Use AP to cluster over relevant electrodes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ap = sklearn.cluster.AffinityPropagation(damping=damping, random_state=random_state)
            event_labels = ap.fit_predict(signals[:, electrode_list_ind, cluster_delay:].reshape((len(signals), -1)) )              
        
        # Get mask over relevant electrodes
        mask_ = mask[electrode_list_ind]

        # Merge clusters that are sufficiently close
        event_labels = merge_clusters_by_noise(cell_electrode_list,signals[:,electrode_list_ind], event_labels, mask_, noise)

        # Generate list of sets containing clustered signals
        event_cliques = [set([i for i, x in enumerate(event_labels == l) if x]) for l in set(event_labels)]

        # Store cliques for each cell
        all_event_cliques += [event_cliques]

    return all_event_cliques

def DAG_estimation(event_labels, electrode_list, signals_tmp, mask_tmp, iterations, noise, data_on_cells, artifact_cluster_estimate = None, raw = False):
    """
    Main run script
    """

    signals = copy.copy(signals_tmp)
    mask = copy.copy(mask_tmp)
    
    note = ""
    
    # Create nodes for explanation graph
    G = nx.DiGraph()
    G.add_nodes_from(list(set(event_labels)))

    constrained = 'gof'
   
    # If all cliques have been merged, quit
    if len(set(event_labels))==1 or iterations == 0:
        note += " early completion, 1 cluster"
        return G, signals, {}, note

    # Compute difference signals
    difference_signals, event_signal_ordering =compute_difference_signals(signals, event_labels)
    
   
    # # Clean the difference signals
    # difference_signals = clean_signals(difference_signals, noise=noise, electrode_list=electrode_list, thr = 1, mode="svd")

    # Find candidate edges in graph from comparing templates and difference signals
    if not raw:
        edge_cell_to_fit_info, edge_cell_to_signal, edge_cell_to_error, edge_cell_to_loss, edge_cell_to_noise_loss, edge_cell_to_latency = find_candidate_edges(electrode_list, data_on_cells, difference_signals, event_signal_ordering,  mask, event_labels, noise, constrained = constrained, artifact_estimate = artifact_cluster_estimate)
    else:
        edge_cell_to_fit_info, edge_cell_to_signal, edge_cell_to_error, edge_cell_to_loss, edge_cell_to_noise_loss, edge_cell_to_latency = find_candidate_edges_raw(electrode_list, data_on_cells, difference_signals, event_signal_ordering,  mask, event_labels, noise, constrained = constrained, artifact_estimate = artifact_cluster_estimate)
    sorted_edges = sorted(edge_cell_to_loss.items(), key=lambda item: item[1])
    
    finished = False

    
    # Add edges that satisfy contraints: no improper chains, cycles, and (in simple mode) v-structures
    while len(sorted_edges) != 0:
        e_c, _ = sorted_edges.pop(0)
        e = e_c[0]
        cell = e_c[1]
        edge_data = nx.get_edge_attributes(G,'cell')
        head = e[0]
        tail = e[1]
        skip = False
        for u, v, ddict in G.in_edges(head, data=True):
            if cell == ddict['cell']:
                skip = True
  
        predecessors_nodes = list(G.predecessors(tail))
        predecessors_edges = [(tail, s) for s in predecessors_nodes]
        
        predecessors_edges_cells = []
        while len(predecessors_edges) != 0:
            p_e = predecessors_edges.pop(0)
            predecessors_edges_cells += [edge_data[(p_e[1], p_e[0])]]
            predecessors_nodes = list(G.predecessors(p_e[1]))
            predecessors_edges += [(p_e[1], p) for p in predecessors_nodes]
       
        successors_nodes = list(G.successors(head))
        successors_edges = [(s, head) for s in successors_nodes]
        
            
        successors_edges_cells = []
        while len(successors_edges) != 0:
            s_e = successors_edges.pop(0)
          
            successors_edges_cells += [edge_data[(s_e[1], s_e[0])]]
            successors_nodes = list(G.successors(s_e[0]))
            successors_edges += [(s, s_e[0]) for s in successors_nodes]
        
            
        if cell in successors_edges_cells:
            skip = True
        elif cell in predecessors_edges_cells:
            skip = True
        elif len(set(successors_edges_cells).intersection(predecessors_edges_cells)) != 0:
            skip = True
            
        if G.in_degree(head) > 0:
            skip = True
            
        if not skip: 
            G.add_edge(tail, head, cell = cell)
            
        if len(list(nx.simple_cycles(G))) != 0:
            G.remove_edge(tail, head)
            skip = True
            
        for n in G.nodes:
            if len(nx.descendants(G, n)) + 1 == len(G.nodes):
                root = n
                finished = True

        if finished:
            break
   
            
        
    # Compute residual/root signal    
    edge_data = nx.get_edge_attributes(G,'cell')
    final_signals = copy.copy(signals)
    for n in list(reversed(list(nx.topological_sort(G)))):
        if len(list(G.in_edges(n))) == 0:
            continue
        u, v = list(G.in_edges(n))[0]
        cell = edge_data[(u,v)]
        sig = edge_cell_to_signal[((v, u), cell)]
        shifts = get_shifts(final_signals,event_labels, v, u)                       
        shifted_sig = shift_sig(sig,shifts)
        final_signals[event_labels == v] -= shifted_sig
        
        event_labels = np.array( [u if l == v else l for l in event_labels] )
    
    edge_to_matched_signals = {}
    
    # Store graph related info
    event_signal_ordering_tup = list(map(tuple, event_signal_ordering))
    for tail, head in G.edges:
        cell = edge_data[(tail,head)]
        d_sig = difference_signals[event_signal_ordering_tup.index((head, tail))]
        c_sig = edge_cell_to_signal[((head, tail), cell)]
        edge_to_matched_signals[((tail, head), cell)] = (d_sig, c_sig, edge_cell_to_loss[((head, tail), cell)], edge_cell_to_fit_info[((head, tail), cell)][2],edge_cell_to_fit_info[((head, tail), cell)][3], edge_cell_to_latency[((head, tail), cell)])
    
    return G, final_signals, edge_to_matched_signals, note

def direct_similarity_wueric(electrode_list: List[int],data_on_cells: Tuple[List[int], np.ndarray, np.ndarray, np.ndarray],difference_signals: np.ndarray,mask: np.ndarray,event_pop: np.ndarray,noise: np.ndarray):
    """
    Alternative approach to direct_similarity, computes the correlations directly
        with the loops in C++ rather than Python.

    Single-threaded.
    :param electrode_list: list of electrode indices
    :param data_on_cells: 4-tuple, elements are

        * cell_ids, List[int] of cell ids
        * cell_eis, np.ndarray of shape (n_cells, n_electrodes, n_timepoints_ei),
            contains reduced EIs of the cells
        * cell_variance, np.ndarray of shape (n_cells, n_electrodes, n_timepoints_ei),
        * ??

    :param difference_signals: np.ndarray, shape (2, n_electrodes, n_timepoints_diff_sig)
    :param mask: np.ndarray, shape (n_electrodes, n_timepoints_diff_sig)
    :param event_pop: np.ndarray
    :param noise: np.ndarray, shape (n_electrodes_total, ), noise for every channel in the array

    output: np.array, dict<key: int, value: np.array>, np.array


    """

    # cell_ids has length n_cells
    # cell_eis has shape (n_cells, n_electrodes, n_timepoints_ei)
    # cell_variance has shape (n_cells, n_electrodes, n_timepoints_ei)
    cell_ids, cell_eis, cell_variance, _ = data_on_cells

    n_cells, n_electrodes, n_timepoints_ei = cell_eis.shape

    cell_precision = 1.0 / cell_variance # shape (n_cells, n_electrodes, n_timepoints_ei)

    # shape (num_difference_signals, n_electrodes, n_timepoints_diff_sig)
    masked_difference_signal = mask[None, :, :] * difference_signals

    # shape (n_cells, n_electrodes, n_timepoints_ei)
    cell_eis_mul_prec = cell_eis * cell_precision

    # shape (n_cells, n_electrodes, n_timepoints_ei)
    cell_eis2_mul_prec = cell_eis * cell_eis_mul_prec


    # shape (n_cells, num_difference_signals, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    aTa_parallel_corr = corr1d.batch_filter_batch_data_channel_correlate1D(
        cell_precision, (masked_difference_signal * masked_difference_signal)
    )

    # shape (n_cells, 1, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    bTb_parallel_corr = corr1d.batch_filter_batch_data_channel_correlate1D(
        cell_eis2_mul_prec, mask[None, :, :]
    )

    # shape (n_cells, num_difference_signals, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    two_aTb_parallel_corr = corr1d.batch_filter_batch_data_channel_correlate1D(
        cell_eis_mul_prec, 2 * masked_difference_signal
    )

    event_pop_axis_sum = np.sum(event_pop, axis=1) # shape (2, )
    log_event_pop_sum = np.log(event_pop_axis_sum) # shape (2, )

    # shape (n_cells, )
    var_bump = np.sum(np.log(cell_variance), axis=(1, 2))[None, :] + \
                (n_electrodes * n_timepoints_ei * log_event_pop_sum[:, None])

    # shape (n_cells, num_difference_signals, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    normalized_mse_all_no_event_pop = aTa_parallel_corr + bTb_parallel_corr - two_aTb_parallel_corr

    # shape (n_cells, num_difference_signals, n_electrodes)
    min_mse_index = np.argmin(normalized_mse_all_no_event_pop, axis=3)

    # shape (n_cells, num_difference_signals, n_electrodes, 1) -> (n_cells, num_difference_signals, n_electrodes)
    min_mse_no_event_pop = np.take_along_axis(normalized_mse_all_no_event_pop,
                                              np.expand_dims(min_mse_index, 3), axis=3).squeeze(-1)
    # shape (n_cells, num_difference_signals) -> (num_difference_signals, n_cells)
    normalized_mse_total = np.sum(min_mse_no_event_pop, axis=2).transpose(1, 0) / event_pop_axis_sum[:, None]

    # shape (num_difference_signals, n_cells) + (num_difference_signals, n_cells) -> (num_difference_signals, n_cells)
    neg_log_likelihood_total = normalized_mse_total + var_bump

    # shape (n_cells, num_difference_signals, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    # -> (n_cells, num_difference_signals, n_electrodes) -> (n_cells, num_difference_signals) -> (num_difference_signals, n_cells)
    noise_neg_log_likelihood_total = np.sum(np.min(aTa_parallel_corr, axis=-1), axis=-1).transpose(1, 0) / event_pop_axis_sum[:, None]

    idxs_total = {el_id : min_mse_index[:,:, idx].T for idx, el_id in enumerate(electrode_list)}

    return normalized_mse_total, idxs_total, neg_log_likelihood_total, noise_neg_log_likelihood_total

def direct_similarity_wueric_fixed_shifts(electrode_list: List[int],data_on_cells: Tuple[List[int], np.ndarray, np.ndarray, np.ndarray],difference_signals: np.ndarray,mask: np.ndarray,event_pop: np.ndarray,noise: np.ndarray):
    """
    Alternative approach to direct_similarity, computes the correlations directly
        with the loops in C++ rather than Python.

    Single-threaded.
    :param electrode_list: list of electrode indices
    :param data_on_cells: 4-tuple, elements are

        * cell_ids, List[int] of cell ids
        * cell_eis, np.ndarray of shape (n_cells, n_electrodes, n_timepoints_ei),
            contains reduced EIs of the cells
        * cell_variance, np.ndarray of shape (n_cells, n_electrodes, n_timepoints_ei),
        * ??

    :param difference_signals: np.ndarray, shape (2, n_electrodes, n_timepoints_diff_sig)
    :param mask: np.ndarray, shape (n_electrodes, n_timepoints_diff_sig)
    :param event_pop: np.ndarray
    :param noise: np.ndarray, shape (n_electrodes_total, ), noise for every channel in the array

    output: np.array, dict<key: int, value: np.array>, np.array


    """

    # cell_ids has length n_cells
    # cell_eis has shape (n_cells, n_electrodes, n_timepoints_ei)
    # cell_variance has shape (n_cells, n_electrodes, n_timepoints_ei)
    cell_ids, cell_eis, cell_variance, _, fixed_shifts_ = data_on_cells

    fixed_shifts = fixed_shifts_[:,None,:,None]

    n_cells, n_electrodes, n_timepoints_ei = cell_eis.shape

    cell_precision = 1.0 / cell_variance # shape (n_cells, n_electrodes, n_timepoints_ei)

    # shape (2, n_electrodes, n_timepoints_diff_sig)
    masked_difference_signal = mask[None, :, :] * difference_signals

    # shape (n_cells, n_electrodes, n_timepoints_ei)
    cell_eis_mul_prec = cell_eis * cell_precision

    # shape (n_cells, n_electrodes, n_timepoints_ei)
    cell_eis2_mul_prec = cell_eis * cell_eis_mul_prec


    # shape (n_cells, 2, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    aTa_parallel_corr = corr1d.batch_filter_batch_data_channel_correlate1D(
        cell_precision, (masked_difference_signal * masked_difference_signal)
    )

    # shape (n_cells, 1, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    bTb_parallel_corr = corr1d.batch_filter_batch_data_channel_correlate1D(
        cell_eis2_mul_prec, mask[None, :, :]
    )

    # shape (n_cells, 2, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    two_aTb_parallel_corr = corr1d.batch_filter_batch_data_channel_correlate1D(
        cell_eis_mul_prec, 2 * masked_difference_signal
    )

    event_pop_axis_sum = np.sum(event_pop, axis=1) # shape (2, )
    log_event_pop_sum = np.log(event_pop_axis_sum) # shape (2, )

    # shape (n_cells, )
    var_bump = np.sum(np.log(cell_variance), axis=(1, 2))[None, :] + \
                (n_electrodes * n_timepoints_ei * log_event_pop_sum[:, None])

    # shape (n_cells, 2, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    normalized_mse_all_no_event_pop = aTa_parallel_corr + bTb_parallel_corr - two_aTb_parallel_corr
    
    # score_fixed_shifts = np.sum(fixed_shifts * normalized_mse_all_no_event_pop, axis=2)
    # min_mse_index_fixed_shifts = np.argmin(score_fixed_shifts, axis=2)
    
    # score_non_fixed_shifts = (1-fixed_shifts) * normalized_mse_all_no_event_pop
    # min_mse_index_non_fixed_shifts = np.argmin(score_non_fixed_shifts, axis=3)
    
    # # shape (n_cells, 2, n_electrodes)
    # # min_mse_index = np.argmin(normalized_mse_all_no_event_pop, axis=3)
    # for i in range(n_cells):
    #     min_mse_index_non_fixed_shifts[i][:,fixed_shifts[i]==1] =  min_mse_index_fixed_shifts[i][fixed_shifts[i]==1]
    # min_mse_index = min_mse_index_non_fixed_shifts

    # shape (n_cells, 2, n_electrodes, 1) -> (n_cells, 2, n_electrodes)
    min_mse_no_event_pop = np.take_along_axis(normalized_mse_all_no_event_pop,
                                              np.expand_dims(min_mse_index, 3), axis=3).squeeze(-1)

    # shape (n_cells, 2) -> (2, n_cells)
    normalized_mse_total = np.sum(min_mse_no_event_pop, axis=2).transpose(1, 0) / event_pop_axis_sum[:, None]

    # shape (2, n_cells) + (2, n_cells) -> (2, n_cells)
    neg_log_likelihood_total = normalized_mse_total + var_bump

    # shape (n_cells, 2, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    # -> (n_cells, 2, n_electrodes) -> (n_cells, 2) -> (2, n_cells)
    noise_neg_log_likelihood_total = np.sum(np.min(aTa_parallel_corr, axis=-1), axis=-1).transpose(1, 0) / event_pop_axis_sum[:, None]

    idxs_total = {el_id : min_mse_index[:,:, idx].T for idx, el_id in enumerate(electrode_list)}

    return normalized_mse_total, idxs_total, neg_log_likelihood_total, noise_neg_log_likelihood_total


def direct_similarity_raw_wueric(electrode_list: List[int],data_on_cells: Tuple[List[int], np.ndarray, np.ndarray, np.ndarray],difference_signals: np.ndarray,mask: np.ndarray,event_pop: np.ndarray,noise: np.ndarray):
    """
    Alternative approach to direct_similarity, computes the correlations directly
        with the loops in C++ rather than Python.

    Single-threaded.

    :param electrode_list: list of electrode indices
    :param data_on_cells: 4-tuple, elements are

        * cell_ids, List[int] of cell ids
        * cell_eis, np.ndarray of shape (n_cells, n_electrodes, n_timepoints_ei),
            contains reduced EIs of the cells
        * cell_variance, np.ndarray of shape (n_cells, n_electrodes, n_timepoints_ei),
        * ??

    :param difference_signals: np.ndarray, shape (2, n_electrodes, n_timepoints_diff_sig)
    :param mask: np.ndarray, shape (n_electrodes, n_timepoints_diff_sig)
    :param event_pop: np.ndarray
    :param noise: np.ndarray, shape (n_electrodes_total, ), noise for every channel in the array

    output: np.array, dict<key: int, value: np.array>, np.array


    """

    # cell_ids has length n_cells
    # cell_eis has shape (n_cells, n_electrodes, n_timepoints_ei)
    # cell_variance has shape (n_cells, n_electrodes, n_timepoints_ei)
    cell_ids, cell_eis, cell_variance, _ = data_on_cells
    
    n_cells, n_electrodes, n_timepoints_ei = cell_eis.shape

    cell_precision = np.ones_like(cell_variance)# shape (n_cells, n_electrodes, n_timepoints_ei)

    # shape (2, n_electrodes, n_timepoints_diff_sig)
    masked_difference_signal = mask[None, :, :] * difference_signals

    # shape (n_cells, n_electrodes, n_timepoints_ei)
    cell_eis_mul_prec = cell_eis * cell_precision

    # shape (n_cells, n_electrodes, n_timepoints_ei)
    cell_eis2_mul_prec = cell_eis * cell_eis_mul_prec


    # shape (n_cells, 2, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    aTa_parallel_corr = corr1d.batch_filter_batch_data_channel_correlate1D(
        cell_precision, (masked_difference_signal * masked_difference_signal)
    )

    # shape (n_cells, 1, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    bTb_parallel_corr = corr1d.batch_filter_batch_data_channel_correlate1D(
        cell_eis2_mul_prec, mask[None, :, :]
    )

    # shape (n_cells, 2, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    two_aTb_parallel_corr = corr1d.batch_filter_batch_data_channel_correlate1D(
        cell_eis_mul_prec, 2 * masked_difference_signal
    )

    event_pop_axis_sum = np.sum(event_pop, axis=1) # shape (2, )
    log_event_pop_sum = np.log(event_pop_axis_sum) # shape (2, )

    # shape (n_cells, )
    var_bump = np.sum(np.log(cell_variance), axis=(1, 2))[None, :] + \
                (n_electrodes * n_timepoints_ei * log_event_pop_sum[:, None])

    # shape (n_cells, 2, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    normalized_mse_all_no_event_pop = aTa_parallel_corr + bTb_parallel_corr - two_aTb_parallel_corr

    # shape (n_cells, 2, n_electrodes)
    min_mse_index = np.argmin(normalized_mse_all_no_event_pop, axis=3)

    # shape (n_cells, 2, n_electrodes, 1) -> (n_cells, 2, n_electrodes)
    min_mse_no_event_pop = np.take_along_axis(normalized_mse_all_no_event_pop,
                                              np.expand_dims(min_mse_index, 3), axis=3).squeeze(-1)

    # shape (n_cells, 2) -> (2, n_cells)
    normalized_mse_total = np.sum(min_mse_no_event_pop, axis=2).transpose(1, 0)

    # shape (2, n_cells) + (2, n_cells) -> (2, n_cells)
    neg_log_likelihood_total = normalized_mse_total #+ var_bump

    # shape (n_cells, 2, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    # -> (n_cells, 2, n_electrodes) -> (n_cells, 2) -> (2, n_cells)
    noise_neg_log_likelihood_total = np.sum(np.min(aTa_parallel_corr, axis=-1), axis=-1).transpose(1, 0) 

    idxs_total = {el_id : min_mse_index[:,:, idx].T for idx, el_id in enumerate(electrode_list)}

    return normalized_mse_total, idxs_total, neg_log_likelihood_total, noise_neg_log_likelihood_total

def find_bad_edges(cell, edge_to_matched_signals,  mask, data_on_cells, electrode_list, noise):
    """
    Purpose: Identify bad edges according to three criteria:
        1. The cosine similarity between the difference signals and the assigned template is too low
        2. The assigned template's latency is too early or too late
        3. The assigned template's is too small
    """
    bad_edges = []
    bad_edges = compute_cosine_error(mask, edge_to_matched_signals, cell, bad_edges)
    peak_electrode = np.argmax(np.max(np.abs(data_on_cells[1][data_on_cells[0].index(cell)]), axis = 1))
    bad_edges = compute_latency_error(edge_to_matched_signals, cell, bad_edges, peak_electrode, offset = data_on_cells[-1][0,0], too_early = 5)
    bad_edges = compute_small_signal_error(edge_to_matched_signals, cell, bad_edges, peak_electrode, too_small = noise[electrode_list][peak_electrode])
    return bad_edges

def find_candidate_edges(electrode_list, data_on_cells, difference_signals, difference_signal_ordering, mask, event_labels, noise, constrained = 'gof',  p = 0.99, max_electrodes_oor = 2, valid_shift_range = 35, artifact_estimate = None, plot = True, no_time = False):
    '''
    Generate potential edges in graph. 
    
    n: int
    electrode_list: list<int>
    data_on_cells: tuple<list<int>, np.array, np.array, np.array >
    difference_signals: np.array,
    difference_signal_ordering: np.array,
    state_to_cell: dict<key: int, value: list<tuple<int, int> > >
    mask: np.array
    event_labels: list<int>
    noise: list<float>

    output: dict<key: tuple<int, int>, value: list<tuple<int, float, int> > >, dict<key: tuple<int, int>, value: np.array >, dict<key: tuple<int, int>, value: np.array >, dict<key: tuple<int, int>, value: float >
    '''
    num_difference_signals = len(difference_signal_ordering)
    
    edge_cell_to_fit_info = {}
    edge_cell_to_sig = {}
    edge_cell_to_error = {}
    edge_cell_to_loss = {}
    edge_cell_to_noise_loss = {}
    edge_cell_to_latency = {}
    
    if len(difference_signals) == 0:
        return edge_cell_to_fit_info, edge_cell_to_sig, edge_cell_to_error, edge_cell_to_loss, edge_cell_to_noise_loss, edge_cell_to_latency

    # Chi-squared determined goodness-of-fit threshold for noise-normalized MSE between templates and difference signals
    goodness_of_fit_threshold = sum([scipy.stats.chi2.ppf(p, sum(mask[i])) for i in range(len(mask)) if sum(mask[i]) != 0])

    # Load up template information
    cell_ids, cell_eis, cell_variance, peak_st = data_on_cells
    num_cells = len(cell_ids)
    central_time = peak_st[0,0]
     
    # Compute noise-normalize mse, the best shift, and negative log-likehood
    event_counts = Counter(event_labels)
   
    event_pop = np.array([[1/event_counts[e[0]],1/event_counts[e[1]]] for e in difference_signal_ordering])
    if not no_time:
        normalized_mse, best_match_idx, negative_log_likelihood, noise_negative_log_likelihood = direct_similarity_wueric(electrode_list, data_on_cells, difference_signals,  mask, event_pop, noise)
    else:
        cell_eis_space_ind = np.argmin(cell_eis, axis = 2)[:,:,None]
        cell_eis_space = np.take_along_axis(cell_eis, cell_eis_space_ind, axis =2)
        cell_variance_space = np.take_along_axis(cell_variance, cell_eis_space_ind, axis =2)
        data_on_cells_space = (cell_ids, cell_eis_space, cell_variance_space, peak_st)
        difference_signals_space = np.min(difference_signals, axis = 2)[:,:,None]
        normalized_mse, best_match_idx, negative_log_likelihood, noise_negative_log_likelihood = direct_similarity_wueric(electrode_list, data_on_cells_space, difference_signals_space,  np.ones((len(mask), 1)), event_pop, noise)
        

    # Choose different signal orientation based on maxmimum likehood
    max_likelihood = np.min(negative_log_likelihood, axis = 1)
    if artifact_estimate != None:
        negative_log_likelihood[difference_signal_ordering[:, 0] == artifact_estimate] = np.inf
        normalized_mse[difference_signal_ordering[:, 0] == artifact_estimate] = np.inf
    
    difference_signal_ordering_rep = list(map(tuple, np.repeat(difference_signal_ordering, len(cell_ids), axis = 0)))
    
    normalized_mse = normalized_mse.flatten()
    best_match_idx = {k:v.flatten() for k, v in best_match_idx.items()}
    negative_log_likelihood = negative_log_likelihood.flatten()
    noise_negative_log_likelihood = noise_negative_log_likelihood.flatten()
   
    num_samples = difference_signals.shape[2]
    best_match_signals = np.zeros((len(negative_log_likelihood), len(electrode_list), num_samples))
    best_match_latency = np.zeros((len(negative_log_likelihood), len(electrode_list)))
    best_match_full = np.zeros((len(negative_log_likelihood), len(electrode_list), cell_eis.shape[2]))
    best_match_variance = np.zeros((len(negative_log_likelihood), len(electrode_list), num_samples))
    
    # Take out the best fitting slice from the template for the corresponding difference signal
    # For each electrode, 
    cell_indices = np.tile(np.arange(len(cell_ids)), num_difference_signals)
    for i, e in enumerate(electrode_list):
        relevant_indices_start = np.array(best_match_idx[e])
        
        cell_subset = cell_eis[cell_indices, i, :]
        variance_subset = cell_variance[cell_indices, i, :]
        
        # print("relevant_indices_start", relevant_indices_start)

        # Slice out the relevant points from the template and template error according to the shift information from 'direct_similarity'
        best_match_signals[:, i, :] = np.array([[cell_subset[j, i] for i in range(l, l +num_samples)] for j, l in enumerate(relevant_indices_start)])
        best_match_latency[:, i] = relevant_indices_start
        
        best_match_variance[:, i, :] = np.array([[variance_subset[j, i] for i in range(l , l+num_samples)] for j, l in enumerate(relevant_indices_start)])   

        best_match_full[:, i, :] = cell_subset
    # For each matched cell, 
    for i, pair in enumerate(difference_signal_ordering_rep):
               
        edge_cell_to_fit_info[(pair, cell_ids[cell_indices[i]])] = (negative_log_likelihood[i], noise_negative_log_likelihood[i], normalized_mse[i],goodness_of_fit_threshold)
        edge_cell_to_sig[(pair, cell_ids[cell_indices[i]])] = best_match_signals[i]
        edge_cell_to_error[(pair, cell_ids[cell_indices[i]])] = best_match_variance[i]
        edge_cell_to_loss[(pair, cell_ids[cell_indices[i]])] = negative_log_likelihood[i]
        edge_cell_to_noise_loss[(pair, cell_ids[cell_indices[i]])] = noise_negative_log_likelihood[i]
        edge_cell_to_latency[(pair, cell_ids[cell_indices[i]])] = best_match_latency[i]
            
           
    return edge_cell_to_fit_info, edge_cell_to_sig, edge_cell_to_error, edge_cell_to_loss, edge_cell_to_noise_loss, edge_cell_to_latency
        
def find_candidate_edges_raw(electrode_list, data_on_cells, difference_signals, difference_signal_ordering, mask, event_labels, noise, constrained = 'gof',  p = 0.99, max_electrodes_oor = 2, valid_shift_range = 35, artifact_estimate = None, plot = True, no_time = False):
    '''
    Generate potential edges in graph. 
    
    n: int
    electrode_list: list<int>
    data_on_cells: tuple<list<int>, np.array, np.array, np.array >
    difference_signals: np.array,
    difference_signal_ordering: np.array,
    state_to_cell: dict<key: int, value: list<tuple<int, int> > >
    mask: np.array
    event_labels: list<int>
    noise: list<float>

    output: dict<key: tuple<int, int>, value: list<tuple<int, float, int> > >, dict<key: tuple<int, int>, value: np.array >, dict<key: tuple<int, int>, value: np.array >, dict<key: tuple<int, int>, value: float >
    '''
    num_difference_signals = len(difference_signal_ordering)
    
    edge_cell_to_fit_info = {}
    edge_cell_to_sig = {}
    edge_cell_to_error = {}
    edge_cell_to_loss = {}
    edge_cell_to_noise_loss = {}
    edge_cell_to_latency = {}
    
    if len(difference_signals) == 0:
        return edge_cell_to_fit_info, edge_cell_to_sig, edge_cell_to_error, edge_cell_to_loss, edge_cell_to_noise_loss, edge_cell_to_latency

    # Chi-squared determined goodness-of-fit threshold for noise-normalized MSE between templates and difference signals
    goodness_of_fit_threshold = sum([scipy.stats.chi2.ppf(p, sum(mask[i])) for i in range(len(mask)) if sum(mask[i]) != 0])

    # Load up template information
    cell_ids, cell_eis, cell_variance, peak_st = data_on_cells
    num_cells = len(cell_ids)
    central_time = peak_st[0,0]
     
    # Compute noise-normalize mse, the best shift, and negative log-likehood
    event_counts = Counter(event_labels)
   
    event_pop = np.array([[1/event_counts[e[0]],1/event_counts[e[1]]] for e in difference_signal_ordering])
    if not no_time:
        normalized_mse, best_match_idx, negative_log_likelihood, noise_negative_log_likelihood = direct_similarity_raw_wueric(electrode_list, data_on_cells, difference_signals,  mask, event_pop, noise)
    else:
        cell_eis_space_ind = np.argmin(cell_eis, axis = 2)[:,:,None]
        cell_eis_space = np.take_along_axis(cell_eis, cell_eis_space_ind, axis =2)
        cell_variance_space = np.take_along_axis(cell_variance, cell_eis_space_ind, axis =2)
        data_on_cells_space = (cell_ids, cell_eis_space, cell_variance_space, peak_st)
        difference_signals_space = np.min(difference_signals, axis = 2)[:,:,None]
        normalized_mse, best_match_idx, negative_log_likelihood, noise_negative_log_likelihood = direct_similarity_wueric(electrode_list, data_on_cells_space, difference_signals_space,  np.ones((len(mask), 1)), event_pop, noise)
        

    # Choose different signal orientation based on maxmimum likehood
    max_likelihood = np.min(negative_log_likelihood, axis = 1)
    if artifact_estimate != None:
        negative_log_likelihood[difference_signal_ordering[:, 0] == artifact_estimate] = np.inf
        normalized_mse[difference_signal_ordering[:, 0] == artifact_estimate] = np.inf
    
    difference_signal_ordering_rep = list(map(tuple, np.repeat(difference_signal_ordering, len(cell_ids), axis = 0)))
    
    normalized_mse = normalized_mse.flatten()
    best_match_idx = {k:v.flatten() for k, v in best_match_idx.items()}
    negative_log_likelihood = negative_log_likelihood.flatten()
    noise_negative_log_likelihood = noise_negative_log_likelihood.flatten()
   
    num_samples = difference_signals.shape[2]
    best_match_signals = np.zeros((len(negative_log_likelihood), len(electrode_list), num_samples))
    best_match_latency = np.zeros((len(negative_log_likelihood), len(electrode_list)))
    best_match_full = np.zeros((len(negative_log_likelihood), len(electrode_list), cell_eis.shape[2]))
    best_match_variance = np.zeros((len(negative_log_likelihood), len(electrode_list), num_samples))
    
    # Take out the best fitting slice from the template for the corresponding difference signal
    # For each electrode, 
    cell_indices = np.tile(np.arange(len(cell_ids)), num_difference_signals)
    for i, e in enumerate(electrode_list):
        relevant_indices_start = np.array(best_match_idx[e])
        
        cell_subset = cell_eis[cell_indices, i, :]
        variance_subset = cell_variance[cell_indices, i, :]
        
        # print("relevant_indices_start", relevant_indices_start)

        # Slice out the relevant points from the template and template error according to the shift information from 'direct_similarity'
        best_match_signals[:, i, :] = np.array([[cell_subset[j, i] for i in range(l, l +num_samples)] for j, l in enumerate(relevant_indices_start)])
        best_match_latency[:, i] = relevant_indices_start
        
        best_match_variance[:, i, :] = np.array([[variance_subset[j, i] for i in range(l , l+num_samples)] for j, l in enumerate(relevant_indices_start)])   

        best_match_full[:, i, :] = cell_subset
    # For each matched cell, 
    for i, pair in enumerate(difference_signal_ordering_rep):
               
        edge_cell_to_fit_info[(pair, cell_ids[cell_indices[i]])] = (negative_log_likelihood[i], noise_negative_log_likelihood[i], normalized_mse[i],goodness_of_fit_threshold)
        edge_cell_to_sig[(pair, cell_ids[cell_indices[i]])] = best_match_signals[i]
        edge_cell_to_error[(pair, cell_ids[cell_indices[i]])] = best_match_variance[i]
        edge_cell_to_loss[(pair, cell_ids[cell_indices[i]])] = negative_log_likelihood[i]
        edge_cell_to_noise_loss[(pair, cell_ids[cell_indices[i]])] = noise_negative_log_likelihood[i]
        edge_cell_to_latency[(pair, cell_ids[cell_indices[i]])] = best_match_latency[i]
            
           
    return edge_cell_to_fit_info, edge_cell_to_sig, edge_cell_to_error, edge_cell_to_loss, edge_cell_to_noise_loss, edge_cell_to_latency


def get_mask(signals, sat_band = 5,unsatured_min= -850, unsaturated_max = 400):
       # Generate mask to deal with saturation
    num_samples = signals.shape[-1]
    signals_max = np.max(signals, axis = 0)
    signals_min = np.min(signals, axis = 0)
    mask = ((signals_max < unsaturated_max)*(signals_min > unsatured_min)).astype(np.float64)
    mask_loc = np.argwhere(1-mask)
    for m in mask_loc:
        mask[m[0],max(0,m[1]-sat_band):min(m[1]+sat_band, num_samples)] = 0
    return mask

def get_probabilities(G, event_labels, num_trials = None):
    G_mod = G.copy()
    for n in [n for n in G_mod.nodes if G_mod.in_degree(n) > 1]:
        edge_list = list(G_mod.in_edges(n))[1:]
        G_mod.remove_edges_from(edge_list)

    cells_on_graph = nx.get_edge_attributes(G_mod, 'cell').values()
    edge_to_cells = nx.get_edge_attributes(G_mod, 'cell')
    cells_to_edges = {v: [ j for j, l in edge_to_cells.items() if l == v] for k, v in edge_to_cells.items()}
    probabilities = Counter()
    event_counter = Counter(event_labels)
    for c in set(list(cells_on_graph)):
        for u, v in cells_to_edges[c]:
            probabilities[c] += event_counter[v]
            for n in nx.descendants(G_mod, v):
                probabilities[c] += event_counter[n]
        if num_trials:
            probabilities[c] /= num_trials
        else:
            probabilities[c] /= len(event_labels)
    return probabilities

def get_shifts(signals,event_labels, m, n):
    """
    Compute difference signal shifts
    signals: np.array
    event_labels: list<int>
    m, n: int

    output: np.array
    """
    
    # Compute pairwise differences between (updating) event signal cliques
    _, num_electrodes, num_samples = signals.shape
    X = signals[event_labels == m]
    Y = np.mean(signals[event_labels == n], axis = 0, keepdims = True)
    Z = all_pairwise(X, Y)
    Z = Z.reshape((len(X), len(Y), num_electrodes, num_samples))
    Z = np.argmin(Z, axis = 3)
    Z = np.median(Z, axis = 1)
    shifts = Z.T
    
    return shifts

def get_alignment(X_, window = 10, res = 2, sample_len = 55):
    """
    Align a set of set of signals to approximatetly minimize global l2 distance
    X: np.array
    window: int, default = 10

    output: tuple<np.array, tuple<int, int> >

    """
    
    # Special case: Only one signal in signal set
    X = X_.reshape((-1,sample_len))
        
    high_res_X = interpolate(X,res = 2)
    
    min_index = np.argmin(X,axis = 1)
    
    # Find signals with median sample of either depolarization or hyperpolarization hump (if there is a cell firing) i.e. find an average signal
    c_i = np.argsort(min_index)[len(min_index)//2]
    
    # Store median signal to align all others to
    x = high_res_X[c_i:c_i + 1,::-1]
    
    one_window = np.zeros_like(x)
    one_window[:,::res] = 1
    one_window = one_window[:,::-1]
    
    # Compute l2 difference between reference and all other signals using convolution
    _, signal_len = x.shape
    conv_Xx = scipy.signal.convolve(high_res_X, x*one_window, mode = 'full')[:,(2*signal_len-1)//2-window*res:(2*signal_len-1)//2+window*res]
    conv_X21 = scipy.signal.convolve(high_res_X**2, one_window, mode = 'full')[:,(2*signal_len-1)//2-window*res:(2*signal_len-1)//2+window*res]
    conv_x21 = scipy.signal.convolve(x**2, one_window, mode = 'full')[:,(2*signal_len-1)//2-window*res:(2*signal_len-1)//2+window*res]
    
    # Indentify and shift signals to approximately minimize global l2 distance   
    amin_Xx = -(np.argmin( conv_X21 + conv_x21 - 2*conv_Xx, axis = 1) - window*res)

    return c_i, amin_Xx



def interpolate(X_,res = 2, sample_len = 55):
    # Upsample data for better signal alignment
    X = X_.reshape((-1,sample_len))
    x = np.arange(0, len(X[0]))
    interpolant = scipy.interpolate.interp1d(x,X, kind='cubic')
    domain = np.linspace(0, len(x)-1, len(x)*res)
    high_res_X = interpolant(domain)
    return high_res_X

def merge_clusters_by_noise(electrode_list, signals, event_labels_tmp,  mask,  noise):
    """
    Iteratively merge event cliques according to average l2 distance.
    This implementation is a bit silly; will be improved later. 
    
    electrode_list: list<int>
    signals: np.array
    event_labels_tmp: list<int>
    state_to_cell: dict<key: int, value: list<tuple<int, int> > >
    mask: np.array
    final_g: networkx.Graph
    noise: list<float>

    output: list<int>, dict<key: int, value: list<tuple<int, int> > >, networkx.Graph
    """
    event_labels = copy.copy(event_labels_tmp)
    
    # Compute noise distance threshold
    _, _, num_samples = signals.shape

    std = np.repeat([noise[e] for e in electrode_list], num_samples-5)
    
    std = math.sqrt(2) * std[None]
    
    k = num_samples*len(electrode_list)
    
    merging = True
    while merging:
        event_set = list(set(event_labels))
        num_events = len(event_set)
        threshold_array = (np.ones((num_events, num_events)) - np.eye(num_events)) * k 
        
        # Compute average noise distance between event cliques
        distance = np.zeros((num_events, num_events))
        for node_pair in list(itertools.combinations(np.arange(num_events), 2 )):
            X = signals[event_labels == event_set[node_pair[0]],:,5:] * mask[None,:,5:]
            Y = signals[event_labels == event_set[node_pair[1]],:,5:] * mask[None,:,5:]

            Z = all_pairwise(X,Y)
            
            num_diff_signals = Z.shape[0]
            Z = Z.reshape((num_diff_signals, -1))
            distance[node_pair[0], node_pair[1]] = np.sum((Z/std)**2)/num_diff_signals
            
            distance[node_pair[1], node_pair[0]] = np.sum((Z/std)**2)/num_diff_signals
        edges = np.argwhere((distance < threshold_array) *(distance != 0))
        if len(edges) == 0:
            break

        shortest_edge = edges[np.argmin([distance[e[0],e[1]] for e in edges])]
        cluster_members = [i for i,x in enumerate(event_labels) if x==event_set[shortest_edge[0]]]
        
        for i in [i for i,x in enumerate(event_labels) if x==event_set[shortest_edge[1]]]:
            cluster_members += [i]
        
        for i in cluster_members:
            event_labels[i] = min(shortest_edge[0], shortest_edge[1])
    return event_labels

def run_pattern_movie_live(signal, preloaded_data):
    """
    Assuming run for a given pattern (p) and a given amplitude (k)

    signal: np.ndarray, tensor of shape T x E x N where T is number of trials, E is electrodes (512 or 519 + 1 for TTL),
                        and N is the number of time samples recorded
    preloaded_data: tuple containing
                    cell_data_dict: {cell_id: data on cell from get_center_eis()}
                    cells_to_gsort: cells to run gsort on for this p, k
                    noise: np.ndarray, visionloader channel noise array
                    _, total_cell_to_electrode_list, mutual_cells, _ = get_cell_info(all_cell_types, vstim_data, compartments, noise, mutual_threshold=mutual_threshold)                    
    """
    # Pre-computed data from calling script 
    cell_data_dict, cells_to_gsort, mutual_cells, total_cell_to_electrode_list, end_time_limit, start_time_limit, cluster_delay, noise = preloaded_data
    
    # Initialize output arrays
    probs = np.zeros(len(cells_to_gsort))
    num_trials = len(signal)

    # Iterate over all potential cells
    cell_ind = 0
    for cell in cells_to_gsort:

        # Verify the electrodes has some electrodes to compare against
        electrode_list =  list(set([e for c in mutual_cells[cell] for e in total_cell_to_electrode_list[c]]))
        
        if len(electrode_list):
            # Trucate signal to relevant region
            raw_signal = signal[:, electrode_list, start_time_limit:end_time_limit].astype(float) 
           
            
            # Get saturation mask
            mask =  get_mask(raw_signal, )
            
            # Get clustering
            cell_to_electrode_list = {k:v for k,v in total_cell_to_electrode_list.items() if k in mutual_cells[cell]}
            cluster_cliques = cluster_each_cell(raw_signal,mask, cell_to_electrode_list, electrode_list, noise, "", cluster_delay = cluster_delay)
            event_labels = convert_cliques_to_labels(cluster_cliques, num_trials)
            event_labels = merge_clusters_by_noise(electrode_list, raw_signal, event_labels,  mask, noise)

            # Run graph estimation algorithm
            data_on_cells = cell_data_dict[cell]
            G, final_signals, edge_to_matched_signals, note = DAG_estimation(event_labels, electrode_list, raw_signal, mask, 1, noise, data_on_cells)

            # Compute probabilities from graph
            cell_to_prob = get_probabilities(G, event_labels)
            
            # Find bad edges in graph
            bad_edges = find_bad_edges(cell, edge_to_matched_signals, mask, data_on_cells, electrode_list, noise)
            
            prob_error = compute_probability_error(G, event_labels, bad_edges)
            probs[cell_ind] = cell_to_prob[cell]-prob_error

        cell_ind += 1
            
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    return probs

def shift_sig(sig, shifts):
    """
    Shift difference signal sig according to shifts
    
    sig: np.array
    shifts: np.array
    
    output: np.array
    """
    peak_times = np.argmin(sig, axis=1)
    offsets = shifts - peak_times[:, None]
    shifted_signals = np.einsum('ijk->jik', np.array([np.array([shift(sig[i],s) for s in offset]) for i, offset in enumerate(offsets)]))
    return shifted_signals

def strided_indexing_roll(A, r):
    """
    Roll each row of a matrix A according to shifts r. 
    A: np.array
    r: np.array
    output: np.array
    """
    # Concatenate with sliced to cover all rolls
    p = np.full((A.shape[0],A.shape[1]-1),np.nan)
    A_ext = np.concatenate((p,A,p),axis=1)

    # Get sliding windows; use advanced-indexing to select appropriate ones
    n = A.shape[1]
    return view_as_windows(A_ext,(1,n))[np.arange(len(r)), -r + (n-1),0]
