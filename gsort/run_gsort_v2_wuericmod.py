import sys

import sklearn.cluster
import numpy as np
import sklearn.decomposition
import scipy.io as sio

from gsort.artifact_estimator_class import *
from gsort.old_labview_data_reader import *
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import scipy.stats
from functools import partial
from functools import partialmethod
from itertools import chain, combinations
# from small_sample_variance import *
from scipy.ndimage.interpolation import shift
import multiprocessing as mp
import gsort.eilib as eil
import warnings
from scipy.signal import wiener

from skimage.util.shape import view_as_windows as viewW



import gsort.elecresploader as el
import scipy.signal
import gsort.signal_alignment as sam


from itertools import product
from itertools import starmap

import fastconv.corr1d as corr1d # compiled dependency wueric, compiled already
from typing import List, Tuple

def run_pattern_movie_live(p,k,preloaded_data, q):
    
    cellids,running_cells_ind,relevant_cells,mutual_cells, total_cell_to_electrode_list, start_time_limit, end_time_limit,estim_analysis_path, noise, outpath, n_to_data_on_cells, NUM_CHANNELS, cluster_delay  = preloaded_data

    time_limit = end_time_limit - start_time_limit
    artifact_signals = [np.zeros((1,time_limit)) for i in range(NUM_CHANNELS)]
    total_electrode_list = []
    init_probs = np.zeros(len(cellids))
    final_probs = np.zeros(len(cellids))
    
    try:
        signal = get_oldlabview_pp_data(estim_analysis_path , p, k)
    except:
        print("Signal doesn't exists", p, k)
        q.put((-1, -1, -1, -1, -1, -1, ""))
        return 

    num_trials = len(signal)
  
    for cell in np.array(cellids)[running_cells_ind]:

        electrode_list =  list(set([e for c in mutual_cells[cell] for e in total_cell_to_electrode_list[c]]))
        cell_to_electrode_list = {k:v for k,v in total_cell_to_electrode_list.items() if k in mutual_cells[cell]}

        relevant_cells
        if cell in relevant_cells[p] and (len(electrode_list)>0):
            
            # print(f"Pre-artifact Processing cell {cell}", p,k)
            
            raw_signal = signal[:, electrode_list, start_time_limit:end_time_limit].astype(float) 
            mask =  get_mask(raw_signal, )
            cluster_cliques = cluster_each_cell(raw_signal,mask, cell_to_electrode_list, electrode_list, noise, "", cluster_delay = cluster_delay)
            event_labels = convert_cliques_to_labels(cluster_cliques, num_trials)
            
            significant_electrodes = np.arange(len(electrode_list))
            event_labels = first_merge_event_cliques_by_noise(electrode_list, raw_signal, event_labels,  mask, significant_electrodes, noise)

            data_on_cells = n_to_data_on_cells[cell]
            finished, G, (_, _), (event_labels_with_virtual, _), (final_clusters, _), edge_to_matched_signals, _, mask, note = gsort_spike_sorting(event_labels, significant_electrodes, electrode_list, raw_signal, mask, 1, 1000, noise, data_on_cells, artifact_cluster_estimate=None )
            total_p, cell_in_clusters = get_probabilities(G, event_labels_with_virtual)
            # print("A")
            p_error = compute_cosine_error(mask, G, edge_to_matched_signals, event_labels_with_virtual, cell)
            
            init_probs[cellids.index(cell)] = total_p[cell]-p_error
            sig = signal[event_labels_with_virtual==final_clusters[0]][:,:,:end_time_limit]
            total_electrode_list += list(set(total_electrode_list+electrode_list))
            # print(p, k, cell, total_p[cell])

            for e in electrode_list:
                artifact_signals[e] = np.concatenate((artifact_signals[e], sig[:,e,:]), axis = 0)
    # print(f"Finished pattern={p}, movie={k}")
    # print("q", q.qsize(), "p", p, "k", k)
    q.put((p, k, init_probs, -1, -1, num_trials, ""))
    
    return 

def run_pattern_movie_average(p,k,preloaded_data, q):
    
    cellids,running_cells_ind,relevant_cells,mutual_cells, total_cell_to_electrode_list, start_time_limit, end_time_limit,estim_analysis_path, noise, outpath, n_to_data_on_cells, NUM_CHANNELS, cluster_delay  = preloaded_data

    time_limit = end_time_limit - start_time_limit
    artifact_signals = [np.zeros((1,time_limit)) for i in range(NUM_CHANNELS)]
    total_electrode_list = []
    init_probs = np.zeros(len(cellids))
    final_probs = np.zeros(len(cellids))
    # print("p,k",p,k)
#     print(f"Started pattern={p}, movie={k}")
    try:
        signal = get_oldlabview_pp_data(estim_analysis_path , p, k)
    except:
        q.put((-1, -1, -1, -1, -1, -1, ""))
        return 

    
    for cell in np.array(cellids)[running_cells_ind]:
        # print("cell", cell)
        electrode_list =  list(set([e for c in mutual_cells[cell] for e in total_cell_to_electrode_list[c]]))
        cell_to_electrode_list = {k:v for k,v in total_cell_to_electrode_list.items() if k in mutual_cells[cell]}

        
        if cell in relevant_cells[p] and (len(electrode_list)>0):
            
            # print(f"Pre-artifact Processing cell {cell}", p,k)
            
            raw_signal = signal[:, electrode_list, start_time_limit:end_time_limit].astype(float) 
            raw_signal = np.r_[np.mean(raw_signal, axis = 0, keepdims = True), raw_signal]
            num_trials = len(raw_signal)
            mask =  get_mask(raw_signal, )
            cluster_cliques = cluster_each_cell(raw_signal,mask, cell_to_electrode_list, electrode_list, noise, "", cluster_delay = cluster_delay)
            event_labels = convert_cliques_to_labels(cluster_cliques, num_trials)
            significant_electrodes = np.arange(len(electrode_list))
            event_labels = first_merge_event_cliques_by_noise(electrode_list, raw_signal, event_labels,  mask, significant_electrodes, noise)

            data_on_cells = n_to_data_on_cells[cell]
            
            finished, G, (_, _), (event_labels_with_virtual, _), (final_clusters, _), edge_to_matched_signals, _, mask, note = gsort_spike_sorting(event_labels, significant_electrodes, electrode_list, raw_signal, mask, 1, 1000, noise, data_on_cells, artifact_cluster_estimate=event_labels[0] )
            
            total_p, cell_in_clusters = get_probabilities(G, event_labels_with_virtual[1:])
            bad_edges = []
            bad_edges = compute_cosine_error(mask, G, edge_to_matched_signals, event_labels_with_virtual, cell, bad_edges)
            peak_electrode = np.argmax(np.max(np.abs(data_on_cells[1][data_on_cells[0].index(cell)]), axis = 1))
            bad_edges = compute_latency_error(mask, G, edge_to_matched_signals, event_labels_with_virtual, cell, bad_edges, peak_electrode, offset = data_on_cells[-1][0,0], too_early = 5)
            bad_edges = compute_small_signal_error(mask, G, edge_to_matched_signals, event_labels_with_virtual, cell, bad_edges, peak_electrode, too_small = noise[electrode_list][peak_electrode])
            
            p_error = compute_error(G, event_labels_with_virtual, cell, bad_edges)
            init_probs[cellids.index(cell)] = total_p[cell]-p_error

            sig = signal[event_labels_with_virtual==final_clusters[0]][:,:,:end_time_limit]
            total_electrode_list += list(set(total_electrode_list+electrode_list))
    
    q.put((p, k, init_probs, -1, -1, num_trials, ""))
    return 

def run_pattern_movie(p,k,preloaded_data, q):
    """
    run_pattern_movie: Core script to run standard, single pattern-movie g-sort
    """
    cellids,running_cells_ind,relevant_cells,mutual_cells, total_cell_to_electrode_list, start_time_limit, end_time_limit,estim_analysis_path, noise, outpath, n_to_data_on_cells, NUM_CHANNELS, cluster_delay  = preloaded_data

    time_limit = end_time_limit - start_time_limit
    artifact_signals = [np.zeros((1,time_limit)) for i in range(NUM_CHANNELS)]
    total_electrode_list = []
    init_probs = np.zeros(len(cellids))
    final_probs = np.zeros(len(cellids))
    # print("p,k",p,k)
#     print(f"Started pattern={p}, movie={k}")
    try:
        signal = get_oldlabview_pp_data(estim_analysis_path , p, k)
    except:
#         print("Signal doesn't exists", p, k)
#         print("Quit")
        q.put((-1, -1, -1, -1, -1, -1, ""))
        return 
#     print(f"Started pattern={p}, movie={k}")
    num_trials = len(signal)
    for cell in np.array(cellids)[running_cells_ind]:
        # print("cell", cell)
        electrode_list =  list(set([e for c in mutual_cells[cell] for e in total_cell_to_electrode_list[c]]))
        cell_to_electrode_list = {k:v for k,v in total_cell_to_electrode_list.items() if k in mutual_cells[cell]}

        
        if cell in relevant_cells[p] and (len(electrode_list)>0):
            
            # print(f"Pre-artifact Processing cell {cell}", p,k)
            
            raw_signal = signal[:, electrode_list, start_time_limit:end_time_limit].astype(float) 
            mask =  get_mask(raw_signal, )
            cluster_cliques = cluster_each_cell(raw_signal,mask, cell_to_electrode_list, electrode_list, noise, "", cluster_delay = cluster_delay)
            event_labels = convert_cliques_to_labels(cluster_cliques, num_trials)
            significant_electrodes = np.arange(len(electrode_list))
            event_labels = first_merge_event_cliques_by_noise(electrode_list, raw_signal, event_labels,  mask, significant_electrodes, noise)

            data_on_cells = n_to_data_on_cells[cell]
            finished, G, (_, _), (event_labels_with_virtual, _), (final_clusters, _), edge_to_matched_signals, _, mask, note = gsort_spike_sorting(event_labels, significant_electrodes, electrode_list, raw_signal, mask, 1, 1000, noise, data_on_cells, artifact_cluster_estimate=None )
            total_p, cell_in_clusters = get_probabilities(G, event_labels_with_virtual)
            bad_edges = []
            bad_edges = compute_cosine_error(mask, G, edge_to_matched_signals, event_labels_with_virtual, cell, bad_edges)
            peak_electrode = np.argmax(np.max(np.abs(data_on_cells[1][data_on_cells[0].index(cell)]), axis = 1))
            bad_edges = compute_latency_error(mask, G, edge_to_matched_signals, event_labels_with_virtual, cell, bad_edges, peak_electrode, offset = data_on_cells[-1][0,0], too_early = 5)
            bad_edges = compute_small_signal_error(mask, G, edge_to_matched_signals, event_labels_with_virtual, cell, bad_edges, peak_electrode, too_small = noise[electrode_list][peak_electrode])
            
            p_error = compute_error(G, event_labels_with_virtual, cell, bad_edges)
            init_probs[cellids.index(cell)] = total_p[cell]-p_error

            sig = signal[event_labels_with_virtual==final_clusters[0]][:,:,:end_time_limit]
            total_electrode_list += list(set(total_electrode_list+electrode_list))
            for e in electrode_list:
                artifact_signals[e] = np.concatenate((artifact_signals[e], sig[:,e,:]), axis = 0)
            
            
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_artifact_signals = np.array([np.mean(s[1:], axis = 0) for s in artifact_signals])


    q.put((p, k, init_probs, mean_artifact_signals, final_probs, num_trials, ""))
    return 


def run_pattern_multi_movie(p,k,movie_offsets, preloaded_data, q):
    
    cellids,running_cells_ind,relevant_cells,mutual_cells, total_cell_to_electrode_list, start_time_limit, end_time_limit,estim_analysis_path, noise, outpath, n_to_data_on_cells, NUM_CHANNELS, cluster_delay  = preloaded_data

    time_limit = end_time_limit - start_time_limit
    artifact_signals = [np.zeros((1,time_limit)) for i in range(NUM_CHANNELS)]
    total_electrode_list = []
    init_probs = np.zeros(len(cellids))
    final_probs = np.zeros(len(cellids))
    # print("p,k",p,k)
#     print(f"Started pattern={p}, movie={k}")
    try:
        signal = get_oldlabview_pp_data(estim_analysis_path , p, k)
    except:
#         print("Signal doesn't exists", p, k)
#         print("Quit")
        q.put((-1, -1, -1, -1, -1, -1, ""))
        return 
#     print(f"Started pattern={p}, movie={k}")
    num_trials = len(signal)
    for cell in np.array(cellids)[running_cells_ind]:
        # print("cell", cell)
        electrode_list =  list(set([e for c in mutual_cells[cell] for e in total_cell_to_electrode_list[c]]))
        cell_to_electrode_list = {k:v for k,v in total_cell_to_electrode_list.items() if k in mutual_cells[cell]}

        
        if cell in relevant_cells[p] and (len(electrode_list)>0):
            
            # print(f"Pre-artifact Processing cell {cell}", p,k)
            
            raw_signal = signal[:, electrode_list, start_time_limit:end_time_limit].astype(float) 
            mask =  get_mask(raw_signal, )
            cluster_cliques = cluster_each_cell(raw_signal,mask, cell_to_electrode_list, electrode_list, noise, "", cluster_delay = cluster_delay)
            event_labels = convert_cliques_to_labels(cluster_cliques, num_trials)
            significant_electrodes = np.arange(len(electrode_list))
            event_labels = first_merge_event_cliques_by_noise(electrode_list, raw_signal, event_labels,  mask, significant_electrodes, noise)

            data_on_cells = n_to_data_on_cells[cell]
            finished, G, (_, _), (event_labels_with_virtual, _), (final_clusters, _), edge_to_matched_signals, _, mask, note = gsort_spike_sorting(event_labels, significant_electrodes, electrode_list, raw_signal, mask, 1, 1000, noise, data_on_cells, artifact_cluster_estimate=None )
            total_p, cell_in_clusters = get_probabilities(G, event_labels_with_virtual)
            bad_edges = []
            bad_edges = compute_cosine_error(mask, G, edge_to_matched_signals, event_labels_with_virtual, cell, bad_edges)
            peak_electrode = np.argmax(np.max(np.abs(data_on_cells[1][data_on_cells[0].index(cell)]), axis = 1))
            bad_edges = compute_latency_error(mask, G, edge_to_matched_signals, event_labels_with_virtual, cell, bad_edges, peak_electrode, offset = data_on_cells[-1][0,0], too_early = 5)
            bad_edges = compute_small_signal_error(mask, G, edge_to_matched_signals, event_labels_with_virtual, cell, bad_edges, peak_electrode, too_small = noise[electrode_list][peak_electrode])
            
            p_error = compute_error(G, event_labels_with_virtual, cell, bad_edges)
            init_probs[cellids.index(cell)] = total_p[cell]-p_error

            sig = signal[event_labels_with_virtual==final_clusters[0]][:,:,:end_time_limit]
            total_electrode_list += list(set(total_electrode_list+electrode_list))
            for e in electrode_list:
                artifact_signals[e] = np.concatenate((artifact_signals[e], sig[:,e,:]), axis = 0)
            
            
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_artifact_signals = np.array([np.mean(s[1:], axis = 0) for s in artifact_signals])
    
    try:
        prev_signal = get_oldlabview_pp_data(estim_analysis_path , p, k-1)
        prev_signal_exists = True
    except:
        prev_signal_exists = False
    
    if prev_signal_exists:
        num_trials_total = len(signal) + len(prev_signal)
        for cell in np.array(cellids)[running_cells_ind]:
                # print("cell", cell)
                electrode_list =  list(set([e for c in mutual_cells[cell] for e in total_cell_to_electrode_list[c]]))
                cell_to_electrode_list = {k:v for k,v in total_cell_to_electrode_list.items() if k in mutual_cells[cell]}
                if cell in relevant_cells[p] and (len(electrode_list)>0):
                    raw_signal = np.vstack((prev_signal, signal))[:, electrode_list, start_time_limit:end_time_limit].astype(float) 
                    mask =  get_mask(raw_signal, )
                    cluster_cliques = cluster_each_cell(raw_signal,mask, cell_to_electrode_list, electrode_list, noise, "", cluster_delay = cluster_delay)
                    event_labels = convert_cliques_to_labels(cluster_cliques, num_trials_total)
                    significant_electrodes = np.arange(len(electrode_list))
                    event_labels = first_merge_event_cliques_by_noise(electrode_list, raw_signal, event_labels,  mask, significant_electrodes, noise)

                    data_on_cells = n_to_data_on_cells[cell]
                    finished, G, (_, _), (event_labels_with_virtual, _), (final_clusters, _), edge_to_matched_signals, _, mask, note = gsort_spike_sorting(event_labels, significant_electrodes, electrode_list, raw_signal, mask, 1, 1000, noise, data_on_cells, artifact_cluster_estimate=None )
                    total_p, cell_in_clusters = get_probabilities(G, event_labels_with_virtual[len(prev_signal):])
                    bad_edges = []
                    bad_edges = compute_cosine_error(mask, G, edge_to_matched_signals, event_labels_with_virtual[len(prev_signal):], cell, bad_edges)
                    peak_electrode = np.argmax(np.max(np.abs(data_on_cells[1][data_on_cells[0].index(cell)]), axis = 1))
                    bad_edges = compute_latency_error(mask, G, edge_to_matched_signals, event_labels_with_virtual[len(prev_signal):], cell, bad_edges, peak_electrode, offset = data_on_cells[-1][0,0], too_early = 5)
                    bad_edges = compute_small_signal_error(mask, G, edge_to_matched_signals, event_labels_with_virtual[len(prev_signal):], cell, bad_edges, peak_electrode, too_small = noise[electrode_list][peak_electrode])
                    p_error = compute_error(G, event_labels_with_virtual[len(prev_signal):], cell, bad_edges)
                    final_probs[cellids.index(cell)] = total_p[cell]-p_error


    q.put((p, k, init_probs, mean_artifact_signals, final_probs, num_trials, ""))
    return 

def compute_cosine_error(mask, G, graph_signal, clustering, n, bad_edges_, cosine_similarity = 0.7):
    
    edge_to_cell = {k:v for k, v in graph_signal.keys()}
    signal_index = np.argwhere([k[1] ==n for k in  graph_signal.keys()]).flatten()
    g_signal_error = 0
    if len(signal_index) != 0:
        edges = np.array([list(graph_signal.keys())[k][0] for k in signal_index])  
        low_power = [np.sum(mask*graph_signal[list(graph_signal.keys())[k]][0]*graph_signal[list(graph_signal.keys())[k]][1])/np.sqrt(np.sum(graph_signal[list(graph_signal.keys())[k]][1]**2))/np.sqrt(np.sum(mask*graph_signal[list(graph_signal.keys())[k]][0]**2))<=cosine_similarity for k in signal_index]
        bad_edges = list(edges[low_power])
        bad_edges_ += [e for e in bad_edges if tuple(e) not in [tuple(b) for b in bad_edges_]]
    return bad_edges_

     
def compute_latency_error(mask, G, graph_signal, clustering, n, bad_edges_, electrode, offset = 50, too_early = 5, too_late = 25):
    
    signal_index = np.argwhere([k[1] ==n for k in  graph_signal.keys()]).flatten()
    
    if len(signal_index) != 0:
        edges = np.array([list(graph_signal.keys())[k][0] for k in signal_index])  
        low_power = [(offset-graph_signal[list(graph_signal.keys())[k]][-1][electrode] < too_early) or (offset-graph_signal[list(graph_signal.keys())[k]][-1][electrode] > too_late ) for k in signal_index]
    

        bad_edges = list(edges[low_power])
        bad_edges_ += [e for e in bad_edges if tuple(e) not in [tuple(b) for b in bad_edges_]]
    return bad_edges_

def compute_small_signal_error(mask, G, graph_signal, clustering, n, bad_edges_, electrode,too_small = 1):
    
    signal_index = np.argwhere([k[1] ==n for k in  graph_signal.keys()]).flatten()
    
    if len(signal_index) != 0:
        edges = np.array([list(graph_signal.keys())[k][0] for k in signal_index])  
        low_power = [(np.max(np.abs(graph_signal[list(graph_signal.keys())[k]][1])[electrode])<=too_small) for k in signal_index]
        bad_edges = list(edges[low_power])
        bad_edges_ += [e for e in bad_edges if tuple(e) not in [tuple(b) for b in bad_edges_]]
    return bad_edges_


def compute_error(G, clustering, n, bad_edges):
    error = 0 
    error += sum([sum(clustering == e[1])/len(clustering) for e in bad_edges])
    error += sum([sum(clustering == n)/len(clustering) for e in bad_edges for n in nx.descendants(G, e[1])])
    return error

# def compute_cosine_error(mask, G, graph_signal, clustering, n, cosine_similarity = 0.7):
    
#     edge_to_cell = {k:v for k, v in graph_signal.keys()}
#     signal_index = np.argwhere([k[1] ==n for k in  graph_signal.keys()]).flatten()
#     g_signal_error = 0

#     if len(signal_index) != 0:

#         edges = np.array([list(graph_signal.keys())[k][0] for k in signal_index])  
#         low_power = [np.sum(mask*graph_signal[list(graph_signal.keys())[k]][0]*graph_signal[list(graph_signal.keys())[k]][1])/np.sqrt(np.sum(graph_signal[list(graph_signal.keys())[k]][1]**2))/np.sqrt(np.sum(mask*graph_signal[list(graph_signal.keys())[k]][0]**2))<=cosine_similarity for k in signal_index]
#         low_power_value = [np.sum(mask*graph_signal[list(graph_signal.keys())[k]][0]*graph_signal[list(graph_signal.keys())[k]][1])/np.sqrt(np.sum(graph_signal[list(graph_signal.keys())[k]][1]**2))/np.sqrt(np.sum(mask*graph_signal[list(graph_signal.keys())[k]][0]**2)) for k in signal_index]
#         # print(low_power_value)
#         g_signal_error += sum([sum(clustering == e[1])/len(clustering) for e in edges[low_power]])
#         g_signal_error += sum([sum(clustering == n)/len(clustering) for e in edges[low_power] for n in nx.descendants(G, e[1])])
#     # print("Compute cosine error")
#     return g_signal_error

def run_movie_cell(p, ks, preloaded_data):
    
    n, mutual_cells, cell_to_electrode_list, electrode_list, data_on_cells, start_time_limit, end_time_limit, estim_analysis_path, noise, outpath, good_patterns = preloaded_data

    shift_window=[start_time_limit, end_time_limit]
    
    
    probs = []
    cosine_probs = []
    notes = []
    num_trials_list = []
    clustering_list = []
    cluster_label_list = []
    root_signals = []
    root_list = []
    all_signals = []
    p_to_cell= {}

    if p not in good_patterns:
        return (p, [i for i in range(len(probs))], cosine_probs, probs, n)
    # print(f"cell {n}, pattern {p} started")
    
    cell_ids, cell_eis, cell_error, cell_spk_times = data_on_cells
    significant_electrodes = np.arange(len(electrode_list))
    
    run_info = {}
    for k in range(ks):
        # print('k',k)
        try:
            signal = get_oldlabview_pp_data(estim_analysis_path , p, k)
        except:
            break

        num_trials = len(signal)
        raw_signal = signal[:, electrode_list, start_time_limit:end_time_limit].astype(float) 
        mask =  get_mask(raw_signal)
        cluster_cliques = cluster_each_cell(raw_signal,mask, cell_to_electrode_list, electrode_list, noise, "")
        event_labels = convert_cliques_to_labels(cluster_cliques, num_trials)


        finished, G, (_, _), (event_labels_with_virtual, _), (_, _), edge_to_matched_signals, _, mask, note = gsort_spike_sorting(event_labels, significant_electrodes, electrode_list, raw_signal, mask, 1, 1000, noise, data_on_cells, artifact_cluster_estimate=None)



        total_p, cell_in_clusters = get_probabilities(G, event_labels_with_virtual)


        cosine_similarity = 0.7
        graph_signal = edge_to_matched_signals
        edge_to_cell = {k:v for k, v in graph_signal.keys()}
        signal_index = np.argwhere([k[1] ==n for k in  graph_signal.keys()]).flatten()
        g_signal_error = 0
        clustering = event_labels_with_virtual
        run_info[k] = {}
        if len(signal_index) != 0:

            edges = np.array([list(graph_signal.keys())[k][0] for k in signal_index])  
            low_power = [np.sum(mask*graph_signal[list(graph_signal.keys())[k]][0]*graph_signal[list(graph_signal.keys())[k]][1])/np.sqrt(np.sum(graph_signal[list(graph_signal.keys())[k]][1]**2))/np.sqrt(np.sum(mask*graph_signal[list(graph_signal.keys())[k]][0]**2))<=cosine_similarity for k in signal_index]
            low_power_value = np.array([np.sum(mask*graph_signal[list(graph_signal.keys())[k]][0]*graph_signal[list(graph_signal.keys())[k]][1])/np.sqrt(np.sum(graph_signal[list(graph_signal.keys())[k]][1]**2))/np.sqrt(np.sum(mask*graph_signal[list(graph_signal.keys())[k]][0]**2)) for k in signal_index])
            correlation = np.array([np.sum( mask*(np.abs(graph_signal[list(graph_signal.keys())[k]][1])>1) )/np.sum(np.abs(cell_eis[cell_ids.index(n)])> 1) for k in signal_index])
            
            g_signal_error += sum([sum(clustering == e[1])/len(clustering) for e in edges[low_power]])
            g_signal_error += sum([sum(clustering == n)/len(clustering) for e in edges[low_power] for n in nx.descendants(G, e[1])])

            edge_dep_probs = np.array([sum(clustering == e[1])/len(clustering) for e in edges])
            edge_dep_probs += np.array([ sum([sum(clustering == n)/len(clustering) for n in nx.descendants(G, e[1])]) for e in edges ])
            cosine_probs += [total_p[n]-g_signal_error]

            run_info[k]['cosine_prob'] = (total_p[n]-g_signal_error, cosine_similarity)
            run_info[k]['edge_probs'] = edge_dep_probs
            run_info[k]['non_saturated_template'] = correlation
            run_info[k]['data_similarity'] = low_power_value

        else:
            cosine_probs += [total_p[n]]
            run_info[k]['cosine_prob'] = (total_p[n], cosine_similarity)
    
        run_info[k]['electrode_list'] = electrode_list    
        run_info[k]['prob'] = total_p[n]    
   
        probs += [total_p[n]]
        # print('cosine_probs',cosine_probs[-1])

        run_info[k]['clustering'] = event_labels_with_virtual
        run_info[k]['graph_info'] = (list(G.nodes), list(G.edges), nx.get_edge_attributes(G, 'cell'))
        run_info[k]['num_trials'] = len(signal)
        run_info[k]['mutual_cells'] = mutual_cells
        run_info[k]['note'] = note
    with open(os.path.join(outpath, 'gsort_info_n' + str(n)+ '_p' + str(p) +'.pkl'), 'wb') as f:
            pickle.dump(run_info, f)

    return (p, [i for i in range(len(probs))], cosine_probs, probs, n)

def run_movie(n, p, ks, preloaded_data):
    
    mutual_cells, cell_to_electrode_list, electrode_list, data_on_cells, start_time_limit, end_time_limit, estim_analysis_path, noise, outpath = preloaded_data

    shift_window=[start_time_limit, end_time_limit]
    
    
    probs = []
    cosine_probs = []
    notes = []
    num_trials_list = []
    clustering_list = []
    cluster_label_list = []
    root_signals = []
    root_list = []
    all_signals = []
    p_to_cell= {}

    
    cell_ids, cell_eis, cell_error, cell_spk_times = data_on_cells
    significant_electrodes = np.arange(len(electrode_list))
    
    run_info = {}
    for k in range(ks):
        print('k',k)
        try:
            signal = get_oldlabview_pp_data(estim_analysis_path , p, k)
        except:
            break

        num_trials = len(signal)
        raw_signal = signal[:, electrode_list, start_time_limit:end_time_limit].astype(float) 
        mask =  get_mask(raw_signal)
        cluster_cliques = cluster_each_cell(raw_signal,mask, cell_to_electrode_list, electrode_list, noise, "")
        event_labels = convert_cliques_to_labels(cluster_cliques, num_trials)


        finished, G, (_, _), (event_labels_with_virtual, _), (_, _), edge_to_matched_signals, _, mask, note = gsort_spike_sorting(event_labels, significant_electrodes, electrode_list, raw_signal, mask, 1, 1000, noise, data_on_cells, artifact_cluster_estimate=None)



        total_p, cell_in_clusters = get_probabilities(G, event_labels_with_virtual)


        cosine_similarity = 0.7
        graph_signal = edge_to_matched_signals
        edge_to_cell = {k:v for k, v in graph_signal.keys()}
        signal_index = np.argwhere([k[1] ==n for k in  graph_signal.keys()]).flatten()
        g_signal_error = 0
        clustering = event_labels_with_virtual
        run_info[k] = {}
        if len(signal_index) != 0:

            edges = np.array([list(graph_signal.keys())[k][0] for k in signal_index])  
            low_power = [np.sum(mask*graph_signal[list(graph_signal.keys())[k]][0]*graph_signal[list(graph_signal.keys())[k]][1])/np.sqrt(np.sum(graph_signal[list(graph_signal.keys())[k]][1]**2))/np.sqrt(np.sum(mask*graph_signal[list(graph_signal.keys())[k]][0]**2))<=cosine_similarity for k in signal_index]
            low_power_value = np.array([np.sum(mask*graph_signal[list(graph_signal.keys())[k]][0]*graph_signal[list(graph_signal.keys())[k]][1])/np.sqrt(np.sum(graph_signal[list(graph_signal.keys())[k]][1]**2))/np.sqrt(np.sum(mask*graph_signal[list(graph_signal.keys())[k]][0]**2)) for k in signal_index])
            correlation = np.array([np.sum( mask*(np.abs(graph_signal[list(graph_signal.keys())[k]][1])>1) )/np.sum(np.abs(cell_eis[cell_ids.index(n)])> 1) for k in signal_index])
            
            g_signal_error += sum([sum(clustering == e[1])/len(clustering) for e in edges[low_power]])
            g_signal_error += sum([sum(clustering == n)/len(clustering) for e in edges[low_power] for n in nx.descendants(G, e[1])])

            edge_dep_probs = np.array([sum(clustering == e[1])/len(clustering) for e in edges])
            edge_dep_probs += np.array([ sum([sum(clustering == n)/len(clustering) for n in nx.descendants(G, e[1])]) for e in edges ])
            cosine_probs += [total_p[n]-g_signal_error]

            run_info[k]['cosine_prob'] = (total_p[n]-g_signal_error, cosine_similarity)
            run_info[k]['edge_probs'] = edge_dep_probs
            run_info[k]['non_saturated_template'] = correlation
            run_info[k]['data_similarity'] = low_power_value

        else:
            cosine_probs += [total_p[n]]
            run_info[k]['cosine_prob'] = (total_p[n], cosine_similarity)
    
        run_info[k]['electrode_list'] = electrode_list    
        run_info[k]['prob'] = total_p[n]    
   
        probs += [total_p[n]]
#         print('cosine_probs',cosine_probs[-1])

        run_info[k]['clustering'] = event_labels_with_virtual
        run_info[k]['graph_info'] = (list(G.nodes), list(G.edges), nx.get_edge_attributes(G, 'cell'))
        run_info[k]['num_trials'] = len(signal)
        run_info[k]['mutual_cells'] = mutual_cells
        run_info[k]['note'] = note
    with open(os.path.join(outpath, 'gsort_info_n' + str(n)+ '_p' + str(p) +'.pkl'), 'wb') as f:
            pickle.dump(run_info, f)

    return ( p, [i for i in range(len(probs))], cosine_probs, probs)

def gsort_spike_sorting(event_labels, significant_electrodes, electrode_list, signals_tmp, mask_tmp, iterations,max_iter, noise, data_on_cells, damping = 0.5, sat_band = 5, unsatured_min= -1000, unsaturated_max = 400, cluster_delay = 7, artifact_cluster_estimate = None, hierachical_cluster = False, no_time = False, raw = False):
    """
    Main run script
    """
    
    total_p = Counter()
    total_p_gof = Counter()
    total_p_greedy = Counter()
    valid_graph = 1
    edge_to_cell_and_fit_info = {}
    

    signals = copy.copy(signals_tmp)
    mask = copy.copy(mask_tmp)
    num_trials, num_electrodes, num_samples = signals.shape
    cells_on_trials = {i:[] for i in range(num_trials)}
    note = ""
    
    initial_event_set = set(event_labels)
    
    # Create nodes for explanation graph
    G = nx.DiGraph()
    G.add_nodes_from(list(set(event_labels)))
    state_to_cells ={e:[] for e in event_labels}

    
    initial_event_labels = copy.copy(event_labels)
    constrained = 'gof'
   
        
    # If all cliques have been merged, quit
    if len(set(event_labels))==1 or iterations == 0:
        note += " early completion, 1 cluster"
        return 1, G, (event_labels, signals), (event_labels, signals), (event_labels, signals), {}, 0, mask, note



    # Compute difference signals
    difference_signals, event_signal_ordering =compute_difference_signals(signals, event_labels)
    
   
    # Clean the difference signals
    difference_signals = clean_signals(difference_signals, noise=noise, electrode_list=electrode_list, thr = 1, mode="svd")

    # Find candidate edges in graph from comparing templates and difference signals
    if not raw:
        edge_cell_to_fit_info, edge_cell_to_signal, edge_cell_to_error, edge_cell_to_loss, edge_cell_to_noise_loss, edge_cell_to_latency = find_candidate_edges(electrode_list, data_on_cells, difference_signals, event_signal_ordering,  mask, event_labels, noise, constrained = constrained, artifact_estimate = artifact_cluster_estimate)
    else:
        edge_cell_to_fit_info, edge_cell_to_signal, edge_cell_to_error, edge_cell_to_loss, edge_cell_to_noise_loss, edge_cell_to_latency = find_candidate_edges_raw(electrode_list, data_on_cells, difference_signals, event_signal_ordering,  mask, event_labels, noise, constrained = constrained, artifact_estimate = artifact_cluster_estimate)
    sorted_edges = sorted(edge_cell_to_loss.items(), key=lambda item: item[1])
    _, _, num_samples = signals.shape   
    
    finished = False
    check_points = {}
    check_points_loss = {}
    v_structure_edges_and_nodes = []
    nodes_to_virtual_nodes = {}
    initial_nodes = len(G.nodes)
    
    skip_edges = []
    tracker = 0
    downstream = {}
    
    # Add edges that satisfy contraints: no improper chains, cycles, and (in simple mode) v-structures
    while len(sorted_edges) != 0:

        
        e_c, loss = sorted_edges.pop(0)
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
    event_labels_with_virtual = copy.copy(event_labels)
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
    
#         print(edge_cell_to_latency[((head, tail), cell)])
    return finished, G, (initial_event_labels, signals_tmp), (event_labels_with_virtual, signals), (event_labels, final_signals), edge_to_matched_signals, edge_cell_to_loss, mask, note

def first_merge_event_cliques_by_noise(electrode_list, signals, event_labels_tmp,  mask, significant_electrodes, noise):
    """
    Iteratively merge event cliques according to average l2 distance.
    This implementation is a bit silly; will be improved later. 
    
    electrode_list: list<int>
    signals: np.array
    event_labels_tmp: list<int>
    state_to_cell: dict<key: int, value: list<tuple<int, int> > >
    mask: np.array
    significant_electrodes: np.array
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

def l2_max(event_pop_difference_signal, eis, ei_errors_, mask_, n):
    """
    Compute the optimal noise normalized MSE and negative log likelihood along with the corresponding alignment between the difference signal and cell templates.
    event_pop_difference_signal: np.array
    eis: np.array
    ei_errors_: np.array
    mask: np.array
    n: int

    output: np.array, np.array, np.array
    """
    difference_signals = event_pop_difference_signal[ 2:]
    event_pop = event_pop_difference_signal[:2]


    # Reverse the signal because convolution reverses it back
    difference_signal = np.expand_dims(difference_signals[::-1], 0)
    mask = np.expand_dims(mask_[::-1], 0)
    
    # Seperate cell variance from baseline noise variance
    # Scale noise variance to account for difference gaussian random variables
    # Scale cell variance to account for number of signals in the event clique
    ei_errors  = sum(event_pop) * ei_errors_

    # Decompose sliding || a - b ||**2 into multiple convolutions
    aTa = scipy.signal.convolve(1/ei_errors, (mask*difference_signal)**2, mode = 'valid')
    bTb = scipy.signal.convolve(eis**2/ei_errors, mask, mode = 'valid')
    aTb = scipy.signal.convolve(eis/ei_errors, mask*difference_signal, mode = 'valid')
    normalized_mse = aTa + bTb  - 2 * aTb 
    
    aTa = scipy.signal.convolve(1/ei_errors, (mask*difference_signal)**2, mode = 'valid')

    var_bump = np.sum(np.log(ei_errors), axis = 1, keepdims=True)
    

    
    negative_log_likelihood =  aTa + bTb  - 2 * aTb  + var_bump
    return np.min(normalized_mse, axis = 1), np.argmin(normalized_mse, axis = 1), np.min(normalized_mse + var_bump, axis = 1), np.min(aTa, axis = 1)

def direct_similarity_raw_wueric(
                            electrode_list: List[int],
                             data_on_cells: Tuple[List[int], np.ndarray, np.ndarray, np.ndarray],
                             difference_signals: np.ndarray,
                             mask: np.ndarray,
                             event_pop: np.ndarray,
                             noise: np.ndarray):
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

def direct_similarity_wueric(electrode_list: List[int],
                             data_on_cells: Tuple[List[int], np.ndarray, np.ndarray, np.ndarray],
                             difference_signals: np.ndarray,
                             mask: np.ndarray,
                             event_pop: np.ndarray,
                             noise: np.ndarray):
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
    normalized_mse_total = np.sum(min_mse_no_event_pop, axis=2).transpose(1, 0) / event_pop_axis_sum[:, None]

    # shape (2, n_cells) + (2, n_cells) -> (2, n_cells)
    neg_log_likelihood_total = normalized_mse_total + var_bump

    # shape (n_cells, 2, n_electrodes, n_timepoints_ei - n_timepoints_diff_sig + 1)
    # -> (n_cells, 2, n_electrodes) -> (n_cells, 2) -> (2, n_cells)
    noise_neg_log_likelihood_total = np.sum(np.min(aTa_parallel_corr, axis=-1), axis=-1).transpose(1, 0) / event_pop_axis_sum[:, None]

    idxs_total = {el_id : min_mse_index[:,:, idx].T for idx, el_id in enumerate(electrode_list)}

    return normalized_mse_total, idxs_total, neg_log_likelihood_total, noise_neg_log_likelihood_total

def get_center_eis(n, electrode_list, ap, array_id = 1501,  num_samples = 25, snr_ratio = 1, power_threshold = 2, excluded_types = ['bad','dup'], excluded_cells = [], sample_len_left = 55,sample_len_right = 75 , with_noise = True):
    """
    Return templates and template parameters
    n: int
    electrode_list: list<int>
    ap: tuple<string, string> 
    array_id: int
    num_samples: int
    snr_ration: float
    
    output: list<int>, np.array, np.array, np.array
    """

    # Store relevant templates because on SNR on the stimulating electrode and SNR on the electrodes relevant to the cell-of-interest
    tl = TemplateLoader(ap[0], '', ap[1], array_id = array_id)
    
    if with_noise:
        tl.store_all_cells_except_with_noise(excluded_types)
    else:
        tl.store_all_cells_except(excluded_types)
    
    # Remove this condition in next set of run
    tl.remove_templates_by_list(excluded_cells)
    tl.remove_templates_with_zero_variance(electrode_list)
    tl.remove_templates_by_elec_power(electrode_list, power_threshold, num_samples)
    
    if n not in tl.cellids:
        tl.store_cells_from_list([n])

    # Align the peak of each template along each electrode
    cell_eis = np.pad(np.array([tl.templates[i][electrode_list, :] for i in range(len(tl.cellids))]), ((0,0),(0,0),(sample_len_left, sample_len_right)), mode='constant')
    cell_eis_variance = np.pad(np.array([tl.templates_variance[i][electrode_list, :]**2 for i in range(len(tl.cellids))]), ((0,0),(0,0),(sample_len_left, sample_len_right)), mode='edge')

    peak_spike_times = np.argmin(cell_eis, axis = 2)
    peak_spike_times[peak_spike_times < sample_len_left] = sample_len_left
    
    cell_eis_tmp = np.zeros((cell_eis.shape[0], cell_eis.shape[1],sample_len_left + sample_len_right))
    cell_variance_tmp = np.zeros((cell_eis.shape[0], cell_eis.shape[1],sample_len_left + sample_len_right))
    
    for i in range(len(peak_spike_times)):
        for j in range(len(electrode_list)):
        
            cell_eis_tmp[i, j] = cell_eis[i,j, peak_spike_times[i][j]-sample_len_left:peak_spike_times[i][j]+sample_len_right]
            cell_variance_tmp[i, j] = cell_eis_variance[i,j, peak_spike_times[i][j]-sample_len_left:peak_spike_times[i][j]+sample_len_right]
    # for j in range(len(electrode_list)):
    #     cell_variance_tmp[i+1, j] = cell_eis_variance[i,j, peak_spike_times[i][j]-sample_len_left:peak_spike_times[i][j]+sample_len_right]
        
    peak_spike_times = np.argmin(cell_eis_tmp, axis = 2)
    cellids = tl.cellids
    # cellids.append(-1)
    return cellids,  cell_eis_tmp, cell_variance_tmp, peak_spike_times

def clean_signals(signals, noise=None, electrode_list=None, thr = 1, mode="svd"):
    """
    Denoise signals with SVD decomposition + thresholding
    signals: np.array
    thr: int
    
    output: np.array
    """
   
    num_putative_signals = signals.shape[0]
    
    
    # SVD of signals flattend across electrodes x time
    if mode == "svd":
        putative_signals_flat = signals.reshape((num_putative_signals, -1))
        try:
            U, S, Vh = np.linalg.svd(putative_signals_flat)

            # Singular values are presorted; store indices with singular value above threshold
            sc = sum(S > thr) 

            # Reconstruct signal
            clean_putative_signals = U[:, :sc] @ np.diag(S[:sc]) @ Vh[:sc, :]
            return clean_putative_signals.reshape(signals.shape)
        except:
            return signals
    else:
        for j in range(signals.shape[1]):
            signals[:,j,:] = wiener(signals[:,j,:], mysize=(1, 5), noise=2*noise[electrode_list][j]**2)
        return signals
            
                  
def compute_difference_signals(signals, event_labels):
    """
    Compute all potential difference signals between event cliques and store the subtraction ordering
    signals: np.array
    event_labels: list<int>

    output: np.array, np.array
    """
    num_trials, num_electrodes, num_samples = signals.shape
    
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
        list_of_tuples = [align_group_main(Z[i], sample_len = num_samples) for i in range(num_electrodes)]
        diff_signal = np.array([t__[0] for t__ in list_of_tuples])

       
        pairwise_signals += [diff_signal]
        pairwise_signals += [-diff_signal]
        signal_ordering += [(event_set[pair_of_event_cliques[0]], event_set[pair_of_event_cliques[1]])]
        signal_ordering += [(event_set[pair_of_event_cliques[1]], event_set[pair_of_event_cliques[0]])]
        
        
    pairwise_signals = np.array(pairwise_signals)
    signal_ordering = np.array(signal_ordering)
    
    return pairwise_signals, signal_ordering

def compute_difference_signals_2(signals, event_labels):
    """
    Compute all potential difference signals between event cliques and store the subtraction ordering
    signals: np.array
    event_labels: list<int>

    output: np.array, np.array
    """
    num_trials, num_electrodes, num_samples = signals.shape
    
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
        c_i, amin_Xx = get_alignment(Z[0], window = 10, res = 2, sample_len = num_samples)

        diff_signal = []
        for i in range(num_electrodes):
            high_res_X = interpolate(Z[i],res = 2, sample_len = num_samples)
            high_res_X = strided_indexing_roll(high_res_X, amin_Xx)
            np.nan_to_num(high_res_X, copy = False)
            diff_signal += [np.mean(high_res_X[:,::2], axis = 0)]
            
        # return (np.mean(high_res_X[:,::res], axis = 0), (min_index, max_index))

        diff_signal = np.array(diff_signal)

       
        pairwise_signals += [diff_signal]
        pairwise_signals += [-diff_signal]
        signal_ordering += [(event_set[pair_of_event_cliques[0]], event_set[pair_of_event_cliques[1]])]
        signal_ordering += [(event_set[pair_of_event_cliques[1]], event_set[pair_of_event_cliques[0]])]
        
        
    pairwise_signals = np.array(pairwise_signals)
    signal_ordering = np.array(signal_ordering)
    
    return pairwise_signals, signal_ordering




def align_group_main_2(x, sample_len = 50):
    """
    Reshape signal and call align_group
    x: np.array
    """
    X = x.reshape((-1,sample_len))

    return align_group_2(X)

def interpolate(X_,res = 2, sample_len = 55):
    # Upsample data for better signal alignment
    X = X_.reshape((-1,sample_len))
    x = np.arange(0, len(X[0]))
    interpolant = scipy.interpolate.interp1d(x,X, kind='cubic')
    domain = np.linspace(0, len(x)-1, len(x)*res)
    high_res_X = interpolant(domain)
    return high_res_X


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
    max_index = np.argmax(X,axis = 1)
    
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
        

def align_group_main(x, sample_len = 50):
    """
    Reshape signal and call align_group
    x: np.array
    """
    X = x.reshape((-1,sample_len))

    return align_group(X)
 
def align_group(X, window = 10, res = 2):
    """
    Align a set of set of signals to approximatetly minimize global l2 distance
    X: np.array
    window: int, default = 10

    output: tuple<np.array, tuple<int, int> >

    """
    
    # Special case: Only one signal in signal set
    if len(X) == 1:
        min_index = np.argmin(X,axis = 1)
        max_index = np.argmax(X,axis = 1)
        return (X[0], (min_index, max_index))
    else:
        
        # Upsample data for better signal alignment
        x = np.arange(0, len(X[0]))
        interpolant = scipy.interpolate.interp1d(x,X, kind='cubic')
        domain = np.linspace(0, len(x)-1, len(x)*res)
        high_res_X = interpolant(domain)
        
        min_index = np.argmin(X,axis = 1)
        max_index = np.argmax(X,axis = 1)
        
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
        
        return (np.mean(high_res_X[:,::res], axis = 0), (min_index, max_index))
  
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

def shift_sig(sig, shifts):
    """
    Shift difference signal sig according to shifts
    
    sig: np.array
    shifts: np.array
    
    output: np.array
    """
    peak_times = np.argmin(sig, axis=1)
    offsets = shifts - peak_times[:, None]
    num_electrodes, num_relevant_trials = shifts.shape
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
    return viewW(A_ext,(1,n))[np.arange(len(r)), -r + (n-1),0]



def find_candidate_edges_raw(electrode_list, data_on_cells, difference_signals, difference_signal_ordering, mask, event_labels, noise,
                               constrained = 'gof',  p = 0.99, max_electrodes_oor = 2, valid_shift_range = 35, artifact_estimate = None, plot = True, no_time = False):
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

def find_candidate_edges(electrode_list, data_on_cells, difference_signals, difference_signal_ordering, mask, event_labels, noise,
                               constrained = 'gof',  p = 0.99, max_electrodes_oor = 2, valid_shift_range = 35, artifact_estimate = None, plot = True, no_time = False):
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
        
def get_significant_electrodes(ei, compartments, noise, cell_spike_window = 25, max_electrodes_considered = 30, rat = 2):
    cell_power = ei**2
    e_sorted = np.argsort(np.sum(ei**2, axis = 1))[::-1]
    e_sorted = [e for e in e_sorted if eil.axonorsomaRatio(ei[e,:]) in compartments]
    cell_power = ei**2
    power_ordering = np.argsort(cell_power, axis = 1)[:,::-1]
    significant_electrodes = np.argwhere(np.sum(np.take_along_axis(cell_power[e_sorted], power_ordering[e_sorted,:cell_spike_window], axis = 1), axis = 1) >= rat * cell_spike_window * np.array(noise[e_sorted])**2).flatten()

    electrode_list = list(np.array(e_sorted)[significant_electrodes][:max_electrodes_considered])
    return electrode_list
            
def get_probabilities(G, event_labels, num_trials = None):
    G_mod = G.copy()
    for n in [n for n in G_mod.nodes if G_mod.in_degree(n) > 1]:
        edge_list = list(G_mod.in_edges(n))[1:]
        G_mod.remove_edges_from(edge_list)

    cells_on_graph = nx.get_edge_attributes(G_mod, 'cell').values()
    edge_to_cells = nx.get_edge_attributes(G_mod, 'cell')
    cells_to_edges = {v: [ j for j, l in edge_to_cells.items() if l == v] for k, v in edge_to_cells.items()}
    probabilities = Counter()
    cell_in_clusters = {}
    event_counter = Counter(event_labels)
    for c in set(list(cells_on_graph)):
        cell_in_clusters[c] = []
        for u, v in cells_to_edges[c]:
            probabilities[c] += event_counter[v]
            cell_in_clusters[c] += [v]
            for n in nx.descendants(G_mod, v):
                probabilities[c] += event_counter[n]
                cell_in_clusters[c] += [n]
        if num_trials:
            probabilities[c] /= num_trials
        else:
            probabilities[c] /= len(event_labels)
    return probabilities, cell_in_clusters

def get_vision_data(ANALYSIS_BASE, dataset, wnoise):
    vis_datapath = os.path.join(ANALYSIS_BASE, dataset, wnoise)
    vis_datarun = wnoise.split('/')[-1]
    vcd = vl.load_vision_data(
        vis_datapath,
        vis_datarun,
        include_neurons=True,
        include_ei=True,
        include_params=True,
        include_noise=True,
    )

    return vcd

def cluster_each_cell(signals,mask, cell_to_electrode_list, electrode_list, noise, note, cluster_delay = 7):
    num_trials, _, num_samples = signals.shape
   
    try: 
        seed_electrode = min(np.argwhere(np.sum(1-mask, axis = 1) == 0)).item()
        non_saturated_channels = list(np.argwhere(np.sum(1-mask, axis = 1) == 0).flatten())
    except:
        note += "all relevant electrodes saturated"

    all_event_cliques = []
    for cell in cell_to_electrode_list.keys():
        cell_electrode_list = cell_to_electrode_list[cell]

        electrode_list_ind = np.array([j for j, e in enumerate(electrode_list) if e in cell_electrode_list])
        significant_electrodes = np.arange(len(electrode_list_ind))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ap = sklearn.cluster.AffinityPropagation(damping=0.5, random_state=0)
            event_labels = ap.fit_predict(signals[:, electrode_list_ind, cluster_delay:].reshape((num_trials, -1)) )              
        mask_ = mask[electrode_list_ind]
        event_labels = first_merge_event_cliques_by_noise(cell_electrode_list,signals[:,electrode_list_ind], event_labels,  mask_, significant_electrodes,   noise)
        event_cliques = [set([i for i, x in enumerate(event_labels == l) if x]) for l in set(event_labels)]
        all_event_cliques += [event_cliques]

        
    return all_event_cliques

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

def convert_cliques_to_labels(cluster_cliques, num_trials):
    recluster = cluster_cliques[0]

    for cc in cluster_cliques:
        tmp = []
        for u in recluster:
            for v in cc:

                w = set.intersection(*[u,v])
                if len(w) > 0:
                    tmp += [w]
        recluster = tmp

    event_labels = np.zeros(num_trials, dtype = int)
    for i, cl in enumerate(recluster):
        event_labels[np.array(list(cl))]=i
    return event_labels

def get_cell_info(cell_types, vstim_data, compartments, noise, mutual_threshold = 0.5, cell_spike_window = 25, max_electrodes_considered = 30, rat = 2):
    total_electrode_list = []
    cell_to_electrode_list = {}
    mutual_cells = {}
    all_cells = [c for type_ in cell_types for c in vstim_data.get_all_cells_similar_to_type(type_)]
    # all_cells = [c for c in vstim_data.get_cell_ids()]


    for cell in all_cells:
        ei = vstim_data.get_ei_for_cell(cell).ei
        electrode_list = get_significant_electrodes(ei, compartments, noise)
        cell_to_electrode_list[cell] = electrode_list
        total_electrode_list += electrode_list
        mutual_cells[cell] = []
    total_electrode_list = list(set(total_electrode_list))

    for cell1_index, cell1 in enumerate(all_cells):
        for cell2_index, cell2 in enumerate(all_cells):
           
            cell1_set = set(cell_to_electrode_list[cell1])
            cell2_set = set(cell_to_electrode_list[cell2])
            ov = 0
            if min(len(cell1_set),len(cell2_set)) > 0:
                ov = len(cell1_set.intersection(cell2_set))/min(len(cell1_set),len(cell2_set))
            if ov >= mutual_threshold:
                mutual_cells[cell1] += [cell2]
    
    mutual_cells = {k:list(set(v)) for k, v in mutual_cells.items()}
    num_electrodes = ei.shape[0]
    if num_electrodes == 519:
        array_id = 1502
    else:
        array_id = 502
        
    
    return total_electrode_list, cell_to_electrode_list, mutual_cells, array_id

def compute_duplicates(vstim_data, noise):
    MIN_CORR = .975
    duplicates = set()
    cellids = vstim_data.get_cell_ids()
    for cell in cellids:
        cell_ei = vstim_data.get_ei_for_cell(cell).ei
        cell_ei_error = vstim_data.get_ei_for_cell(cell).ei_error
        cell_ei_max = np.abs(np.amin(cell_ei,axis=1))
        cell_ei_power = np.sum(cell_ei**2,axis=1)
        celltype = vstim_data.get_cell_type_for_cell(cell).lower()
        if "dup" in celltype or "bad" in celltype:
            continue 
        if "parasol" in celltype:
            celltype = 'parasol'
        elif "midget" in celltype:
            celltype = 'midget'
        elif "sbc" in celltype:
            celltype = 'sbc'
        else:
            celltype = 'other'
        for other_cell in cellids:
            other_celltype = vstim_data.get_cell_type_for_cell(other_cell).lower()
            if cell == other_cell or cell in duplicates or other_cell in duplicates:
                continue
            if "dup" in other_celltype or "bad" in other_celltype:
                continue
            if "parasol" in other_celltype:
                other_celltype = 'parasol'
            elif "midget" in other_celltype:
                other_celltype = 'midget'
            elif "sbc" in other_celltype:
                other_celltype = 'sbc'
            else:
                other_celltype = 'other'
            # Quit out if both cell types are in the big five.
            if celltype in ['parasol','midget','sbc'] and other_celltype in ['parasol','midget','sbc']:
                continue
            other_cell_ei = vstim_data.get_ei_for_cell(other_cell).ei
            other_cell_ei_max = np.abs(np.amin(other_cell_ei,axis=1))
            other_cell_ei_power = np.sum(other_cell_ei**2,axis=1)
            # Compute the correlation and figure out if we have duplicates: take the larger number of spikes.
            corr = np.corrcoef(cell_ei_power,other_cell_ei_power)[0,1]
            if corr >= MIN_CORR:
                n_spikes_cell = vstim_data.get_spike_times_for_cell(cell).shape[0]
                n_spikes_other_cell = vstim_data.get_spike_times_for_cell(other_cell).shape[0]
                # Take the larger number of spikes, unless the one with fewer is a light responsive type.
                if celltype in ['parasol','midget','sbc'] or n_spikes_cell > n_spikes_other_cell:
                    duplicates.add(other_cell)
                else:
                    duplicates.add(cell)
                    


    for cell in set(cellids).difference(duplicates):
        cell_ei_error = vstim_data.get_ei_for_cell(cell).ei_error[noise != 0]
        
        if np.any(cell_ei_error == 0):
            duplicates.add(cell)     
    return duplicates, cell_ei