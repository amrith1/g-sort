import os
import numpy as np
import pickle
import tqdm
import gsort.multielec_utils as mutils
import re 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gsort.old_labview_data_reader import *
import gsort.utilities.visionloader as vl

def get_bootstrap_information(ANALYSIS_BASE, gsort_path,dataset, estim, wnoise, ps, n, electrodes_stored = True):
    electrical_path = os.path.join(ANALYSIS_BASE, dataset, estim)
    relevant_movies = []
    relevant_paths = []
    filepath = os.path.join(gsort_path, dataset, estim, wnoise)
    for p in ps:
        # filepath = os.path.join(gsort_path, 
        #                         dataset, estim, wnoise, "p" + str(p))
      
        relevant_movies += [f for f in os.listdir(filepath) if (f"_p{p}.pkl" in f) and (f"_n{n}_" in f)]
    num_movies = 0
    total_probs = []
    edge_probs = []
    clusters = []
    edges = []
    num_trials = []
    electrodes = []
    pattern_movies = []
    
    for p, cell_pattern in tqdm.tqdm(zip(ps, relevant_movies)):
        with open(os.path.join(filepath,cell_pattern), "rb") as f:
            prob_dict = pickle.load(f)
        for k in sorted(prob_dict.keys()):
            
            pattern_movies += [(p,k)]
            total_probs += [prob_dict[k]["cosine_prob"][0]]
            if 'edge_probs' in prob_dict[k].keys():
                edge_probs+=[prob_dict[k]['edge_probs']]
            else:
                edge_probs+=[np.array([-1])]
            clusters += [prob_dict[k]['clustering']]

            edge_set = []
            for e,c in prob_dict[k]['graph_info'][2].items():
                if c == n:
                    edge_set += [e]
            edges += [edge_set]

            num_trials += [prob_dict[k]["num_trials"]]
   
            if electrodes_stored:
                electrodes = prob_dict[k]['electrode_list']
    total_probs = np.array(total_probs)
    return pattern_movies, total_probs, edge_probs, clusters, edges, electrodes, num_trials



def get_movies_indices(relevant_movies):
    return [int(re.findall('\d+', m)[-1]) for m in relevant_movies]
    

def get_pattern_movies_indices(relevant_movies):
    return [[int(re.findall('\d+', m)[-2]), int(re.findall('\d+', m)[-1])] for m in relevant_movies]
    
def get_sorted_movies_filenames(relevant_movies):
    movies = get_movies_indices(relevant_movies)
    movie_sorting = np.argsort(movies)
    return list(np.array(relevant_movies)[movie_sorting]), movie_sorting


def make_surface_plot(relevant_movies, total_probs, ANALYSIS_BASE,dataset, estim, p, n):
    electrical_path = os.path.join(ANALYSIS_BASE, dataset, estim)
    
    triplet_elecs = mutils.get_stim_elecs_newlv(electrical_path, p)
    amplitudes = mutils.get_stim_amps_newlv(electrical_path, p)
    _, movie_sorting = get_sorted_movies_filenames(relevant_movies)
    
    sorted_probs = total_probs[movie_sorting]

    p_thr = 2/19
    p_upper = 1

    good_inds = np.where((sorted_probs > p_thr) & (sorted_probs < p_upper))[0]

    fig = plt.figure()
    ax = Axes3D(fig)
    plt.xlabel(r'$I_1$')
    plt.ylabel(r'$I_2$')
    ax.set_zlabel(r'$I_3$')

    scat = ax.scatter(amplitudes[:, 0][good_inds], 
                amplitudes[:, 1][good_inds],
                amplitudes[:, 2][good_inds], marker='o', s=20, c=sorted_probs[good_inds], alpha=0.8)
    ax.set_title(f"n={n}, p={p}")
    clb = plt.colorbar(scat)
    clb.set_label('Activation Probability')
    plt.show()



def make_lollipop_comparison_plot(all_relevant_movies, all_total_probs, all_new_total_probs, all_new_instance_probs, ANALYSIS_BASE,dataset, estim):
    pattern_movies_indices = np.array(get_pattern_movies_indices(all_relevant_movies))
    for p in set(pattern_movies_indices[:,0]):
        sel = np.argwhere(pattern_movies_indices[:,0]== p).flatten()
        relevant_movies  = [all_relevant_movies[i] for i in sel]
        total_probs = all_total_probs[sel]
        new_total_probs = all_new_total_probs[sel]
        new_instance_probs = all_new_instance_probs[sel]

        electrical_path = os.path.join(ANALYSIS_BASE, dataset, estim)
        
        amplitudes = mutils.get_stim_amps_newlv(electrical_path, p)
        _, movie_sorting = get_sorted_movies_filenames(relevant_movies)
        
        sorted_probs = total_probs[movie_sorting]
        sorted_new_probs = new_total_probs[movie_sorting]
        sorted_instance_probs = new_instance_probs[movie_sorting]
    

        fig = plt.figure()
        
        ax = plt.subplot(211)
        ax.set_title(f"p={p}")
        ax.plot(-amplitudes[:,0], sorted_probs,"-o")
        ax.plot(-amplitudes[:,0], sorted_new_probs,"-o")
        ax.plot(-amplitudes[:,0], sorted_instance_probs,"-o")
        ax.set_ylim([0,1])

        ax = plt.subplot(212)
        ax.plot(sorted_probs,"-o")
        ax.plot(sorted_new_probs,"-o")
        ax.plot(sorted_instance_probs,"-o")
        ax.set_ylim([0,1])
      
        plt.show()

def plot_sigmoid(p, all_relevant_movies, all_total_probs, all_new_total_probs, all_new_instance_probs, ANALYSIS_BASE,dataset, estim):
    pattern_movies_indices = np.array(get_pattern_movies_indices(all_relevant_movies))
    
    sel = np.argwhere(pattern_movies_indices[:,0]== p).flatten()
    relevant_movies  = [all_relevant_movies[i] for i in sel]
    total_probs = all_total_probs[sel]
    new_total_probs = all_new_total_probs[sel]
    new_instance_probs = all_new_instance_probs[sel]

    electrical_path = os.path.join(ANALYSIS_BASE, dataset, estim)
    
    amplitudes = mutils.get_stim_amps_newlv(electrical_path, p)
    _, movie_sorting = get_sorted_movies_filenames(relevant_movies)
    
    sorted_probs = total_probs[movie_sorting]
    sorted_new_probs = new_total_probs[movie_sorting]
    sorted_instance_probs = new_instance_probs[movie_sorting]


    fig = plt.figure()
    
    ax = plt.subplot(211)
    ax.set_title(f"p={p}")
    ax.plot(-amplitudes[:,0], sorted_probs,"-o")
    ax.plot(-amplitudes[:,0], sorted_new_probs,"-o")
    ax.plot(-amplitudes[:,0], sorted_instance_probs,"-o")
    ax.set_ylim([0,1])

    ax = plt.subplot(212)
    ax.plot(sorted_probs,"-o")
    ax.plot(sorted_new_probs,"-o")
    ax.plot(sorted_instance_probs,"-o")
    ax.set_ylim([0,1])
    
    plt.show()

def get_difference_signals(pattern_movies,clusters,edges,ANALYSIS_BASE, dataset, estim, electrodes):
    electrical_path = os.path.join(ANALYSIS_BASE, dataset, estim)
    
    all_ds = []
    all_ds_inds = []
    all_ds_stack =  []
    count =  0
    movie_stack = []
    num_electrodes = 0
    num_samples = 0 
    for kk, (p, k) in enumerate(tqdm.tqdm(pattern_movies)):
        
        edge_ds = []
        edge_ds_inds = []
        if len(edges[kk]):
            signal = get_oldlabview_pp_data(electrical_path, p, k)[:,electrodes,:55]
            _, num_electrodes, num_samples = signal.shape

        for e in edges[kk]:
            Y_cluster_indices = np.argwhere(clusters[kk]==e[1]).flatten()
            Y = signal[Y_cluster_indices]

            X_cluster_indices = np.argwhere(clusters[kk]==e[0]).flatten()
            X = signal[X_cluster_indices]
            
            ds_inds = [(i, j) for i in  Y_cluster_indices for j in X_cluster_indices]
            ds = Y[:,None,:,:] - X[None,:,:,:]

  
            ds = ds.reshape((len(X) * len(Y), len(electrodes), -1))
            movie_stack += [k] * (len(X) * len(Y))
            edge_ds_inds += [ds_inds]
            edge_ds += [ds]
            count += 1
        all_ds += [edge_ds]
        all_ds_inds += [edge_ds_inds]
        all_ds_stack += edge_ds 
    all_ds_stack = np.vstack(all_ds_stack)
    return all_ds, all_ds_stack, all_ds_inds, num_electrodes, num_samples, movie_stack

def compute_average_diff_signal_and_error(dss,dss_stack, num_electrodes, num_samples, best_electrode):
    latencies = np.argmin(dss_stack, axis = 2)
    print(latencies.shape)
    latencies = np.tile(np.argmin(dss_stack[:,best_electrode,:], axis = 1), (num_electrodes,1 )).T
    print(latencies.shape)
    avg_latency = np.median(latencies, axis = 0)
    latency_deviation = latencies-avg_latency
    
    start_j = np.maximum(latency_deviation, 0).astype(int)
    start_i = np.maximum(-latency_deviation, 0).astype(int)
    end_j = np.minimum(55+latency_deviation, 55).astype(int)
    end_i = np.minimum(55-latency_deviation, 55).astype(int)


    shifted_dss_stack = np.zeros_like(dss_stack)
    best_error_stack = np.zeros(len(dss_stack))
    worst_error_stack = np.zeros(len(dss_stack))
    for i, es in enumerate(dss_stack):
        for e, s in enumerate(es):
            # shifted_dss_stack[i, e, start_i[i,e]:end_i[i,e]] = dss_stack[i, e, start_j[i,e]:end_j[i,e]]
            shifted_dss_stack[i, e, :] = np.roll(dss_stack[i, e], -int(latency_deviation[i,e]))#[:end_j[i,e]-start_j[i,e]]
    
    average_ds = np.mean(shifted_dss_stack, axis = 0)
    error_stack  = np.linalg.norm(shifted_dss_stack - average_ds[None], axis = 2)
    # error_stack  = np.max(shifted_dss_stack - average_ds[None], axis = 2)
    
    average_error = np.mean(error_stack, axis = 0)
    std_error= np.std(error_stack, axis = 0)
    
    
    count = 0
    shifted_dss = []
    error = []
    for i in dss:
        edge_ds = []
        edge_error = []
        for j in i:
            tmp_shifted_array = []
            tmp_error_array = []
            for k in j:
                tmp_shifted_array += [shifted_dss_stack[count][None]]
                tmp_error_array += [error_stack[count]]
                count += 1
            edge_ds += [np.vstack(tmp_shifted_array)]
            edge_error += [np.array(tmp_error_array)]
        shifted_dss += [edge_ds]
        error += [edge_error]
        
    return average_ds, average_error,std_error, shifted_dss, shifted_dss_stack,error, error_stack

def get_mean_edge_error(error):
    mean_edge_error = []
    for i in error:
        tmp_mean_error = []
        for j in i:
            tmp_mean_error += [np.mean(j, axis = 0 )]
        mean_edge_error += [tmp_mean_error]
    return mean_edge_error


def get_modified_edge_probs(mean_edge_error, edge_probs, average_error, std_error, factor, test_set):
    new_edge_probs = []
    test_set = np.array(test_set)
    for i1, i2 in zip(mean_edge_error, edge_probs):
        tmp_edge_probs = []
        if not len(i1):
            tmp_edge_probs += [0]
        for j1, j2 in zip(i1,i2):
            tmp_prob = 0
            if np.all(j1[test_set] < average_error[test_set] +factor * std_error[test_set]) :
                tmp_prob = j2
            tmp_edge_probs+=[tmp_prob]
        new_edge_probs+= [np.array(tmp_edge_probs)]
    return new_edge_probs


def plot_spike_split_on_electrode(shifted_dss_stack,average_ds, error_stack,average_error, std_error, factor,electrode , test_set):
    test_set = np.array(test_set)
    plt.figure()
    ax = plt.subplot(212)
    ax.plot(shifted_dss_stack[ ~np.all(error_stack[:,test_set]<= (average_error[test_set]+factor*std_error[test_set]), axis = 1),electrode,:].T, color = "C0")

    ax.plot(average_ds[electrode,:].T, color = "C1")
    ax.set_title("Bad spikes")
    ax = plt.subplot(211)
    ax.plot(shifted_dss_stack[ np.all(error_stack[:,test_set]<=  (average_error[test_set]+factor*std_error[test_set]), axis = 1),electrode,:].T, color = "C0")
    ax.plot(average_ds[electrode,:].T, color = "C1")

    ax.set_title("Good spikes")
    per_good_spikes = np.mean(np.all(error_stack[:,test_set]<= (average_error[test_set]+factor*std_error[test_set]), axis = 1))
    plt.suptitle(f"% good spikes = {per_good_spikes}")
    plt.show()

def get_instance_modified_edge_probs(edge_probs, error, dss_inds, average_error, std_error, factor, test_set, num_trials):
    good_nodes = []
    net_nodes = []
    test_set = np.array(test_set)
    for i1, i3 in zip(error, dss_inds):
        i4 = []
        i5 = []
        for j1, j3 in zip(i1, i3):
            j4 = []
            j5 = []
            for k1, k3 in zip(j1, j3):
                k4 = []
                # print("k1", k1[test_set])
                # print("avg_error", average_error[test_set])
                if np.all(k1[test_set] < average_error[test_set] +factor * std_error[test_set]) :
                    
                    k4 += [k3[0]]
                j4 += k4
                j5 += [k3[0]]
            i4 += [len(list(set(j4)))]
            i5 += [len(list(set(j5)))]
        
        net_nodes += [i4]
        good_nodes += [i5]
    modified_prob = np.zeros(len(good_nodes))


    for ii, (i1, i2, i3) in enumerate(zip(edge_probs, good_nodes, net_nodes)):
        all_bad_assignments = True
        for j1, j2, j3 in zip(i1, i2, i3 ):
            modified_prob[ii] += j1 - (j2 - j3)/num_trials[ii] if j3 else 0
    return modified_prob

def make_dss_comparison(p, average_ds, dss, all_relevant_movies, all_total_probs, all_new_total_probs, all_new_instance_probs, ANALYSIS_BASE,dataset, estim,error, average_error, std_error, factor, test_set):
    pattern_movies_indices = np.array(get_pattern_movies_indices(all_relevant_movies))
    sel = np.argwhere(pattern_movies_indices[:,0]== p).flatten()
    relevant_movies  = [all_relevant_movies[i] for i in sel]

    total_probs = all_total_probs[sel]
    new_total_probs = all_new_total_probs[sel]
    new_instance_probs = all_new_instance_probs[sel]

    electrical_path = os.path.join(ANALYSIS_BASE, dataset, estim)
    
    amplitudes = mutils.get_stim_amps_newlv(electrical_path, p)
    _, movie_sorting = get_sorted_movies_filenames(relevant_movies)
    
    sorted_probs = total_probs[movie_sorting]
    sorted_new_probs = new_total_probs[movie_sorting]
    sorted_instance_probs = new_instance_probs[movie_sorting]


    fig = plt.figure()
    
    ax = plt.subplot(211)
    ax.set_title(f"p={p}")
    ax.plot(-amplitudes[:,0], sorted_probs,"-o")
    ax.plot(-amplitudes[:,0], sorted_new_probs,"-o")
    ax.plot(-amplitudes[:,0], sorted_instance_probs,"-o")
    ax.set_ylim([0,1])

    ax = plt.subplot(212)
    ax.plot(sorted_probs,"-o")
    ax.plot(sorted_new_probs,"-o")
    ax.plot(sorted_instance_probs,"-o")
    ax.set_ylim([0,1])
    
    for ss, s in enumerate(sel[movie_sorting]):
        fig = plt.figure()
        ax = plt.subplot(211)
        ax2 = plt.subplot(212)
        ax.set_title(f"k={ss}")
        
        for dsds, ds in enumerate(dss[s]):
            for ds_sds_s, ds_s in enumerate(ds):
                if np.all(error[s][dsds][ds_sds_s][test_set] < average_error[test_set] +factor * std_error[test_set]) :
                    ax.plot(ds_s.flatten(), color = f"C{dsds}" )
                else:
                    ax2.plot(ds_s.flatten(), color = f"C{dsds}")
        ax.plot(average_ds.flatten(), color = "black")
        ax2.plot(average_ds.flatten(), color = "black")

    plt.show()

def sweep_I3(I1, I2, amplitudes):
    indices = []
    amps = []
    for i, a in enumerate(amplitudes):
        if a[1]==I2 and a[0]==I1:
            indices += [i]
            amps += [a[2]]
            
    return indices,amps
    