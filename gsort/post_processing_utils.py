import numpy as np
from scipy.optimize import curve_fit
from gsort.run_gsort_v2_wuericmod import *

def enforce_noisy_monotonicity(sigmoid, st, noise_limit):
    
    thr = np.argwhere(sigmoid >= st).flatten()[0]
    J_array = []
    max_value = st
    trigger = False
    for i in range(len(sigmoid)):
        if sigmoid[i] >= max_value*noise_limit:
            max_value = sigmoid[i]
            trigger = True
            J_array += [1]
        else:
            if not trigger:
                J_array += [1]
            else:
                J_array += [0]
    #print(J_array)
    J_array = np.array(J_array).astype(np.int16)
    if J_array[0] == 1 and sum(J_array) == 1:
        J_array[-1] = 1
    return J_array

def enforce_noisy_monotonicity_add(sigmoid, st, noise_limit):
    
    thr = np.argwhere(sigmoid >= st).flatten()[0]
    J_array = []
    max_value = st
    trigger = False
    for i in range(len(sigmoid)):
        if sigmoid[i] >= max_value - noise_limit:
            max_value = sigmoid[i]
            trigger = True
            J_array += [1]
        else:
            if not trigger:
                J_array += [1]
            else:
                J_array += [0]
    #print(J_array)
    J_array = np.array(J_array).astype(np.int16)
    if J_array[0] == 1 and sum(J_array) == 1:
        J_array[-1] = 1
    return J_array

def fsigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a*(x-b)))
    
    
def infer_sigmoid(auto_sigmoid,auto_amplitude, mono_threshold=0.5, noise_limit=0.8 , kind='mul'):
    if kind=='mul':
        if max(auto_sigmoid) >= mono_threshold:
            finalized = enforce_noisy_monotonicity(auto_sigmoid, mono_threshold, noise_limit)
        else:
            finalized = np.ones(len(auto_sigmoid)).astype(np.int16)
    elif kind=='add':
        if max(auto_sigmoid) >= mono_threshold:
            finalized = enforce_noisy_monotonicity_add(auto_sigmoid, mono_threshold, noise_limit)
        else:
            finalized = np.ones(len(auto_sigmoid)).astype(np.int16)


    try:
        popt, pcov = curve_fit(fsigmoid, auto_amplitude[finalized==1], auto_sigmoid[finalized==1])
   
        return popt[1], popt[0]/4

    except RuntimeError:
        asf = auto_sigmoid[finalized==1]
        aaf = auto_amplitude[finalized==1]

        est_slp = 0 
        thr = 0 
        if max(asf) >= 0.5:
            j = list(asf >= 0.5).index(True)
            est_slp = (asf[j] - asf[j-1])/(aaf[j] - aaf[j-1])
            thr = (0.5 - asf[j-1])/est_slp +  aaf[j-1] 

      
        return thr, est_slp
    
def filter_cell(cell, vstim_data, cell_spike_window = 25, rat = 2, max_electrodes_considered = 30, pt = 1.5):
    """
    cell: ID; int
    vstim_data: visionloader data
    """
    noise = vstim_data.channel_noise
    ei = vstim_data.get_ei_for_cell(cell).ei
    num_electrodes = ei.shape[0]
    if num_electrodes == 519:
        array_id = 1502
    else:
        array_id = 502

    cell_power = ei**2
    power_ordering = np.argsort(cell_power, axis = 1)[:,::-1]
    significant_electrodes = np.argwhere(np.sum(np.take_along_axis(cell_power, power_ordering[:,:cell_spike_window], axis = 1), axis = 1) >= rat * cell_spike_window * np.array(noise)**2).flatten()
    print("significant_electrodes", significant_electrodes)


    e_sorted = np.argsort(np.sum(ei**2, axis = 1))[::-1]
    e_sorted = [e for e in e_sorted if eil.axonorsomaRatio(ei[e,:]) == 'soma' or eil.axonorsomaRatio(ei[e,:]) == 'mixed']
    electrode_list = list(e_sorted[:min(max_electrodes_considered,len(significant_electrodes))])
    print("significant_electrodes", significant_electrodes)
    sub_cell_power = np.sum(ei[electrode_list]**2)
    
    sub_noise_power = np.sum(rat * cell_spike_window * np.array(noise[electrode_list])**2)
    print(sub_cell_power, sub_noise_power)
    return sub_cell_power < sub_noise_power


def disambiguate_sigmoid(sigmoid_, spont_limit = 0.2, noise_limit = 0.0):
    sigmoid = copy.copy(sigmoid_)
    if np.max(sigmoid) <= spont_limit:
        return sigmoid
    above_limit = np.argwhere(sigmoid > spont_limit).flatten()
    
    i = np.argmin(np.abs(sigmoid[above_limit]-0.5))
    upper_tail = sigmoid[above_limit[i]:]
    upper_tail[upper_tail<=noise_limit] = 1
    
    sigmoid[above_limit[i]:] = upper_tail
    return sigmoid

def diff_sig_likelihood(i, num_diff_sig_inst, errors, fun_distribution):
    indices = np.cumsum([0]+num_diff_sig_inst)
    selected_errors = errors[indices[i]:indices[i+1]]
    return np.mean([fun_distribution.pdf(e) for e in selected_errors])

def diff_sig_error(i, num_diff_sig_inst, errors, fun_distribution):
    indices = np.cumsum([0]+num_diff_sig_inst)
    selected_errors = errors[indices[i]:indices[i+1]]
    return np.mean([e for e in selected_errors])


    
