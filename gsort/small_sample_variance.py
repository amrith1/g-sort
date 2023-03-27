import bin2py
import visionloader as vl
import numpy as np
from visionloader import SpikesReader
import pickle
import os
import electrode_map as elcmp
import old_labview_data_reader as oldlv
from scipy.io import loadmat
import elecresploader as el
from scipy.interpolate import CubicSpline
import scipy as sp
import sys
import copy
import pdb
import random
import matplotlib.pyplot as plt

# Project specific imports.
from ellipsort_config import *
from build_data_matrix import *

def get_sample_variances(cells_considered, PARENT_ANALYSIS, PARENT_DATA, dataset, datarun, electrode_list, num_trials = 25, batch_size = 1000):

    # Command line arguments have it all.
    analysis_path = os.path.join(PARENT_ANALYSIS,dataset,datarun)
    raw_data_path = os.path.join(PARENT_DATA,dataset,datarun)

    vcd = vl.load_vision_data(analysis_path ,
                              datarun,
                              include_params=True,
                              include_ei=True,
                              include_neurons=True)

    with vl.GlobalsFileReader(analysis_path,datarun) as gbfr:
        array_id = gbfr.get_image_calibration_params().array_id

    adj_mat = elcmp.get_litke_array_adj_mat_by_array_id(array_id)
    num_electrodes = vcd.get_electrode_map().shape[0]
    
    # Initialize raw file object, make the raw neurons map, and get spike times.
    pbr = bin2py.PyBinFileReader(raw_data_path)
    raw_neurons_map = make_raw_neurons_map(analysis_path,datarun,num_electrodes)
    nr = vl.NeuronsReader(analysis_path,datarun,NEURONS_RAW_EXT)
    spike_times_neurons_raw = nr.get_spike_sample_nums_for_all_real_neurons()
    nr.close()
        
    final_cell_variance = np.zeros((len(cells_considered),len(electrode_list)*(L_SAMPLES + R_SAMPLES+1))) 
    for k, cell in enumerate(cells_considered):
        
            
        # Get the seed electrode from Vision and it's neighbors. 
        with vl.NeuronsReader(analysis_path,datarun) as nr:
            seed_elec = nr.get_identifier_electrode_for_neuron(cell)

            
        # Get the spike times on this electrode, and specific to this cell. 
        spike_times = get_spike_times_seed_elec(analysis_path,datarun,
                                                    num_electrodes,seed_elec,
                                                    spike_times_neurons_raw)

        cell_variance = np.zeros(len(electrode_list)*(L_SAMPLES + R_SAMPLES+1))
        for i in range(batch_size):
            selected_times = random.sample(list(spike_times),k=num_trials)
            data_matrix_dict = get_data_matrix(np.array(selected_times), electrode_list, pbr,raw_neurons_map,spike_times_neurons_raw)
            
            
            plt.plot(data_matrix_dict['data_matrix'])
            
            cell_variance += np.var(data_matrix_dict['data_matrix'], axis = 1)/batch_size
        
        final_cell_variance[k, :] = cell_variance
    return cell_variance
        