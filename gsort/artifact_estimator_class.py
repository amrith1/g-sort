import numpy as np
import sys
# import gsort.utilities import visionloader as vl


import gsort.utilities.visionloader as vl
print(vl.__file__)


from gsort.old_labview_data_reader import *
import collections
from scipy.optimize import curve_fit
import copy
import gsort.utilities.electrode_map
import time
from scipy.io import loadmat,savemat
import pickle
import math
import scipy.special
import itertools

class TemplateLoader:
    """
    Load templates for electrical spike sorting. This base class includes various methods
    to store, preprocess, and subselect templates based on relevance. The templates are 
    sourced from the Python Vision API.
    
    Example usage:
    tl = TemplateLoader('/Volumes/Analysis/', '2015-11-09-3/', 'full_resolution/', 'data000')
    tl = TemplateLoader('/Volumes/Analysis/', '2015-11-09-3/', 'full_resolution/', 'data000', include_noise = False); relevant for spike sorting data from Softwares like Kilosort (which don't automatically compute the noise parameter); use the first version otherwise.
    
    After instantiating a TemplateLoader per above, you have to load the relevant EIs. 
    You can do this with 'store_cells_from_list' or 'store_cell_of_types'. For simplicity,
    this only adds to the data already loaded into the TemplateLoader object; so, for instance,
    if you call 'store_cells_from_list' twice on the same list of cell ids, you will store two 
    copies of the same template data. This would be silly of you to do and potentially dangerous
    depending on the e-stim spikesorting scheme you are using. 
    
    If you want to remove some of the templates according to a criteria, like significant signal
    on a selected electrode, you can use 'remove_templates_by_snr' or 'remove_templates_by_threshold.'
    
    If you want to remove all the templates loaded, 'reset_templates' does that for you. 

    If you want to mask your templates (remove electrodes irrelevant for e-stim spike sorting), 
    call 'mask_non_main_channels,' 'mask_by_snr,' or 'mask_by_threshold.'
    
    """
    def __init__(self, analysis_path, subfolders, datarun, array_id = 502, include_noise = True):
        spikesorting = vl.load_vision_data(analysis_path + subfolders + datarun,
                                  datarun,
                                  include_params=True,
                                  include_ei=True,
                                  include_neurons = True,
                                  include_noise=include_noise)
        self.analysis_path = analysis_path
        self.subfolders = subfolders
        self.datarun = datarun
        self.full_analysis_path = analysis_path + subfolders + datarun
        self.include_noise = include_noise

        self.spikesorting = spikesorting
        
        self.cellids = []
        self.celltypes = []
        self.templates = []
        self.templates_variance = []
        
        self.noise = self.spikesorting.channel_noise
        
        self.groups = {}
        array_map = gsort.utilities.electrode_map.get_litke_array_adj_mat_by_array_id(array_id)
        for center, surround in enumerate(array_map):
            self.groups[center] = [center] + surround
        
        
    def store_cells_from_list(self, celllist):
        '''
        Store templates from a list of cell IDs
        '''
        self.cellids += celllist
        error = np.ones_like(self.spikesorting.get_ei_for_cell(self.cellids[0]).ei)*self.noise[:,None]
        
        for cellid in celllist:
            self.templates += [self.spikesorting.get_ei_for_cell(cellid).ei]
            self.templates_variance += [error]
            
            self.celltypes += [self.spikesorting.get_cell_type_for_cell(cellid)]
            
        
    def store_cells_of_types(self, celltypes):
        '''
        Store templates from all cells of types in list of cell types.
        '''
        for celltype in celltypes:
            
            cellids = self.spikesorting.get_all_cells_similar_to_type(celltype)
            self.cellids += cellids
            self.celltypes += [celltype]*len(cellids)
            for cellid in cellids:
                self.templates += [self.spikesorting.get_ei_for_cell(cellid).ei]
                self.templates_variance += [self.spikesorting.get_ei_for_cell(cellid).ei_error]
                
    def store_good_cells(self):
        '''
        Store templates from all cells of types in list of cell types.
        '''
        
        cellids = list(set(self.spikesorting.get_cell_ids()) - set(self.spikesorting.get_all_cells_similar_to_type('bad')) - set(self.spikesorting.get_all_cells_similar_to_type('duplicate')))
        self.cellids += cellids
        for cellid in cellids:
            self.celltypes += [self.spikesorting.get_cell_type_for_cell(cellid)]
            self.templates += [self.spikesorting.get_ei_for_cell(cellid).ei]  
            self.templates_variance += [self.spikesorting.get_ei_for_cell(cellid).ei_error]
            
    def store_all_cells_except(self, excluded_types):
        '''
        Store templates from all cells of types in list of cell types.
        '''
        
        exclude_cellids = [self.spikesorting.get_all_cells_similar_to_type(type_) for type_ in excluded_types]
        exclude_cellids = itertools.chain(*exclude_cellids)
        
        
        cellids = list(set(self.spikesorting.get_cell_ids()) - set(exclude_cellids) )
        self.cellids += cellids
        for cellid in cellids:
            self.celltypes += [self.spikesorting.get_cell_type_for_cell(cellid)]
            self.templates += [self.spikesorting.get_ei_for_cell(cellid).ei]  
            self.templates_variance += [self.spikesorting.get_ei_for_cell(cellid).ei_error]
            
    def store_all_cells_except_with_noise(self, excluded_types):
        '''
        Store templates from all cells of types in list of cell types.
        '''
        
        exclude_cellids = [self.spikesorting.get_all_cells_similar_to_type(type_) for type_ in excluded_types]
        exclude_cellids = itertools.chain(*exclude_cellids)
        
        
        cellids = list(set(self.spikesorting.get_cell_ids()) - set(exclude_cellids) )
        error = np.ones_like(self.spikesorting.get_ei_for_cell(cellids[0]).ei)*self.noise[:,None]
        
        self.cellids += cellids
        for cellid in cellids:
            self.celltypes += [self.spikesorting.get_cell_type_for_cell(cellid)]
            self.templates += [self.spikesorting.get_ei_for_cell(cellid).ei]  
            self.templates_variance += [error]


    def store_good_cells_no_crap(self):
        '''
        Store templates from all cells of types in list of cell types.
        '''
        
        cellids = list(set(self.spikesorting.get_cell_ids()) - set(self.spikesorting.get_all_cells_similar_to_type('bad')) - set(self.spikesorting.get_all_cells_similar_to_type('duplicate'))-set(self.spikesorting.get_all_cells_similar_to_type('crap')))
        self.cellids += cellids
        for cellid in cellids:
            self.celltypes += [self.spikesorting.get_cell_type_for_cell(cellid)]
            self.templates += [self.spikesorting.get_ei_for_cell(cellid).ei]  
            self.templates_variance += [self.spikesorting.get_ei_for_cell(cellid).ei_error]
    
    def reset_templates(self):
        '''
        Remove all templates and associated data
        '''
        self.cellids = []
        self.celltypes = []
        self.templates = []
        self.templates_variance = []
    
    def get_template_from_cell(self, cell):
        return self.templates[cell == self.cellids]
        
    
    def mask_non_main_channels(self):
        '''
        For each template, mask electrodes that are not either the cells identifier 
        electrode (computed in Vision) or its neighbors
        ''' 
        with vl.NeuronsReader(self.full_analysis_path,self.datarun) as nr:
            for i, cellid in enumerate(self.cellids):
                seed_elec_ind = nr.get_identifier_electrode_for_neuron(cellid)-1
                group = self.groups[seed_elec_ind]
                mask = np.array([bool(i in group) for i in range(self.templates[0].shape[0])])
                self.templates[i][~mask, :] = 0
                
                
    def mask_non_significant_channels(self):
        '''
        For each template, mask electrodes that are not either the cells largest electrode
        or its neighbors
        ''' 

        for i, cellid in enumerate(self.cellids):
            template = self.templates[i]
            largest_electrode = np.argmax(np.max(np.abs(template), axis = 1)).item()
            group = self.groups[largest_electrode]
            mask = np.array([bool(i in group) for i in range(self.templates[0].shape[0])])
            self.templates[i][~mask, :] = 0
            
    def mask_from_list(self, electrode_list):
        '''
        For each template, mask electrodes that are not either the cells largest electrode
        or its neighbors
        ''' 

        for i, cellid in enumerate(self.cellids):
            template = self.templates[i]
            group = electrode_list
            mask = np.array([bool(i in group) for i in range(self.templates[0].shape[0])])
            self.templates[i][~mask, :] = 0
    
    def mask_by_snr(self, ratio):
        
        if self.include_noise:
            noise = self.spikesorting.channel_noise
        else:
            noise = np.ones((self.templates[0].shape[0],))
            
        for i, cellid in enumerate(self.cellids):
            template = self.templates[i]
            
            self.templates[i][np.max(np.abs(template), axis = 1) < noise*ratio, :] = 0
            
            
    def mask_by_latency(self, latency = 1):
        noise = self.spikesorting.channel_noise
        for i, cellid in enumerate(self.cellids):
            templates = self.templates[i]
       
            peak_samples = np.argmin(templates, axis = 1)
            peak_vals = templates[np.arange(len(peak_samples)), peak_samples]
            peak_elec = np.argmax(np.abs(peak_vals))
            
            
            self.templates[i][((np.abs(peak_samples - peak_samples[peak_elec]) >= latency+1) | (np.abs(peak_vals) < noise)), :]= 0
            
    def mask_by_threshold(self, threshold):
        '''
        For each template, mask electrodes whos channel max signal < threshold.
        Ex. threshold = 30
        '''
        for i, cellid in enumerate(self.cellids):
            template = self.templates[i]
            max_signals = np.max(np.abs(template), axis = 1)
            self.templates[i][max_signals < threshold, :] = 0
            
    def mask_by_FN_rate(self, d = 55, p_threshold = 0.04, max_electrodes = 10, N_approx = 100):
        def electrode_FN_rate(u, d1, d2, N = N_approx):
            cdf = 0
            for j in range(N):
                cdf += (0.5*u)**j/math.factorial(j)*math.exp(-0.5*u)*scipy.special.betainc(0.5*d1+j, 0.5*d2, 0.5)
            return cdf

        def compute_total_FN_rate(FN_rate):
            num_electrodes = len(FN_rate)
            combinations = list(map(np.array, list(itertools.product([0, 1], repeat=num_electrodes))))[1:]
            Pr = 0
            for comb in combinations:
                Pr += np.prod(comb*np.array(FN_rate)+(1-comb)*(1-np.array(FN_rate)))
            return Pr

        def determine_max_viable_electrodes(FN_rate, p_threshold= p_threshold, max_electrodes = max_electrodes):
            FN_rate_tmp = []
            sorted_electrodes = np.argsort(FN_rate)

            for e in sorted_electrodes:
                
                FN_rate_tmp += [FN_rate[e]]
                P = compute_total_FN_rate(FN_rate_tmp)
                
                if P > p_threshold or len(FN_rate_tmp)-1== max_electrodes:
                    return sorted_electrodes[:len(FN_rate_tmp)-1] 
            return sorted_electrodes[:len(FN_rate_tmp)-1] 
                
        noise = self.spikesorting.channel_noise
        
        all_electrodes = []
        
        for n_idx, cellid in enumerate(self.cellids):
            template = self.templates[n_idx]
            
            
        
            live_electrodes = (np.max(np.abs(template), axis = 1) != 0 ) * (noise != 0)
            
            live_electrodes = np.argwhere(live_electrodes).flatten()
            
            
            
            
            mus = np.sum((template[live_electrodes, :]/noise[live_electrodes].reshape((-1,1)))**2, axis = 1)
            FN_rate = np.zeros(len(live_electrodes))

            for i, mu in enumerate(mus):
                FN_rate[i] = electrode_FN_rate(mu, d, d, N_approx)
    
    
            group = live_electrodes[determine_max_viable_electrodes(FN_rate, p_threshold=p_threshold,max_electrodes=max_electrodes )]
            
            all_electrodes += list(group)
        
            mask = np.array([bool(i in group) for i in range(self.templates[0].shape[0])])
            self.templates[n_idx][~mask, :] = 0
            
        
        return list(set(all_electrodes))
                
            
            
    def remove_templates_by_snr(self, r, n, add_surround = True):
        '''
        For each template, remove it if signal on channel r < n * noise on channel r 
        '''
        try:
            noise = self.spikesorting.channel_noise
        except UnboundLocalError:
            print('Noise parameters not loaded! Set include_noise to True or try remove_templates_by_threshold.')
            raise
           
        r -= 1
        if add_surround:
            r = self.groups[r]
        
            
        new_cellids = []
        new_templates = []
        new_celltypes = []
        new_templates_variance = []
        for i, cellid in enumerate(self.cellids):
            template = self.templates[i]
            
            if add_surround:
                signal = np.max(np.abs(template[r, :]), axis = 1)
            else:
                signal = np.max(np.abs(template[r, :]))
            celltype = self.celltypes[i]
            if (signal > n*noise[r]).any():
                new_cellids += [cellid]
                new_templates += [template]
                new_celltypes += [celltype]
                new_templates_variance += [self.templates_variance[i]]

        self.cellids = new_cellids
        self.templates = new_templates
        self.celltypes = new_celltypes
        self.templates_variance = new_templates_variance
    def remove_templates_with_zero_variance(self, electrode_list):
        '''
        For each template, remove it if signal on channel r < n * noise on channel r 
        '''
        
        
        
            
        new_cellids = []
        new_templates = []
        new_celltypes = []
        new_templates_variance = []
        for i, cellid in enumerate(self.cellids):
            templates_variance = self.templates_variance[i]
            
            if (templates_variance[electrode_list]!=0).all():
                new_cellids += [cellid]
                new_templates += [self.templates[i]]
                new_celltypes += [self.celltypes[i]]
                new_templates_variance += [self.templates_variance[i]]

        self.cellids = new_cellids
        self.templates = new_templates
        self.celltypes = new_celltypes
        self.templates_variance = new_templates_variance
        
    def remove_templates_with_excess_variance(self, electrode_list):
        '''
        For each template, remove it if signal on channel r < n * noise on channel r 
        '''
        
        
        
            
        new_cellids = []
        new_templates = []
        new_celltypes = []
        new_templates_variance = []
        for i, cellid in enumerate(self.cellids):
            templates_variance = self.templates_variance[i][electrode_list]
            template = self.templates[i][electrode_list]
            
            abs_template = np.abs(template)
            template_power = np.sum(abs_template[abs_template >= 1])
            error_power = np.sum((templates_variance)[abs_template >= 1])
            
            if template_power > error_power:
                new_cellids += [cellid]
                new_templates += [self.templates[i]]
                new_celltypes += [self.celltypes[i]]
                new_templates_variance += [self.templates_variance[i]]

        self.cellids = new_cellids
        self.templates = new_templates
        self.celltypes = new_celltypes
        self.templates_variance = new_templates_variance
        
    def remove_templates_by_elec_power(self, electrode_list, n, num_samples):

        try:
            noise = self.spikesorting.channel_noise
        except UnboundLocalError:
            print('Noise parameters not loaded! Set include_noise to True or try remove_templates_by_threshold.')
            raise
           
        
        
            
        new_cellids = []
        new_templates = []
        new_celltypes = []
        new_templates_variance = []
        
        
        for i, cellid in enumerate(self.cellids):
            template = self.templates[i]
            
            signal_power = np.sum(template[electrode_list, :]**2)
            celltype = self.celltypes[i]
            if (signal_power > (n-1)*sum([noise[e]**2 for e in electrode_list])*num_samples):
                new_cellids += [cellid]
                new_templates += [template]
                new_celltypes += [celltype]
                new_templates_variance += [self.templates_variance[i]]
                
                
                
#         for i, cellid in enumerate(self.cellids):
#             template = self.templates[i]
            
#             signal_power = np.sum(template[electrode_list, :]**2, axis = 1)
#             celltype = self.celltypes[i]
#             if sum(signal_power > np.array([3*noise[e]**2*num_samples for e in electrode_list])) >= 1:
#                 new_cellids += [cellid]
#                 new_templates += [template]
#                 new_celltypes += [celltype]
#                 new_templates_variance += [self.templates_variance[i]]


        self.cellids = new_cellids
        self.templates = new_templates
        self.celltypes = new_celltypes
        self.templates_variance = new_templates_variance
            
    def remove_templates_by_threshold(self, r, threshold, add_surround = True):
        '''
        For each template, remove it if signal on channel r < threshold
        '''
        r -= 1
        if add_surround:
            r = self.groups[r]
        
            
        new_cellids = []
        new_templates = []
        new_celltypes = []
        new_templates_variance = []
        for i, cellid in enumerate(self.cellids):
            template = self.templates[i]
            if add_surround:
                signal = np.max(np.abs(template[r, :]), axis = 1)
            else:
                signal = np.max(np.abs(template[r, :]))
            celltype = self.celltypes[i]
            if (signal >= threshold).any():
                new_cellids += [cellid]
                new_templates += [template]
                new_celltypes += [celltype]
                new_templates_variance += [self.templates_variance[i]]
        
        self.cellids = new_cellids
        self.templates = new_templates
        self.celltypes = new_celltypes
        self.templates_variance = new_templates_variance
        
    def remove_templates_by_list(self, cell_list):
        '''
        For each template, remove it if signal on channel r < threshold
        '''

        new_cellids = []
        new_templates = []
        new_celltypes = []
        new_templates_variance = []
        for i, cellid in enumerate(self.cellids):
            template = self.templates[i]
            celltype = self.celltypes[i]
            if cellid not in cell_list:
                new_templates_variance += [self.templates_variance[i]]
                new_cellids += [cellid]
                new_templates += [template]
                new_celltypes += [celltype]
        
        self.cellids = new_cellids
        self.templates = new_templates
        self.celltypes = new_celltypes
        self.templates_variance = new_templates_variance
            

class AritifactEstimator:
    
    '''
    Base class for any electrical stimulation artifact estimation method. 
    The purpose of this class is to be inheritable for any artifact estimation scheme. To 
    use it, the class has the following arguments:
        - analysis_path: path to dataset (e.g. '/Volumes/Analysis/2019-06-20-0/')
        - datarun: datarun identifier of electrical stimulation data (e.g. data001)
        - p: stimulating electrode 
        - tl:template loader instance
        - OPTIONAL: stim_recorded_samples: # of samples in EI  
        - OPTIONAL: array_id: 1502 for 519 array; 502 for 512 array
     
    This class facilitates doing simple things, like getting the artifact of a particular
    stimulation level (see get_artifact), the response probablity for a particular cell
    and amplitude (see get_response_probability_for_cell), the response probability 
    array for a cell (see get_sigmoid), the cell stimulable >0.5 efficiency 
    (see get_responsive_cells), the sigmoid parameters of responsive cells 
    (see get_responsive_cells_data). You can also reset all of the artifact estimations
    with reset_artifacts.
    
    '''
    
    def __init__(self, analysis_path, datarun, p, tl, stim_recorded_samples = 55, array_id = 502):
        self.tl = tl
        self.array_id = array_id
        array_map = gsort.electrode_map.get_litke_array_adj_mat_by_array_id(array_id)
        self.num_electrodes = len(array_map)
        self.artifacts = {}
        self.artifacts[0] = np.zeros((stim_recorded_samples*self.num_electrodes,))
        self.sigmoids = {}
        self.sorted_cells_and_spikes = {}
        self.stim_recorded_samples = stim_recorded_samples
        self.analysis_path = analysis_path
        self.datarun = datarun
        self.p = p
        Y_low = get_oldlabview_pp_data(analysis_path + datarun, p, 0)
        self.mu = np.mean(Y_low, axis = 0)[:,:stim_recorded_samples].reshape((-1,))
    
        self.num_trials = [] # To avoid complications that come with uneven trials 
                             # during stimulation-recording sessions, this is computed
                             # when autosort is run. sigh
                
        
    
    def get_artifact(self, j):
        # Return artifact estimate for particular cell and stimulation level
        try:
            return self.artifacts[j].reshape((self.num_electrodes, -1))
        except KeyError:
            raise KeyError('No artifact estimate for stimulation level %s' %j)
                        
    def get_all_artifacts(self):
        return self.artifacts
    
    def get_response_probability_for_cell(self, j, cell):
        # Return response probability for particular cell and stimulation level
        return self.sigmoids[j][cell]/self.num_trials[j]
    
    def get_sigmoid(self, cell):
        # Return cell response vector accross stimulation levels
        sigmoid = []
        for j in list(self.sigmoids.keys()):
            sigmoid += [self.sigmoids[j][cell]]
        return np.array(sigmoid)/self.num_trials
    
    def get_responsive_cells(self):
        # cells with >50% response efficiency. 
        responsive_cells = []
        for j in list(self.sigmoids.keys()):
            for cell in self.sigmoids[j]:
                if self.sigmoids[j][cell]/self.num_trials[j] >= 0.5:
                    responsive_cells += [cell]
        return list(set(responsive_cells))
    
    def get_responsive_cells_data(self, amplitudes, noisy_monotonicity = 0.8):
        # Return sigmoid parameters for cells with >50% response efficiency. 
        # Given the noisiness in spike sorting but known monotonicity of spike 
        # probability vector, choose a toleration band for including
        # data in sigmoid estimation 
        def fsigmoid(x, a, b):
            return 1.0 / (1.0 + np.exp(-a*(x-b)))
        
       

        responsive_cells = self.get_responsive_cells()
        responsive_cell_data = {}
        for cell in responsive_cells:
            sigmoid = self.get_sigmoid(cell)
            if noisy_monotonicity:
                finalized = self.enforce_noisy_monotonicity( sigmoid, noisy_monotonicity)
            else:
                finalized = np.ones(len(amplitudes)).astype(np.int16)

                
                
            try:
                popt, pcov = curve_fit(fsigmoid, amplitudes[finalized==1], sigmoid[finalized==1])
            except RuntimeError:
                responsive_cell_data[cell] = (None,None, finalized)
                continue
            responsive_cell_data[cell] = (popt[0]/4, popt[1], finalized)
        
        return responsive_cell_data
    def enforce_noisy_monotonicity(self, sigmoid, noise_limit):
        # Function to force monotonicity while allowing some noise 
        # tolerance. Works by finding the threshold where spike
        # response probability first exceeds 0.5, then tracks how the 
        # response probability changes, storing the high spike response probability
        # through stimulation levels. For each subsequent response probability,
        # if it is lower than noise_limit*maximum_response probability,
        # don't include it in the sigmoid estimate.
        thr = np.argwhere(sigmoid >= 0.5).flatten()[0]
        J_array = [1 for _ in range(thr+1)]
        max_value = sigmoid[thr]
        for i in range(thr+1, len(sigmoid)):
            if sigmoid[i] >= max_value*noise_limit:
                max_value = sigmoid[i]
                J_array += [1]
            else:
                J_array += [0]
        return np.array(J_array).astype(np.int16)
                
            
                    
    def reset_artifacts(self):
        self.artifacts = []
        self.artifacts[0] = np.zeros((self.stim_recorded_samples*self.num_electrodes,))
        
    
class SimpleAutosortVectorized(AritifactEstimator):
    '''
    The artifact estimator implements the algorithm designed by Mena et al. It
    is the simplified version of the algorithm and works by alternating between
    optimizing spike inference via match pursuit and artifact inference. 
    
    To initialize the pbject, in addition to specifying the parameters needed 
    for any ArtifactEstimator, you cam specify the shift_window, a parameter
    to determine the window of samples considered in the convolutional matching
    part of matching pursuit. The default are the samples 5 to 30. 
    '''
    def __init__(self, analysis_path, datarun, p, tl, stim_recorded_samples = 55, shift_window = [5,30], array_id = 502):
        super().__init__(analysis_path, datarun, p, tl, stim_recorded_samples, array_id)
        self.shift_window = shift_window
        self.mu = self.mu.reshape((1,-1))
        self.convergence = []
        self.sorted_cells_and_spikes = {}
        self.iterations = []
        self.max_iter = None
        self.at_iteration_artifacts = []
    
    def run_simple_autosort(self, end_j,  cell_criteria = True, max_iter = None, stop_at_break = True,include_iter_art = False):
        
        self.num_trials = np.zeros(end_j).astype(int)
        self.max_iter = max_iter
        self.convergence = 1
        if include_iter_art:
            self.at_iteration_artifacts = np.zeros((end_j, self.max_iter, len(self.artifacts[0])))
        
        for j in range(end_j):
            print('Artifact Estimation for Amplitude Level %s' %j)
            changing = True
            iteration = 0
            known_spikes = {}
            known_cellids = [-1]
            Ys = get_oldlabview_pp_data(self.analysis_path + self.datarun, self.p, j)[:,:,:self.stim_recorded_samples]
            self.num_trials[j] = Ys.shape[0]
            Ys = Ys.reshape((self.num_trials[j],-1))-self.mu
            
            while changing:
                print('Iterations %s' %iteration)
                iteration += 1
                changing = False
                spikes = self.matching_pursuit(Ys, j)
                
                
                if not cell_criteria:
                    if known_spikes != spikes:
                        known_spikes = spikes
                        changing = True
                
                cellids = self.artifact_inference_from_spikes(Ys, j, spikes)  
                
                if cell_criteria:
                    if known_cellids != cellids:
                        known_cellids = cellids
                        changing = True
                if include_iter_art:
                    self.at_iteration_artifacts[j, iteration-1, :] = self.get_artifact(j).reshape((1,-1))
                if iteration == max_iter:
                    self.convergence = 0
                    break
                
                
            if not self.convergence:
                if stop_at_break:
                    break
                
                
            self.iterations += [iteration]
            for cell in cellids:
                if cell not in self.sorted_cells_and_spikes.keys():
                    self.sorted_cells_and_spikes[cell] = np.zeros((end_j, max(self.num_trials))).astype(int)
                for trace in spikes.keys():
                    if self.tl.cellids.index(cell) in spikes[trace]:
                        self.sorted_cells_and_spikes[cell][j, trace] =  spikes[trace][self.tl.cellids.index(cell)]+self.shift_window[0]-1
            
            
            self.sigmoids[j] = collections.Counter(cellids)
            if j+1 < end_j and not (j + 1 in self.artifacts.keys()) :
                self.artifacts[j+1] = self.artifacts[j]
                    
    def matching_pursuit(self, Ys, j):
        '''
        This method implements the matching pursuit algorithm. 
        
        The algorithm works on the simple idea of greedily finding the templates
        that minimize the residual power of the artifact subtracted signal trace. In 
        order to do this quickly, all of the trials are optimized simultaneously.
        Unfortunately, this adds speed while sacrificing a bit of readability. 
        C'est la vie.
        
        As parameters, the algorithm takes in the signals traces 'Ys' and the 
        stimulation level 'j'. 
        
        Simply put, the goal of the algorithm is to keep track of the templates
        that minimize the residual power, disallow multiple addition of templates
        coming from the same cell, and stop running when no more additions
        improve (reduce) the the resdiual power. Like I said, simple. 
        
        I'll leave the rest of the comments (and jokes) as in-line ones. 
        
        '''
        
        # Get the initianal residual and residual signal power. 
        A = self.get_artifact(j).reshape((1,-1))
        residual = Ys - A
        residual_power = np.sum(residual**2, axis = 1, keepdims = True)
        print(sum(residual_power))
        
        num_traces = len(residual_power)
        num_templates = len(self.tl.templates)
        
        # Create binary tensor to store information about templates added. The first
        # dimension keeps data for the traces separate (difference traces will have
        # different templates the minimize the residual). 
        # The second dimension keeps track of the particular templates
        # added for each trace. Note +1 in the second dimension. An extra dimension 
        # is added to allow the algorithm to not add any new templates (if any template
        # subtract would increase the residual power). 
        # The third dimension is to keep track of the time information of the best template.
        # For each template, it is considered over a window of spike times to address
        # the fact that we do not know the latency of the spikes. 
        unavailable_templates = np.zeros((num_traces ,num_templates+1, self.shift_window[1]-self.shift_window[0]))
        
 
        # All variables to keep running tally of templates that improve the residual the most
        best_score = copy.copy(residual_power.flatten())
        best_template = np.zeros(num_traces).astype(int)-1
        best_b = np.zeros(num_traces).astype(int)
        
        improving = True
        while improving:
            # Store optimal templates
            found_spikes = np.zeros((num_traces, self.num_electrodes*self.stim_recorded_samples))
            
            # For each template
            for i in range(num_templates):
                
                # Create sliding window of templates or a set of "timed templates"
                M = self.create_M(i)
                
                # Compute residual power considering the removal of each of the 
                # sliding templates
                delta = residual_power -2*residual @ M + np.sum(M**2, axis = 0, keepdims = True)

                # Store the timed templates and the score that reduced the residual the most
                # for each trace
                b_prime = np.argmin(delta, axis = 1)
                score_prime = delta[np.arange(num_traces),b_prime]

                # For each trace where the timed templates reduced the 
                # residual power
                for k in np.argwhere(score_prime < best_score).flatten():

                    # and the template hasn't been used already for 
                    # that trace
                    if (unavailable_templates[k,i, :] == 0).all():
                        
                        # update the values of the best timed template
                        # removal
                        best_score[k] = score_prime[k]
                        best_template[k] = i
                        best_b[k] = b_prime[k]
                        found_spikes[k,:] = M[:,b_prime[k]].T
            # If no new templates have been added (i.e. best_template == all -1s)
            if not (best_template+1).any():
                
                # Stop running
                improving = False
            else:
                # Update the residual, the residual power, and the unavailable tempaltes
                residual -= found_spikes
                residual_power = np.sum(residual**2, axis = 1, keepdims = True)            
                unavailable_templates[np.arange(num_traces), best_template, best_b] = 1
                
                # and reset everything
                best_score = copy.copy(residual_power.flatten())
                best_template = np.zeros(num_traces).astype(int)-1
                best_b = np.zeros(num_traces).astype(int)
        
        # create a dictionary that maps traces -> templates -> template timings. 
        spikes = {}
        for i in range(num_traces):
            spikes[i] = {}
        for index in np.argwhere(unavailable_templates[:,:-1,:]):
            spikes[index[0]][index[1]] = index[2] 
        return spikes
    
    
    def artifact_inference_from_spikes(self, Ys, j, spikes):
        mean_Y = np.mean(Ys, axis = 0)
        mean_spikes = np.zeros_like(mean_Y)
        cellids = []
        for trace in spikes.keys():
            for cell in spikes[trace].keys():
                mean_spikes += self.create_M(cell)[:,spikes[trace][cell]]
                cellids += [self.tl.cellids[cell]]
        mean_spikes /= self.num_trials[j]
        self.artifacts[j] = mean_Y - mean_spikes
        return cellids

    def create_M(self, i):
        template = self.tl.templates[i]
        peak_channel, peak_sample = np.unravel_index(np.argmax(np.abs(template)), template.shape)
        M = np.zeros((self.num_electrodes*self.stim_recorded_samples, self.shift_window[1]-self.shift_window[0]))
        
        for t in range(self.shift_window[1]-self.shift_window[0]):
            shift = peak_sample-(self.shift_window[0]-1)-t
            start = max(shift, 0)
            end = start+self.stim_recorded_samples+min(shift,0)
            cutout = np.zeros((self.num_electrodes, self.stim_recorded_samples))
            cutout[:, -min(shift, 0):] = template[:, start:end]
            M[:, t] = cutout.reshape(-1,)
        return M
    
    def write_to_autosortfile(self, autosort_path):
        
        elecrespauto_path = os.path.join(autosort_path,
                                     'elecRespAuto_p%s.mat'%str(self.p))
        elecrespauto = loadmat(elecrespauto_path,struct_as_record=False,
                           squeeze_me=True)['elecRespAuto']
        pyartifact = []
        for j in range(len(self.iterations)):
            pyartifact += [self.get_artifact(j)]
        
        sorted_cells = list(self.sorted_cells_and_spikes.keys())
        sorted_spikes = list(self.sorted_cells_and_spikes.values())
            
        
        setattr(elecrespauto.stimInfo,'pyV0_Arts',pyartifact)
        setattr(elecrespauto.Log,'pyV0_Iter',np.array(self.iterations))
        setattr(elecrespauto.Log,'pyV0_Convergence',self.convergence)
        setattr(elecrespauto.neuronInfo,'pyV0_neuronIds',sorted_cells)
        setattr(elecrespauto.neuronInfo,'pyV0_spikes',sorted_spikes)
        
        elecrespauto_dict = dict()
        elecrespauto_dict['elecRespAuto'] = elecrespauto
        savemat(elecrespauto_path,elecrespauto_dict)
    
    def write_to_pyautosortfile(self, save_path):
        amplitudes_all = np.array([.1005, .1131, .1194, .1320, .1445, .1634, .1759,
    .1948, .2136, .2388, .2576, .2780, .3033, .3539, .3791, .4297, .4550, .5055,
    .5561, .6066, .6824, .7330, .8088, .8846,.9857, 1.1043, 1.2047, 1.3051,
    1.4055, 1.6063, 1.7067, 1.9075, 2.1083, 2.3091, 2.5098, 2.8110, 3.1122,
    3.4134, 3.7146, 4.1161])
        
        data = {}
        pyartifact = []
        for j in range(len(self.iterations)):
            pyartifact += [self.get_artifact(j)]
        
        sorted_cells = list(self.sorted_cells_and_spikes.keys())
        sorted_spikes = list(self.sorted_cells_and_spikes.values())
            
        data['finalized artifacts'] = pyartifact
        data['at-iteration flattened artifacts'] = self.at_iteration_artifacts
        data['iterations'] = self.iterations
        data['convergence_limit'] = self.max_iter
        data['templates considered'] = self.tl.cellids
        data['sorted neuron ids'] = sorted_cells
        data['spikes'] = sorted_spikes
        data['num trials'] = self.num_trials
        data['amplitudes'] = amplitudes_all[:len(self.iterations)]
        
        sigmoids = []
        for cell in sorted_cells:
            sigmoids += [self.get_sigmoid(cell)]
        data['sigmoids'] = sigmoids
        
        with open(save_path + 'p' + str(self.p) + '.pkl', 'wb') as f:
            pickle.dump(data, f)
            
    def write_to_manualcomparison_file(self, save_path, n, p, r, amplitudes, manual_sigmoid):
        
        
        data = {}
        sorted_cells = list(self.sorted_cells_and_spikes.keys())
        data['auto sigmoid'] = {}
        data['artifacts'] = self.artifacts
        
        for cell in sorted_cells:
            data['auto sigmoid'][cell] = self.get_sigmoid(cell)

        data['num trials'] = self.num_trials
        data['convergence'] = self.convergence
        data['manual sigmoid COI'] = manual_sigmoid
        data['amplitudes'] = amplitudes
        
        with open(save_path + str(n) + 'n_' + str(p) + 'p_' + str(r) +'r'+ '.pkl', 'wb') as f:
            pickle.dump(data, f)
            
        
   