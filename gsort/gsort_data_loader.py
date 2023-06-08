import os
import numpy as np
import tqdm
import visionloader as vl
from scipy.io import loadmat

from joblib import Memory
from .vision_template_loader_class import TemplateLoader

memory = Memory(os.getcwd())


def axonorsomaRatio(wave,uppBound=1.6,lowBound=0.05):
    try:
        #Get index and value of (negative) min {only one}
        minind = np.argmin(wave)
        minval = np.min(wave)

        #Get max vals to either side of min
        maxvalLeft = np.max(wave[0:minind])
        maxvalRight = np.max(wave[minind:])

        if np.abs(minval) < max(maxvalLeft,maxvalRight):
            rCompt = 'dendrite'
        else:
            if maxvalRight == 0:
                ratio = 0
            else:
                ratio = maxvalLeft/maxvalRight
            if ratio > uppBound:
                rCompt = 'axon'
            elif ratio < lowBound: #FUDGED
                rCompt = 'soma'
            else:
                rCompt = 'mixed'
    except ValueError:
        rCompt = 'error' #wave is abnormally shaped (usually has min at leftmost or rightmost point)

    return rCompt

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
    return cellids,  cell_eis_tmp, cell_variance_tmp, peak_spike_times

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


def get_significant_electrodes(ei, compartments, noise, cell_spike_window = 25, max_electrodes_considered = 30, rat = 2):
    cell_power = ei**2
    e_sorted = np.argsort(np.sum(ei**2, axis = 1))[::-1]
    e_sorted = [e for e in e_sorted if axonorsomaRatio(ei[e,:]) in compartments]
    cell_power = ei**2
    power_ordering = np.argsort(cell_power, axis = 1)[:,::-1]
    significant_electrodes = np.argwhere(np.sum(np.take_along_axis(cell_power[e_sorted], power_ordering[e_sorted,:cell_spike_window], axis = 1), axis = 1) >= rat * cell_spike_window * np.array(noise[e_sorted])**2).flatten()

    electrode_list = list(np.array(e_sorted)[significant_electrodes][:max_electrodes_considered])
    return electrode_list

def get_cell_info(cell_types, vstim_data, compartments, noise, mutual_threshold = 0.5):
    """
    Purpose: Return various useful bits of information around the relevant electrodes for cells and their overlap
    """
    total_electrode_list = []
    cell_to_electrode_list = {}
    mutual_cells = {}
    all_cells = [c for type_ in cell_types for c in vstim_data.get_all_cells_similar_to_type(type_)]
    
    for cell in all_cells:
        ei = vstim_data.get_ei_for_cell(cell).ei
        electrode_list = get_significant_electrodes(ei, compartments, noise)
        cell_to_electrode_list[cell] = electrode_list
        total_electrode_list += electrode_list
        mutual_cells[cell] = []
    total_electrode_list = list(set(total_electrode_list))

    for cell1 in all_cells:
        for cell2 in all_cells:
           
            cell1_set = set(cell_to_electrode_list[cell1])
            cell2_set = set(cell_to_electrode_list[cell2])
            ov = 0
            if min(len(cell1_set),len(cell2_set)) > 0:
                ov = len(cell1_set.intersection(cell2_set))/len(cell1_set)
            if ov >= mutual_threshold:
                mutual_cells[cell1] += [cell2]
    
    mutual_cells = {k:list(set(v)) for k, v in mutual_cells.items()}
    num_electrodes = ei.shape[0]
    if num_electrodes == 519:
        array_id = 1502
    else:
        array_id = 502
        
    
    return total_electrode_list, cell_to_electrode_list, mutual_cells, array_id


@memory.cache
def load_vision_data_for_gsort(estim_type:str, visual_analysis_base:str, dataset:str, vstim_datarun:str, patterns = None, cell_types = ['parasol', 'midget'], excluded_types = ['bad', 'dup']):     # RAT: 'ON' and 'OFF'
    """Load vision data for g-sort analysis to be passed to run_pattern_movie_live"""
    compartments = ['soma', 'mixed']
    vstim_analysis_path = os.path.join(visual_analysis_base, dataset, vstim_datarun)
    vstim_data = vl.load_vision_data(vstim_analysis_path,
                                    vstim_datarun.rsplit('/', maxsplit=1)[-1],
                                    include_params=True,
                                    include_ei=True,
                                    include_noise=True,
                                    include_neurons=True)

    NOISE = vstim_data.channel_noise
    duplicates, cell_ei = compute_duplicates(vstim_data, NOISE)
    MUTUAL_THRESHOLD = 1
    NOISE_THRESH = 2
    POWER_THRESHOLD = 1.5
    END_TIME_LIMIT = 30
    START_TIME_LIMIT = 0
    CLUSTER_DELAY = 0
    TIME_LIMIT = END_TIME_LIMIT - START_TIME_LIMIT
    WINDOW_BUFFER = 20

    def get_collapsed_ei_thr(cell_no, thr_factor):
        # Read the EI for a given cell
        cell_ei = vstim_data.get_ei_for_cell(cell_no).ei
        
        # Collapse into maximum value
        collapsed_ei = np.amin(cell_ei, axis=1)
        
        # Threshold the EI to pick out only electrodes with large enough values
        good_inds = np.argwhere(np.abs(collapsed_ei) > thr_factor * NOISE).flatten()
        
        return good_inds, np.abs(collapsed_ei)

    if estim_type == 'single':
        if vstim_data.electrode_map.shape[0] == 519:
            if patterns == None:
                BAD_ELECS_519 = np.array([1, 130, 259, 260, 389, 390, 519], dtype=int)
                patterns = np.setdiff1d(np.arange(2, 519, dtype=int), BAD_ELECS_519)

            else:
                assert type(patterns) == np.ndarray, "User-input patterns should be numpy.ndarray"

            stim_elecs = patterns.reshape(-1, 1)

        elif vstim_data.electrode_map.shape[0] == 512:
            if patterns == None:
                patterns = np.arange(1, 513, dtype=int)

            else:
                assert type(patterns) == np.ndarray, "User-input patterns should be numpy.ndarray"

            stim_elecs = patterns.reshape(-1, 1)

    elif estim_type == 'triplet':
        assert type(patterns) == np.ndarray, "User-input patterns should be numpy.ndarray"
        triplet_dicts = loadmat('triplet_adj.mat')
        if vstim_data.electrode_map.shape[0] == 519:
            BAD_ELECS_519 = np.array([1, 130, 259, 260, 389, 390, 519], dtype=int)
            stim_elecs = triplet_dicts['LITKE_519'][patterns-1] + 1
            bad_pattern_inds = np.where(np.any(np.isin(stim_elecs, BAD_ELECS_519), axis=1))[0]
            if len(bad_pattern_inds) > 0:
                raise ValueError(f'Triplet patterns {patterns[bad_pattern_inds]} contain one or more inactive electrodes.')

        elif vstim_data.electrode_map.shape[0] == 512:
            stim_elecs = triplet_dicts['LITKE_512'][patterns-1] + 1

    all_cell_types = [ct for ct in vstim_data.get_all_present_cell_types() if 'bad' not in ct and 'dup' not in ct]
    total_electrode_list, total_cell_to_electrode_list, mutual_cells, array_id = get_cell_info(all_cell_types, vstim_data, compartments, NOISE, mutual_threshold=MUTUAL_THRESHOLD)
    cell_data_dict = {}
    cells_to_gsort_dict = {}
    cells_to_gsort_dict = dict([(pattern, []) for pattern in patterns])
    
    for type_ in cell_types:
        print(f'Loading data for cell type {type_}')
        
        for cell in tqdm.tqdm(vstim_data.get_all_cells_similar_to_type(type_)):
            good_inds, _ = get_collapsed_ei_thr(cell, NOISE_THRESH)        
            relevant_patterns = []
            for i in range(len(stim_elecs)):
                if np.any(np.in1d(stim_elecs[i], good_inds + 1)):
                    relevant_patterns.append(patterns[i])
            relevant_patterns = np.array(relevant_patterns)
        
            if len(relevant_patterns)==0:
                continue
            
            electrode_list =  list(set([e for c in mutual_cells[cell] for e in total_cell_to_electrode_list[c]]))
            cell_to_electrode_list = {k:v for k,v in total_cell_to_electrode_list.items() if k in mutual_cells[cell]}
            if len(electrode_list) == 0:
                # print('No significant electrodes.')
                continue

            for i in range(len(relevant_patterns)):
                cells_to_gsort_dict[relevant_patterns[i]] += [cell]
            
            cell_data_dict[cell] = get_center_eis(cell, electrode_list, ap = (vstim_analysis_path[:-7], vstim_datarun.rsplit('/')[-1]), excluded_types = excluded_types, excluded_cells = list(duplicates), power_threshold=POWER_THRESHOLD, array_id = array_id, sample_len_left = TIME_LIMIT +WINDOW_BUFFER,sample_len_right = TIME_LIMIT+WINDOW_BUFFER)
    return cell_data_dict, cells_to_gsort_dict, mutual_cells, total_cell_to_electrode_list, END_TIME_LIMIT, START_TIME_LIMIT, CLUSTER_DELAY, NOISE
