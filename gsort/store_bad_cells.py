import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1


# added by raman
import sys

if '/Volumes/Lab/Users/vilkhu/artificial-retina-software-pipeline/utilities/' in sys.path:
    sys.path.remove('/Volumes/Lab/Users/vilkhu/artificial-retina-software-pipeline/utilities/')
# end

from run_gsort_v2 import *
import argparse
from scipy.io import loadmat
from itertools import product
import tqdm
import logging


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, help='Dataset in format YYYY-MM-DD-P.')
parser.add_argument('-w', '--wnoise', type=str, help='White noise run in format dataXXX (or streamed/dataXXX or kilosort_dataXXX/dataXXX, etc.).')
parser.add_argument('-e', '--estim', type=str, help='Estim run in format dataXXX.')
parser.add_argument('-o', '--output', type=str, help='/path/to/output/directory/.')
parser.add_argument('-t', '--threads', type=int, help='Number of threads to use in computation.')
parser.add_argument('-v', '--verbose', help="increase output verbosity", action="store_true")
parser.add_argument('-i', '--interest', type=float, help="Noise threshold for cells of interest")
parser.add_argument('-a', '--all', help="Run gsort on all cell types.", action="store_true")
args = parser.parse_args()

dataset = args.dataset
vstim_datarun = args.wnoise
estim_datarun = args.estim
filepath = args.output
verbose = args.verbose
noise_thresh = args.interest
all_types = args.all



DEFAULT_THREADS = 24
if args.threads is not None:
    threads = args.threads
else:
    threads = DEFAULT_THREADS

    
    

vstim_analysis_path = '/Volumes/Analysis/'+ dataset + '/' + vstim_datarun

estim_analysis_path = '/Volumes/Analysis/'+ dataset + '/' + estim_datarun +'/'
pattern_path = estim_analysis_path + 'pattern_files/'

vstim_data = vl.load_vision_data(vstim_analysis_path,
                                 vstim_datarun.rsplit('/')[-1],
                                 include_params=True,
                                 include_ei=True,
                                 include_noise=True,
                                 include_neurons=True)

if all_types:
    all_cell_types = vstim_data.get_all_present_cell_types()
    
    cell_types = []
    for type_ in all_cell_types:
        if 'dup' in type_ or 'bad' in type_:
            continue
        else:
            cell_types.append(type_)

MIN_CORR = .975
duplicates = set()
cellids = vstim_data.get_cell_ids()
remove_set = set()
for cell in cellids:
    cell_ei = vstim_data.get_ei_for_cell(cell).ei
    cell_ei_error = vstim_data.get_ei_for_cell(cell).ei_error
    cell_ei_max = np.abs(np.amin(cell_ei,axis=1))
    cell_ei_power = np.sum(cell_ei**2,axis=1)
    celltype = vstim_data.get_cell_type_for_cell(cell).lower()
    if "dup" in celltype or "bad" in celltype:
        remove_set.add(cell)
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
                
remove_set = remove_set.union(duplicates)

with open(filepath + dataset +  "/excluded_neurons.txt", "a") as f:
    for cell in remove_set:
        f.write(str(cell) + "\n")


                
