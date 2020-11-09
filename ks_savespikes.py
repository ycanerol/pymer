"""
Utility script for saving the spikes array under the ks_sorted file after manual curation is done.


"""
import sys
import readks
import analysis_scripts as asc
import iofuncs as iof
from datetime import datetime
from miscfuncs import timediff

try:
    filename = sys.argv[1]
except IndexError:
    filename = input('Enter the path or experiment name of ks_sorted for which spikes will be extracted: ')

filename = iof.exp_dir_fixer(filename)
filename = asc.kilosorted_path(filename)

if not asc.iskilosorted(filename):
    raise ValueError('Experiment lacks ks_sorted folder. Is this a kilosorted folder?')


start = datetime.now()
print(f'Starting preprocessing for folder: {filename}')
readks.preprocess_ks(filename)
print(f'Preprocessing complete. {timediff(start)} was elapsed.')
