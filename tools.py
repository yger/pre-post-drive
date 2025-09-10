# import csv
# file = open('mapping.csv')
# csvreader = csv.reader(file)
# rows = []
# positions = {}
# channel_offset = 2
# for row in csvreader:
#     channel = int(row[0].split('=')[1].split(')')[0])
#     positions[channel - channel_offset] = [int(row[1]), int(row[2])]

import probeinterface
import os
import spikeinterface.full as si
import pandas as pd
import probeinterface
import h5py

import numpy as np
def create_prb_file(channels, filename='my_mea.prb'):
    
    mea_probe = {}
    with open('mea_256.prb', 'r') as f:
        probetext = f.read()
        exec(probetext, mea_probe)
    del mea_probe['__builtins__']
    
    new_probe = {}
    new_probe['channel_groups'] = {1 : {}}
    new_probe['channel_groups'][1]["channels"] = np.arange(len(channels))
    new_probe['channel_groups'][1]["geometry"] = {}
    for count, i in enumerate(channels):
        new_probe['channel_groups'][1]["geometry"][count] = mea_probe['channel_groups'][1]["geometry"][i]
    f = open(filename, 'w')
    f.write('channel_groups    = {1 : {}}\n')
    f.write(f'channel_groups[1]["channels"] = {list(np.arange(len(channels)))}\n')
    positions = new_probe['channel_groups'][1]["geometry"]
    f.write(f'channel_groups[1]["geometry"] = {positions}')
    f.close()   

def load_experiment(file, base_folder, remove_center=False):
    key, ext = os.path.splitext(file)
    print(base_folder / file)
    rec = si.read_mcsh5(base_folder / file, 0) 
    channels = h5py.File(base_folder / file)['Data/Recording_0/AnalogStream/Stream_0/InfoChannel']['ChannelID']
    create_prb_file(channels - 2)
    electrodes = pd.read_csv(base_folder / (key + '.csv'))
    probe = probeinterface.read_prb('my_mea.prb')
    recording = rec.set_probegroup(probe.probes[0])
    
    if remove_center:
        boundaries = infer_boundaries(electrodes)
        upper_limit = lambda x: boundaries['src'][1] + boundaries['src'][0]*x
        lower_limit = lambda x: boundaries['tgt'][1] + boundaries['tgt'][0]*x
        positions = recording.get_channel_locations()
        valid_cells = np.logical_or(positions[:,1] > upper_limit(positions[:,0]), positions[:,1] < lower_limit(positions[:,0]))
        channel_ids = recording.channel_ids
        recording = recording.channel_slice(channel_ids[valid_cells])
    
    return {'raw' : recording, 'mapping' : electrodes}
    
def get_positions(recording):
    upper_limit = lambda x: recording['boundaries']['src'][1] + recording['boundaries']['src'][0]*x
    lower_limit = lambda x: recording['boundaries']['tgt'][1] + recording['boundaries']['tgt'][0]*x
    sa = recording['analyzer']
    if sa.get_extension('unit_locations') is None:
        sa.compute('unit_locations', method='monopolar_triangulation')
    
    cells_positions = sa.get_extension('unit_locations').get_data()

    is_pre = cells_positions[:,1] > upper_limit(cells_positions[:,0])
    is_post = cells_positions[:,1] < lower_limit(cells_positions[:,0])
    output = np.zeros(len(cells_positions), dtype='int')
    output[is_pre] = 1
    output[is_post] = 2
    res = []
    for i in output:
        if i == 0:
            res += ['center']
        elif i == 1:
            res += ['pre']
        elif i == 2:
            res += ['post']
    return res, cells_positions[:,0], cells_positions[:,1]

def infer_boundaries(mapping):
    results = {}
    for i, key in enumerate(['src', 'tgt']):
        mask = mapping['designation'] == i + 1
        valid_x = mapping['x'][mask].values
        valid_y = mapping['y'][mask].values
        x_min = valid_x[np.argmin(valid_x)]
        y_min = np.max(valid_y[valid_x == x_min])
        x_max = valid_x[np.argmax(valid_x)]
        y_max = np.min(valid_y[valid_x == x_max])
        a = (y_max - y_min)/(x_max - x_min)
        b = y_min - a*x_min
        results[key] = (a, b)
    return results

def filter_experiment(recordings, patterns):
    res = []
    for key in recordings.keys():
        is_in = True
        for pattern in patterns:
            is_in *= (key.find(pattern) > -1)
        if is_in:
            res += [key]
    return res

def select_all_experiments(recordings, experiments):
    experiments = experiments.copy()
    if experiments['condition'] is None:
        experiments['condition'] = ['all']
    if experiments['div'] is None:
        experiments['div'] = ['all']
    if experiments['type'] is None:
        experiments['type'] = ['all']
    result = {}
    for condition in experiments['condition']:
        for div in experiments['div']:
            for type in experiments['type']:
                recs = filter_experiment(recordings, [condition, div, type])
                if len(recs) > 0:
                    result[(condition, div, type)] = recs
    return result

def get_good_units(metric, quality_criteria = 'snr > 3 & isi_violations_ratio < 0.1'):
   return metric.query(quality_criteria).index

    