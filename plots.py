import pylab as plt
from tools import get_good_units, select_all_experiments
import numpy as np
from pathlib import Path
import spikeinterface.full as si

def plot_rates(recordings, quality_criteria = 'snr > 3 & isi_violations_ratio < 0.1', folder='plots'):
    path = Path(folder) / "rates"
    path.mkdir(parents=True, exist_ok=True)
    for key in recordings.keys():       
        fig, axes = plt.subplots(2, 2, figsize=(10,10))
        sa = recordings[key]['analyzer']
        if sa.get_extension('unit_locations') is None:
            sa.compute('unit_locations', method='monopolar_triangulation')
        cells_positions = sa.get_extension('unit_locations').get_data()

        good_unit_ids = get_good_units(recordings[key]['metrics'], quality_criteria)
        upper_limit = lambda x:recordings[key]['boundaries']['src'][1] + recordings[key]['boundaries']['src'][0]*x
        lower_limit = lambda x: recordings[key]['boundaries']['tgt'][1] + recordings[key]['boundaries']['tgt'][0]*x

        crossing_limit = (upper_limit(cells_positions[:,0]) + lower_limit(cells_positions[:,0]))/2
        
        rates = recordings[key]['metrics']['firing_rate'][good_unit_ids]
        unit_ids = sa.unit_ids
        is_pre = cells_positions[:,1] > crossing_limit
        pre_cells = unit_ids[np.where(is_pre)[0]]
        is_post = cells_positions[:,1] < crossing_limit
        post_cells = unit_ids[np.where(is_post)[0]]     

        mask_pre = np.isin(pre_cells, good_unit_ids)
        valid_pre_cells = pre_cells[mask_pre]
        mask_post = np.isin(post_cells, good_unit_ids)
        valid_post_cells = post_cells[mask_post]

        mask_cc_pre = np.isin(unit_ids, valid_pre_cells)
        mask_cc_post = np.isin(unit_ids, valid_post_cells)
        
        try:
            axes[0,0].violinplot(rates, [0], showmedians=True)
        except Exception:
            pass
        axes[0,0].set_ylabel('Rate [Hz]')
        data = [recordings[key]['metrics']['firing_rate'][mask_cc_pre],
                recordings[key]['metrics']['firing_rate'][mask_cc_post]]
        try:
            axes[0,1].violinplot(data, [0, 1], showmedians=True)
        except Exception:
            pass
        axes[0,1].set_xticks([0, 1], ['pre', 'post'])

        axes[1,0].bar([0],[len(good_unit_ids)])
        axes[1,0].set_xticks([0], ['all'])
        axes[1,0].set_ylabel('# neurons')
        axes[1,1].bar([0,1],[np.sum(mask_cc_pre), np.sum(mask_cc_post)])
        axes[1,1].set_xticks([0, 1], ['pre', 'post'])
        plt.savefig(path / f"{key}.png")
        plt.close()


def plot_positions(recordings, quality_criteria = 'snr > 3 & isi_violations_ratio < 0.1', folder='plots'):
    #recs = select_all_experiments(recordings, experiments)
    path = Path(folder) / "positions"
    path.mkdir(parents=True, exist_ok=True)
    for key in recordings.keys():        
        fig, ax = plt.subplots(1, 1, figsize=(15,10))
        good_unit_ids = get_good_units(recordings[key]['metrics'], quality_criteria)
        sa = recordings[key]['analyzer']
        if sa.get_extension('unit_locations') is None:
            sa.compute('unit_locations', method='monopolar_triangulation')
        si.plot_unit_locations(sa, plot_legend=False, ax=ax, unit_ids=good_unit_ids)
        upper_limit = lambda x:recordings[key]['boundaries']['src'][1] + recordings[key]['boundaries']['src'][0]*x
        lower_limit = lambda x: recordings[key]['boundaries']['tgt'][1] + recordings[key]['boundaries']['tgt'][0]*x

        xaxis = np.arange(-100, 500)
        ax.plot(xaxis, upper_limit(xaxis), 'b--')
        ax.plot(xaxis, lower_limit(xaxis), 'r--')
        crossing_limit = (upper_limit(xaxis) + lower_limit(xaxis))/2
        ax.plot(xaxis, crossing_limit, 'k--')
        plt.savefig(path / f"{key}.png")
        plt.close()
            

def plot_rasters(recordings, quality_criteria = 'snr > 3 & isi_violations_ratio < 0.1', folder='plots', time_range=None):
    #recs = select_all_experiments(recordings, experiments)
    path = Path(folder) / "rasters"
    path.mkdir(parents=True, exist_ok=True)
    for key in recordings.keys():        
        fig, axes = plt.subplots(1, 3, figsize=(15,5))
        good_unit_ids = get_good_units(recordings[key]['metrics'], quality_criteria)
        si.plot_rasters(recordings[key]['sorting'], ax=axes[0], unit_ids=good_unit_ids, time_range=time_range)
        axes[0].set_title('all')
        sa = recordings[key]['analyzer']
        if sa.get_extension('unit_locations') is None:
            sa.compute('unit_locations', method='monopolar_triangulation')
        cells_positions = sa.get_extension('unit_locations').get_data()
        upper_limit = lambda x:recordings[key]['boundaries']['src'][1] + recordings[key]['boundaries']['src'][0]*x
        lower_limit = lambda x: recordings[key]['boundaries']['tgt'][1] + recordings[key]['boundaries']['tgt'][0]*x

        crossing_limit = (upper_limit(cells_positions[:,0]) + lower_limit(cells_positions[:,0]))/2
        
        unit_ids = sa.unit_ids
        is_pre = cells_positions[:,1] > crossing_limit
        pre_cells = unit_ids[np.where(is_pre)[0]]
        is_post = cells_positions[:,1] < crossing_limit
        post_cells = unit_ids[np.where(is_post)[0]]     

        mask_pre = np.isin(pre_cells, good_unit_ids)
        valid_pre_cells = pre_cells[mask_pre]
        mask_post = np.isin(post_cells, good_unit_ids)
        valid_post_cells = post_cells[mask_post]

        si.plot_rasters(recordings[key]['sorting'], ax=axes[1], unit_ids=valid_pre_cells, color='C0', time_range=time_range)
        axes[1].set_title('pre')
        si.plot_rasters(recordings[key]['sorting'], ax=axes[2], unit_ids=valid_post_cells, color='C1', time_range=time_range)
        axes[2].set_title('post')
        plt.savefig(path / f"{key}.png")
        plt.close()

def plot_drives(recordings, quality_criteria = 'snr > 3 & isi_violations_ratio < 0.1', window_ms=20, bin_ms=5, folder='plots', ignore_zero=False):
    path = Path(folder) / "drives"
    path.mkdir(parents=True, exist_ok=True)
    for key in recordings.keys():       
        sa = recordings[key]['analyzer']
        if sa.get_extension('unit_locations') is None:
            sa.compute('unit_locations', method='monopolar_triangulation')
        cells_positions = sa.get_extension('unit_locations').get_data()
        if sa.get_extension('correlograms') is None:
            sa.compute('correlograms', bin_ms=bin_ms, window_ms=window_ms)
        correlograms = sa.get_extension('correlograms').get_data()

        good_unit_ids = get_good_units(recordings[key]['metrics'], quality_criteria)
        upper_limit = lambda x:recordings[key]['boundaries']['src'][1] + recordings[key]['boundaries']['src'][0]*x
        lower_limit = lambda x: recordings[key]['boundaries']['tgt'][1] + recordings[key]['boundaries']['tgt'][0]*x

        unit_ids = sa.unit_ids

        crossing_limit = (upper_limit(cells_positions[:,0]) + lower_limit(cells_positions[:,0]))/2
        
        is_pre = cells_positions[:,1] > crossing_limit
        pre_cells = unit_ids[np.where(is_pre)[0]]
        is_post = cells_positions[:,1] < crossing_limit
        post_cells = unit_ids[np.where(is_post)[0]]

        print(pre_cells, post_cells)
        center = correlograms[0].shape[2]//2

        mask_pre = np.isin(pre_cells, good_unit_ids)
        valid_pre_cells = pre_cells[mask_pre]
        mask_post = np.isin(post_cells, good_unit_ids)
        valid_post_cells = post_cells[mask_post]
            
        mask_cc_pre = np.isin(unit_ids, valid_pre_cells)
        mask_cc_post = np.isin(unit_ids, valid_post_cells)
                
        drive_before = correlograms[0][mask_cc_pre][:, mask_cc_post][:,:,:center-1].mean(axis=2)
        drive_after = correlograms[0][mask_cc_pre][:, mask_cc_post][:,:,center+1:].mean(axis=2)
    
        data = (drive_before/drive_after)
        data[np.isinf(data)] = np.nan
        data[np.isnan(data)] = np.nan
        print(data)
        d = {}
        import pandas as pd
        for count, i in enumerate(unit_ids[mask_cc_post]):
            d[i] = data[:, count]
        df = pd.DataFrame(data=d, index=unit_ids[mask_cc_pre])
        df.to_csv(path / f"{key}.csv")
        
        data = data.flatten()
        data = data[~np.isinf(data)]
        data = data[~np.isnan(data)]
        if ignore_zero:
            data = data[~(data == 0)]

        fig, axes = plt.subplots(1, 1, figsize=(5,5))
        xmin, xmax = axes.get_xlim()
        axes.plot([-0.5, 0.5], [1, 1], 'k--')
        try:
            axes.violinplot(data, [0], showmedians=True)
        except Exception:
            pass
        plt.savefig(path / f"{key}.png")
        plt.close()

