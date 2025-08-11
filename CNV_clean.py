# region setting up the environment
from epoch_rejection import epoch_rejection
import mne
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from collections import Counter


# endregion setting up the environment
######################################################################################
######################################################################################
## Set up data ##

CNV = mne.io.read_raw_bdf('P16/P16_CNV.bdf', preload=True)

Ch_to_pick = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
                 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19',
                   'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28',
                     'A29', 'A30', 'A31', 'A32', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6',
                       'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16',
                         'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26',
                           'B27', 'B28', 'B29', 'B30', 'B31', 'B32', 'Status']


mapping = {
    'A1': 'Fp1', 'A2': 'AF7', 'A3': 'AF3', 'A4': 'F1', 'A5': 'F3', 'A6': 'F5',
    'A7': 'F7', 'A8': 'FT7', 'A9': 'FC5', 'A10': 'FC3', 'A11': 'FC1', 'A12': 'C1',
    'A13': 'C3', 'A14': 'C5', 'A15': 'T7', 'A16': 'TP7', 'A17': 'CP5', 'A18': 'CP3',
    'A19': 'CP1', 'A20': 'P1', 'A21': 'P3', 'A22': 'P5', 'A23': 'P7', 'A24': 'P9',
    'A25': 'PO7', 'A26': 'PO3', 'A27': 'O1', 'A28': 'Iz', 'A29': 'Oz', 'A30': 'POz',
    'A31': 'Pz', 'A32': 'CPz', 'B1': 'Fpz', 'B2': 'Fp2', 'B3': 'AF8', 'B4': 'AF4', 'B5': 'AFz',
    'B6': 'Fz', 'B7': 'F2', 'B8': 'F4', 'B9': 'F6', 'B10': 'F8', 'B11': 'FT8',
    'B12': 'FC6', 'B13': 'FC4', 'B14': 'FC2', 'B15': 'FCz', 'B16': 'Cz', 'B17': 'C2', 'B18': 'C4',
    'B19': 'C6', 'B20': 'T8', 'B21': 'TP8', 'B22': 'CP6', 'B23': 'CP4', 'B24': 'CP2', 'B25': 'P2',
    'B26': 'P4', 'B27': 'P6', 'B28': 'P8', 'B29': 'P10', 'B30': 'PO8', 'B31': 'PO4', 'B32': 'O2', 
    'Status': 'Stim'
}

CNV = CNV.pick(Ch_to_pick)
CNV.rename_channels(mapping)
biosemi_montage = mne.channels.make_standard_montage("biosemi64")
CNV.set_montage(biosemi_montage)
# endregion loading data

######################################################################################
# region raw data & bads  
#CNV.plot()
#CNV.info['sfreq']
CNV.info['bads'].extend([]) # bad channels should be listed here 
CNV.interpolate_bads()
######################################################################################
## Cleaning:  ##

CNV.set_eeg_reference(ref_channels='average', projection=True).apply_proj()

Notch_freqs = (60, 120, 180, 240, 300, 360)
CNV.notch_filter(Notch_freqs)

# endregion preprocessing 
######################################################################################
# region ICAs
CNV_filter_ICA = CNV.copy().crop(tmin = 50, tmax = 100).filter(l_freq=1, h_freq=None)
CNV_ICA = mne.preprocessing.ICA(n_components=20, random_state=97).fit(CNV_filter_ICA)

#CNV_ICA.plot_sources(CNV)
#CNV_ICA.plot_components(nrows = 5, ncols = 4)
#CNV_ICA.plot_properties(CNV, picks = [3, 7, 9, 11])
#CNV_ICA.plot_properties(CNV, picks = range(13))

CNV_ICA.exclude = [] # ICAs to be removed should be listed here

CNV_ICA.apply(CNV)
# endregion ICAs
#######################################
CNV.filter(l_freq=0.2, h_freq=40)
#CNV.plot()
######################################################################################
######################################################################################
# region finding events #
CNV_events = mne.find_events(CNV, stim_channel='Stim', initial_event=True, verbose=True, 
                              min_duration=0.002, shortest_event=0.002, consecutive=True, 
                              output='onset')

# The events were labelled using triggers.
# However, using the auditory cue onset using the cedrus trigger improves trigger's temporal accuracy 

# setting events to cedrus triggers 
Condition = None
One_row = None
One_96 = []

for row in CNV_events:
    if row[2] == 256:
        Condition = row
    elif row[2] == 96 and row[1] == 0:
        if Condition is not None:
            One_row = row
            One_96.append(One_row)
            One_row = None
            Condition = None          

One_96 = np.array(One_96)

Condition = None
Two_row = None
Two_96 = []

for row in CNV_events:
    if row[2] == 512:
        Condition = row
    elif row[2] == 96 and row[1] == 0:
        if Condition is not None:
            Two_row = row
            Two_96.append(Two_row)
            Two_row = None
            Condition = None

Two_96 = np.array(Two_96)

One_trigs = CNV_events[CNV_events[:, 2] == 256]
Two_trigs = CNV_events[CNV_events[:, 2] == 512]

All_events = np.concatenate((One_96, Two_96, One_trigs, Two_trigs), axis=0)
#mne.viz.plot_events(Two_cedrus, sfreq=CNV.info['sfreq'], first_samp=CNV.first_samp)
# endregion finding events
######################################################################################
# region Epoching
One_rejection_criteria = dict(eeg=42.5e-6)
Two_rejection_criteria = dict(eeg=49e-6)

One_dropped_len_min = (0.15 * (len(One_96))) - 1
One_dropped_len_max = (0.15 * (len(One_96))) + 2

Two_dropped_len_min = (0.15 * (len(Two_96))) -1
Two_dropped_len_max = (0.15 * (len(Two_96))) + 2


One_epochs, One_rejection_criteria = epoch_rejection(
    CNV, One_96, -0.1, 2,
    One_dropped_len_min, One_dropped_len_max,
    One_rejection_criteria)

Two_epochs, Two_raw_rejection_criteria = epoch_rejection(
    CNV, Two_96, -0.1, 2,
    Two_dropped_len_min, Two_dropped_len_max,
    Two_rejection_criteria)



#One_epochs.plot_drop_log()
#Two_epochs.plot_drop_log()

#One_epochs.drop([])
#Two_epochs.drop([])

######################################################################################
# averaging the epochs
CNV_One = One_epochs['96'].average()
CNV_Two = Two_epochs['96'].average()
# endregion Epoching
######################################################################################
# region Visualization
#mne.viz.plot_events(All_events, sfreq=CNV.info['sfreq'], first_samp=CNV.first_samp)

CNV_dict = {
    'One': CNV_One,
    'Two': CNV_Two
}

mne.viz.plot_compare_evokeds(CNV_dict, picks = 'Cz')
# endregion Visualization
######################################################################################
# region exporting the data 
# Exporting the epochs

# Epochs can be save to file for later import and manupulation using the following line: 

#One_epochs.save('Epochs/CNV/P16_CNV_One-epo.fif', overwrite=True)
#Two_epochs.save('Epochs/CNV/P16_CNV_Two-epo.fif', overwrite=True)