# region setting up wd 
from epoch_rejection import epoch_rejection
import mne
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from collections import Counter



######################################################################################
######################################################################################
# region Import data
Stim = mne.io.read_raw_bdf('P_n/Pn_Stim.bdf', preload=True)
Control = mne.io.read_raw_bdf('P_n/Pn_Standing.bdf', preload=True)

Stim.info['sfreq']
Control.info['sfreq']

# endregion Import data
######################################################################################
# region Montage
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


Stim = Stim.pick(Ch_to_pick)
Control = Control.pick(Ch_to_pick)

Stim.rename_channels(mapping)
Control.rename_channels(mapping)

biosemi_montage = mne.channels.make_standard_montage("biosemi64")

Stim.set_montage(biosemi_montage)
Control.set_montage(biosemi_montage)
# endregion Montage
######################################################################################
######################################################################################
# region Preprocessing
#Stim.plot()
#Control.plot()

#Stim.info['bads'].extend(['Bad channels to be removed should be listed here'])
#Control.info['bads'].extend(['CP6', 'PO3']) #like this

## Refrence ##
Stim.set_eeg_reference(ref_channels='average', projection=True).apply_proj()
Control.set_eeg_reference(ref_channels='average', projection=True).apply_proj()

#Stim.plot(picks = ['Cz', 'C2', 'C1', 'FCz', 'CPz', "FC2", 'FC1', 'CP2', 'CP1'])
#Control.plot()
#######################################
# Notch filter: 
#Stim.compute_psd(fmax = 400).plot(average=True, amplitude = False, exclude = 'bads')
Notch_freqs = (60, 120, 180, 240, 300, 360)
Stim.notch_filter(freqs = Notch_freqs)
Control.notch_filter(freqs = Notch_freqs)
#endregion Preprocessing
######################################################################################
# region ICAs
Stim_filter_ICA = Stim.copy().crop(tmin = 50, tmax = 100).filter(l_freq=1, h_freq=None)
Control_filter_ICA = Control.copy().crop(tmin = 50, tmax = 100).filter(l_freq=1, h_freq=None)

#######################################
Stim_ICA = mne.preprocessing.ICA(n_components=20, random_state=97).fit(Stim_filter_ICA)

#Stim_ICA.plot_sources(Stim)
#Stim_ICA.plot_components(nrows = 5, ncols = 4)
#Stim_ICA.plot_properties(Stim, picks = [15])
#Stim_ICA.plot_properties(Stim, picks = list(range(15)))

#Stim_ICA.get_explained_variance_ratio(Stim_filter_ICA, components = [1], ch_type = 'eeg')

Stim_ICA.exclude = [] #the number of the ICAs to be removed should be listed here
Stim_ICA.apply(Stim)
#######################################

Control_ICA = mne.preprocessing.ICA(n_components=20, random_state=97).fit(Control_filter_ICA)

#Control_ICA.plot_sources(Control)
#Control_ICA.plot_components(nrows = 5, ncols = 4)

#Control_ICA.plot_properties(Control, picks = list(range(12)))
#Control_ICA.plot_properties(Control, picks = [7, 14])

#Control_ICA.get_explained_variance_ratio(Stim_filter_ICA, components = [3], ch_type = 'eeg')
#Control_ICA.get_explained_variance_ratio(Stim_filter_ICA, components = [8], ch_type = 'eeg')
#Control_ICA.get_explained_variance_ratio(Stim_filter_ICA, components = [12], ch_type = 'eeg')

#Control_ICA.exclude = [0, 3, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] #like this
Control_ICA.apply(Control)
# endregion ICAs
######################################################################################
# Filtering: #
# region Filtering
Stim.filter(l_freq=0.1, h_freq=40, n_jobs = 18)
Control.filter(l_freq=0.1, h_freq=40, n_jobs = 18)
# endregion Filtering
######################################################################################
######################################################################################
## Finding events ##
# region Finding events
Stim_events = mne.find_events(Stim, stim_channel='Stim', initial_event=True, verbose=True, 
                              min_duration=0.002, shortest_event=0.002, consecutive=True, 
                              output='onset')

Control_events = mne.find_events(Control, stim_channel='Stim', initial_event=True, verbose=True, 
                              min_duration=0.002, shortest_event=0.002, consecutive=True, 
                              output='onset')

#mne.viz.plot_events(Stim_events, sfreq=Stim.info['sfreq'], first_samp=Stim.first_samp)
#mne.viz.plot_events(Control_events, sfreq=Control.info['sfreq'], first_samp=Control.first_samp)
# endregion Finding events
######################################################################################
######################################################################################
# region Parsing e-stims
# We wanted to also analyze the triggers based on the order in which they were presented.
# This region sorts these triggers based on the order they were presented

# After data collection, we realized the electrical stimulation was being outputted slighlty later than its triggers.
# the following section detects these triggers and repairs them.  

Condition = None
Stim_count = 0

One_1st_stim_row = None
One_2nd_stim_row = None
One_3rd_stim_row = None
One_4th_stim_row = None

One_1st_stim = []
One_2nd_stim = []
One_3rd_stim = []
One_4th_stim = []

#################################################

Two_1st_stim_row = None
Two_2nd_stim_row = None
Two_3rd_stim_row = None
Two_4th_stim_row = None

Two_1st_stim = []
Two_2nd_stim = []
Two_3rd_stim = []
Two_4th_stim = []

#################################################

Three_1st_stim_row = None
Three_2nd_stim_row = None
Three_3rd_stim_row = None
Three_4th_stim_row = None

Three_1st_stim = []
Three_2nd_stim = []
Three_3rd_stim = []
Three_4th_stim = []

######################################################################################

for row in Stim_events:
    if row[2] == 256:
        Stim_count = 0
        Condition = "One"

    if row[2] == 512:
        Stim_count = 0
        Condition = "Two"

    if row[2] == 1024 and row[1] == 0:
        Stim_count += 1

    if Stim_count == 1:
        if Condition == "One":
            One_1st_stim_row = row
            One_1st_stim.append(One_1st_stim_row)
            One_1st_stim_row = None

        if Condition == "Two":
            Two_1st_stim_row = row
            Two_1st_stim.append(Two_1st_stim_row)
            Two_1st_stim_row = None

    if Stim_count == 2:
        if Condition == "One":
            One_2nd_stim_row = row
            One_2nd_stim.append(One_2nd_stim_row)
            One_2nd_stim_row = None

        if Condition == "Two":
            Two_2nd_stim_row = row
            Two_2nd_stim.append(Two_2nd_stim_row)
            Two_2nd_stim_row = None

    if Stim_count == 3:
        if Condition == "One":
            One_3rd_stim_row = row
            One_3rd_stim.append(One_3rd_stim_row)
            One_3rd_stim_row = None

        if Condition == "Two":
            Two_3rd_stim_row = row
            Two_3rd_stim.append(Two_3rd_stim_row)
            Two_3rd_stim_row = None

    if Stim_count == 4:
        if Condition == "One":
            One_4th_stim_row = row
            One_4th_stim.append(One_4th_stim_row)
            One_4th_stim_row = None
            Condition = None

        if Condition == "Two":
            Two_4th_stim_row = row
            Two_4th_stim.append(Two_4th_stim_row)
            Two_4th_stim_row = None
            Condition = None


######################################################################################
# Creating arrays for events: 
One_1st_stim_arr = np.array(One_1st_stim)
One_2nd_stim_arr = np.array(One_2nd_stim)
One_3rd_stim_arr = np.array(One_3rd_stim)
One_4th_stim_arr = np.array(One_4th_stim)

Two_1st_stim_arr = np.array(Two_1st_stim)
Two_2nd_stim_arr = np.array(Two_2nd_stim)
Two_3rd_stim_arr = np.array(Two_3rd_stim)
Two_4th_stim_arr = np.array(Two_4th_stim)

One_all = np.concatenate((One_1st_stim_arr, One_2nd_stim_arr, One_3rd_stim_arr, One_4th_stim_arr))
Two_all = np.concatenate((Two_1st_stim_arr, Two_2nd_stim_arr, Two_3rd_stim_arr, Two_4th_stim_arr))

One_1st_half_stim_arr = np.concatenate((One_1st_stim_arr, One_2nd_stim_arr))
One_2nd_half_stim_arr = np.concatenate((One_3rd_stim_arr, One_4th_stim_arr))

Two_1st_half_stim_arr = np.concatenate((Two_1st_stim_arr, Two_2nd_stim_arr))
Two_2nd_half_stim_arr = np.concatenate((Two_3rd_stim_arr, Two_4th_stim_arr))


######################################################################################
# Coppying the arrays for correction: 
Control_events_corrrected = Control_events.copy()

One_events_corrrected = One_all.copy()
Two_events_corrected = Two_all.copy()

One_1st_corr = One_1st_stim_arr.copy()
One_2nd_corr = One_2nd_stim_arr.copy()
One_3rd_corr = One_3rd_stim_arr.copy()
One_4th_corr = One_4th_stim_arr.copy()

Two_1st_corr = Two_1st_stim_arr.copy()
Two_2nd_corr = Two_2nd_stim_arr.copy()
Two_3rd_corr = Two_3rd_stim_arr.copy()
Two_4th_corr = Two_4th_stim_arr.copy()


One_cue = Stim_events[Stim_events[:, 2] == 256]
One_cue_stim = np.concatenate((One_cue, One_all))
#mne.viz.plot_events(One_cue_stim, sfreq=Stim.info['sfreq'], first_samp=Stim.first_samp)
# endregion Parsing e-stims

######################################################################################
######################################################################################
# region Raw data epoching 
rejection_criteria = dict(eeg=29.75e-6)
Raw_One_rejection_criteria = dict(eeg=32e-6)
Raw_Two_rejection_criteria = dict(eeg=31.5e-6)

Control_dropped_len_min = (0.15 * len(Control_events)) - 3
Control_dropped_len_max = (0.15 * len(Control_events)) + 3

One_dropped_len_min = (0.15 * len(One_all)) - 2
One_dropped_len_max = (0.15 * len(One_all)) + 2

Two_dropped_len_min = (0.15 * len(Two_all)) - 2
Two_dropped_len_max = (0.15 * len(Two_all)) + 2



Control_epochs, Control_rejection_criteria = epoch_rejection(
    Control, Control_events, tmin=-0.05, tmax=0.5, 
    dropped_len_min = Control_dropped_len_min, dropped_len_max = Control_dropped_len_max,
    initial_rejection_criteria = rejection_criteria
)

One_epochs, One_raw_rejection_criteria = epoch_rejection(
    Stim, One_all, tmin=-0.05, tmax=0.5, 
    dropped_len_min = One_dropped_len_min, dropped_len_max = One_dropped_len_max,
    initial_rejection_criteria = Raw_One_rejection_criteria
)

Two_epochs, Two_raw_rejection_criteria = epoch_rejection(
    Stim, Two_all, tmin=-0.05, tmax=0.5, 
    dropped_len_min = Two_dropped_len_min, dropped_len_max = Two_dropped_len_max,
    initial_rejection_criteria = Raw_Two_rejection_criteria
)



###############################################
One_evoked = One_epochs['1024'].average()
Two_evoked = Two_epochs['1024'].average()
Control_evoked = Control_epochs['1024'].average()
###############################################
raw_evoked_dict = {
    'One': One_evoked,
    'Two': Two_evoked,
    'Control': Control_evoked
}

Raw_plot = mne.viz.plot_compare_evokeds(raw_evoked_dict, picks='Cz', invert_y = True, title = 'Raw', show = False)
plt.show()
# endregion Raw data epoching

######################################################################################
######################################################################################
######################################################################################
# region Trigger delta
Control_trig_diff = 267
One_trig_diff = 277
Two_trig_diff = 277
# endregion Trigger delta
######################################################################################
######################################################################################
######################################################################################
# region Moving the triggers
One_events_corrrected[:, 0] += One_trig_diff
Two_events_corrected[:, 0] += Two_trig_diff
Control_events_corrrected[:, 0] += Control_trig_diff

One_1st_corr[:, 0] += One_trig_diff
One_2nd_corr[:, 0] += One_trig_diff
One_3rd_corr[:, 0] += One_trig_diff
One_4th_corr[:, 0] += One_trig_diff

Two_1st_corr[:, 0] += Two_trig_diff
Two_2nd_corr[:, 0] += Two_trig_diff
Two_3rd_corr[:, 0] += Two_trig_diff
Two_4th_corr[:, 0] += Two_trig_diff
# endregion Moving the triggers

######################################################################################
######################################################################################
# region Corrected data epoching all
rejection_criteria_Control_corrected = dict(eeg=30e-6)
rejection_crietria_One_corrected = dict(eeg=31e-6)
rejection_criteria_Two_corrected = dict(eeg=32.5e-6)

Control_epochs_corrected, Control_corrected_rejection_criteria = epoch_rejection(
    Control, Control_events_corrrected, tmin=-0.05, tmax=0.5,
    dropped_len_min = Control_dropped_len_min, dropped_len_max = Control_dropped_len_max,
    initial_rejection_criteria = rejection_criteria_Control_corrected
)

One_epochs_corrected, One_corrected_rejection_criteria = epoch_rejection(
    Stim, One_events_corrrected, tmin=-0.05, tmax=0.5,
    dropped_len_min = One_dropped_len_min, dropped_len_max = One_dropped_len_max,
    initial_rejection_criteria = rejection_crietria_One_corrected
)

Two_epochs_corrected, One_corrected_rejection_criteria = epoch_rejection(
    Stim, Two_events_corrected, tmin=-0.05, tmax=0.5,
    dropped_len_min = Two_dropped_len_min, dropped_len_max = Two_dropped_len_max,
    initial_rejection_criteria = rejection_criteria_Two_corrected
)

One_evoked_corrected = One_epochs_corrected['1024'].average()
Two_evoked_corrected = Two_epochs_corrected['1024'].average()
Control_evoked_corrected = Control_epochs_corrected['1024'].average()


corrected_evoked_dict = {
    'One': One_evoked_corrected,
    'Two': Two_evoked_corrected,
    'Control': Control_evoked_corrected
}

corrected_plot = mne.viz.plot_compare_evokeds(corrected_evoked_dict, picks='Cz', invert_y=True, title = 'Corrected', show = False)
plt.show()

# endregion Corrected data epoching all
######################################################################################
# region ordered stim epoching
# 1st stim comparasion: 
One_1st_stim_epochs = mne.Epochs(Stim, events=One_1st_corr, tmin=-0.05, tmax=0.5,
                            reject = rejection_crietria_One_corrected, preload=True)

Two_1st_stim_epochs = mne.Epochs(Stim, events=Two_1st_corr, tmin=-0.05, tmax=0.5,
                            reject = rejection_criteria_Two_corrected, preload=True)

######################################################################################
# 2nd stim comparasion: 

One_2nd_stim_epochs = mne.Epochs(Stim, events=One_2nd_corr, tmin=-0.05, tmax=0.5,
                            reject = rejection_crietria_One_corrected, preload=True)

Two_2nd_stim_epochs = mne.Epochs(Stim, events=Two_2nd_corr, tmin=-0.05, tmax=0.5,
                            reject = rejection_criteria_Two_corrected, preload=True)

#########################################
# 3rd stim comparasion:

One_3rd_stim_epochs = mne.Epochs(Stim, events=One_3rd_corr, tmin=-0.05, tmax=0.5,
                            reject = rejection_crietria_One_corrected, preload=True)

Two_3rd_stim_epochs = mne.Epochs(Stim, events=Two_3rd_corr, tmin=-0.05, tmax=0.5,
                            reject = rejection_criteria_Two_corrected, preload=True)

#########################################
# 4th stim comparasion:

One_4th_stim_epochs = mne.Epochs(Stim, events=One_4th_corr, tmin=-0.05, tmax=0.5,
                            reject = rejection_crietria_One_corrected, preload=True)

Two_4th_stim_epochs = mne.Epochs(Stim, events=Two_4th_corr, tmin=-0.05, tmax=0.5,
                            reject = rejection_criteria_Two_corrected, preload=True)
######################################################################################
# One: 
One_1st_stim_evoked = One_1st_stim_epochs['1024'].average()
One_2nd_stim_evoked = One_2nd_stim_epochs['1024'].average()
One_3rd_stim_evoked = One_3rd_stim_epochs['1024'].average()
One_4th_stim_evoked = One_4th_stim_epochs['1024'].average()

# Two:
Two_1st_stim_evoked = Two_1st_stim_epochs['1024'].average()
Two_2nd_stim_evoked = Two_2nd_stim_epochs['1024'].average()
Two_3rd_stim_evoked = Two_3rd_stim_epochs['1024'].average()
Two_4th_stim_evoked = Two_4th_stim_epochs['1024'].average()
######################################################################################
First_stim_evoked_dict = {
    'One': One_1st_stim_evoked,
    'Two': Two_1st_stim_evoked
}

Second_stim_evoked_dict = {
    'One': One_2nd_stim_evoked,
    'Two': Two_2nd_stim_evoked
}

Third_stim_evoked_dict = {
    'One': One_3rd_stim_evoked,
    'Two': Two_3rd_stim_evoked
}

Fourth_stim_evoked_dict = {
    'One': One_4th_stim_evoked,
    'Two': Two_4th_stim_evoked
}

mne.viz.plot_compare_evokeds(First_stim_evoked_dict, picks='Cz', invert_y=True, title = '1st Stim', show = False)
mne.viz.plot_compare_evokeds(Second_stim_evoked_dict, picks='Cz', invert_y=True, title = '2nd Stim', show = False)
mne.viz.plot_compare_evokeds(Third_stim_evoked_dict, picks='Cz', invert_y=True, title = '3rd Stim', show = False)
mne.viz.plot_compare_evokeds(Fourth_stim_evoked_dict, picks='Cz', invert_y=True, title = '4th Stim', show = False)
#plt.show()
# endregion ordered stim epoching
######################################################################################
######################################################################################
# region Saving the data
# Exporting to csv:



######################################################################################
# Saving Epochs: 
#Epochs can be saved to disk for import and further processing using the code below: 

#Control_epochs_corrected.save('Epochs/Control/P_n_Control_epochs-epo.fif', overwrite=True)

#One_epochs_corrected.save('Epochs/All/P_n_One_all_epochs-epo.fif', overwrite=True)
#Two_epochs_corrected.save('Epochs/All/P_n_Two_all_epochs-epo.fif', overwrite=True)

#One_1st_stim_epochs.save('Epochs/Ordered/P_n_One_1st_epochs-epo.fif', overwrite=True)
#One_2nd_stim_epochs.save('Epochs/Ordered/P_n_One_2nd_epochs-epo.fif', overwrite=True)
#One_3rd_stim_epochs.save('Epochs/Ordered/P_n_One_3rd_epochs-epo.fif', overwrite=True)
#One_4th_stim_epochs.save('Epochs/Ordered/P_n_One_4th_epochs-epo.fif', overwrite=True)

#Two_1st_stim_epochs.save('Epochs/Ordered/P_n_Two_1st_epochs-epo.fif', overwrite=True)
#Two_2nd_stim_epochs.save('Epochs/Ordered/P_n_Two_2nd_epochs-epo.fif', overwrite=True)
#Two_3rd_stim_epochs.save('Epochs/Ordered/P_n_Two_3rd_epochs-epo.fif', overwrite=True)
#Two_4th_stim_epochs.save('Epochs/Ordered/P_n_Two_4th_epochs-epo.fif', overwrite=True)

# endregion Saving the data
