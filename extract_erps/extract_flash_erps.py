from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytz
import tqdm
import os

# goes through column D13 (where we have the trial start pulses) starting at row=start_row and returns teh index of the next 1 or 0.5
def find_start_row(eeg_df, start_row):
    for i, row in eeg_df[start_row:].iterrows():
        if (row[23] == 1 or row[23] == 0.5):
            return i

# returns the rows where each trial starts
def read_file(eeg_file):

    eeg_df = pd.read_csv(eeg_file, sep = '\t', header=None)

    flash_indices = []

    # change 1250 to whenever your session actually begins. 1250 is just 10 seconds.
    i = 1250
    while i < len(eeg_df.index):
        i = find_start_row(eeg_df, i)
        if (i != None):
            flash_indices.append(i)
            i += 100
        else:
            i = eeg_df.index

    return eeg_df, flash_indices

def save_trials(trials, file_path):
    direcotry_path = os.path.dirname(file_path)
    os.makedirs(direcotry_path, exist_ok=True)
    np.save(file_path, trials)

### Helper function to slice windows from row indices corresponding to stimulus presentation in the EEG DataFrame ###
def get_trials(eeg_df, flash_indices):

    flash_trials = []

    for index in flash_indices:
        data = eeg_df.iloc[index:(index + 250), 1:17]
        data = np.array(data).T
        # change 250 to be iti*125
        if (data.shape == (16, 250)):
            flash_trials.append(data)

    flash_trials = np.concatenate(np.expand_dims(flash_trials, 0))
    return flash_trials


### Given a path to a single EEG file and a single aligned metadata_file along with before and after times to extract ERP windows, extract ERPs for congruent and incongruent trials ###
### Return two numpy arrays each in the shape '(num_erps, num_channel, num_timesteps)' ###
def extract(data_file):
    eeg_df, flash_indices = read_file(data_file)
    flash_trials = get_trials(eeg_df, flash_indices)
    return flash_trials


### Given a list of paths to EEG data files and an aligned list of paths to the corresponding metadata_paths, extracts ERPs in the shape '(num_erps, num_channel, num_timesteps)' ###
### ERP windows are are extracted from onset_time seconds before the stimulus was presented until after_time seconds after the stimulus was presented ###
### Extracted ERPs for the congruent trials and incongruent trials are saved separately as .npy files to a new directory called outputs for each EEG file ###
### The name of the new file will the original filename preceded by 'congruent_trials' or 'incongruent_trials' ###
def process_data(data_paths:list, metadata_paths:list, output_dir_name = "outputs"):
    for index, (eeg_file, metadata_paths) in enumerate(tqdm.tqdm(zip(data_paths, metadata_paths), total=len(data_paths))):
        flash_trials = extract(eeg_file)
        eeg_file = os.path.normpath(eeg_file)
        f_name = eeg_file.split(os.sep)[-1].split(".")[0]
        flash_file_path = os.path.join(output_dir_name, f"flash_trials_{f_name}.npy")
        save_trials(flash_trials, flash_file_path)
