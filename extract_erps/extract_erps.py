from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytz
import tqdm
import os

def find_closest_row(time, eeg_df, last_index):
    last_difference = float('inf')

    for i, row in eeg_df[last_index:].iterrows():
        unix_timestamp = row[30]
        unix_dt = datetime.fromtimestamp(unix_timestamp)
        unix_dt = unix_dt.replace(tzinfo=timezone.utc)
        time = time.replace(tzinfo=timezone.utc)
        difference = abs((unix_dt - time).total_seconds())
        print(f"EEG timestamp: {unix_dt}")
        print(f"Trial timestamp: {time}")
        print(f"Difference in seconds: {difference}")
        if(unix_dt > time):
            print(f"Difference in seconds final: {last_difference}")
            return i - 1, last_difference
        
        if abs(last_difference-difference) > 0.2:
            last_difference = difference

### Helper function that returns row indices in the EEG DataFrame that correspond to the appearance of a Stroop word according to metadata ###
def read_file(eeg_file, metadata_file, onset_time = 0, after_time = 1.0, sr = 125):
    metadata_df = pd.read_csv(metadata_file).reset_index(drop=True)

    #start_time = metadata_df.iloc[0]['expStart']
    #start_time = datetime.strptime(start_time, "%Y-%m-%d %Hh%M.%S.%f %z")

    eeg_df = pd.read_csv(eeg_file, sep = '\t', header=None)

    congruent_row_indices = []
    incongruent_row_indices = []
    last_index = 0
    start_time = datetime.strptime(metadata_df.iloc[0]['Timestamp'], "%Y-%m-%d_%H:%M:%S:%f")
    start_row, difference = find_closest_row(start_time,eeg_df,last_index)
    start_row += int(difference*sr)
    last_row = start_row
    iti = metadata_df.iloc[0]['ITI']
    for index, row in metadata_df.iloc[1:].iterrows():
        #time = start_time + timedelta(seconds=row['trial.started'])
        #time = datetime.strptime(row['Timestamp'], "%Y-%m-%d_%H:%M:%S:%f")
        #row_index = find_closest_row(time, eeg_df, last_index)
        row_index = last_row + int((iti*sr))
        last_row = row_index
        iti = row['ITI']
        #last_index = row_index
        if(row_index < (onset_time * sr) or row_index + (after_time * sr) >= len(eeg_df)):
            continue
        if str(row['Word']) == str(row['Color']):
            congruent_row_indices.append(row_index)
        else:
            incongruent_row_indices.append(row_index)
    return metadata_df, eeg_df, congruent_row_indices, incongruent_row_indices

def save_trials(trials, file_path):
    direcotry_path = os.path.dirname(file_path)
    os.makedirs(direcotry_path, exist_ok=True)
    np.save(file_path, trials)

### Helper function to slice windows from row indices corresponding to stimulus presentation in the EEG DataFrame ###
def get_trials(eeg_df, congruent_row_indices, incongruent_row_indices, onset_time = 0, after_time = 1.0, sr = 125):
    before_points = int(onset_time * sr)
    after_points = int(after_time * sr)

    congruent_trials = []
    incongruent_trials = []

    for index in congruent_row_indices:
        data = eeg_df.iloc[(index - before_points):(index + after_points), 1:17]
        data = np.array(data).T
        congruent_trials.append(data)

    for index in incongruent_row_indices:
        data = eeg_df.iloc[(index - before_points):(index + after_points), 1:17]
        data = np.array(data).T
        incongruent_trials.append(data)

    congruent_trials = np.concatenate(np.expand_dims(congruent_trials, 0))
    incongruent_trials = np.concatenate(np.expand_dims(incongruent_trials, 0))
    return congruent_trials, incongruent_trials

### Given a path to a single EEG file and a single aligned metadata_file along with before and after times to extract ERP windows, extract ERPs for congruent and incongruent trials ###
### Return two numpy arrays each in the shape '(num_erps, num_channel, num_timesteps)' ###
def extract(data_file, metadata_file, onset_time = 0, after_time = 0, sr = 125):
    metadata_df, eeg_df, congruent_row_indices, incongruent_row_indices = read_file(data_file, metadata_file)
    congruent_trials, incongruent_trials = get_trials(eeg_df, congruent_row_indices, incongruent_row_indices, onset_time=onset_time, after_time=after_time,sr=sr)
    return congruent_trials, incongruent_trials


### Given a list of paths to EEG data files and an aligned list of paths to the corresponding metadata_paths, extracts ERPs in the shape '(num_erps, num_channel, num_timesteps)' ###
### ERP windows are are extracted from onset_time seconds before the stimulus was presented until after_time seconds after the stimulus was presented ###
### Extracted ERPs for the congruent trials and incongruent trials are saved separately as .npy files to a new directory called outputs for each EEG file ###
### The name of the new file will the original filename preceded by 'congruent_trials' or 'incongruent_trials' ###
def process_data(data_paths:list, metadata_paths:list, output_dir_name = "outputs",onset_time = 0, after_time = 1.0, sr = 125):
    for index, (eeg_file, metadata_file) in enumerate(tqdm.tqdm(zip(data_paths, metadata_paths), total=len(data_paths))):
        congruent_trials, incongruent_trials = extract(eeg_file, metadata_file, onset_time=onset_time, after_time=after_time, sr=sr) 
        eeg_file = os.path.normpath(eeg_file)
        f_name = eeg_file.split(os.sep)[-1].split(".")[0]
        congruent_file_path = os.path.join(output_dir_name, f"congruent_trials_{f_name}.npy")
        incongruent_file_path = os.path.join(output_dir_name, f"incongruent_trials_{f_name}.npy")
        save_trials(congruent_trials, congruent_file_path)
        save_trials(incongruent_trials, incongruent_file_path)