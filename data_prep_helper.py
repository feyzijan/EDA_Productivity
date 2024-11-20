import numpy as np
import pandas as pd
import neurokit2 as nk
import datetime
import pickle
import os
import sys
import datetime
import os
import json
import copy

from scipy.stats import linregress
from scipy.fft import fft
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from eda_helper import get_eda_features

# Parameters
eda_freq = 4
upsampled_freq = 4 
window_length = 4 # 4 seconds window

butterpass_cutoff = 1
butterpass_order = 3
hr_freq = 1
acc_freq = 32
temp_freq = 4
bvp_freq = 64

data_types = ["EDA", "HR", "ACC", "TEMP", "BVP", "tags"]

# Path variables and participant list
all_a3_participants = ['P3','P3_2','P5','P8','P8_2','P9','P9_2', 'P11','P12','P13','P15','P16','P19','P20', 'P21','P24','P25',
           'P26','P28','P29','P30','P34','P34_2','P40','P42', 'P43','P43_2','P46','P47']

all_a4_participants = ["P9", "P11", "P12", "P15","P16", "P19", "P20", "P26", "P29", "P30",
             "P34", "P34_2", "P36", "P42", "P43", "P46","P50", "P52", "P53"]

# Exclude participants with poor data (seemingly incorrectly warn watch, very flat response etc)
a3_participants_to_remove_eda = ['P16', 'P36', 'P37','P40', 'P50']
a4_participants_to_remove_eda = ['P3', 'P8', 'P22']

local_directory_path = "/Users/feyzjan/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/GatechCourses/CS 8903 Research"
a3_path = f"{local_directory_path}/A3_DataFiles"
a4_path = f"{local_directory_path}/A4_DataFiles"

p_list_a3_eda = [x for x in all_a3_participants if x not in a3_participants_to_remove_eda]
p_list_a4_eda = [x for x in all_a4_participants if x not in a4_participants_to_remove_eda]


a3_participants_to_remove_keylogs = ['P15', 'P25', 'P26', 'P34_2', 'P46'] + ['P3_2','P8_2','P9_2','P43_2']
a4_participants_to_remove_keylogs = ['P12', 'P26', 'P46'] + ["P34_2"]
p_list_a3_log_analysis =[x for x in p_list_a3_eda if x not in a3_participants_to_remove_keylogs]
p_list_a4_log_analysis =[x for x in p_list_a4_eda if x not in a4_participants_to_remove_keylogs]

"""
Read in the EDA data from folders named in the p_list, and return a dictionary with the dataframes

Returns:
    dict: key1: participant ID[eg. 'P10']  key2: DataType (eg.EDA) value: DataFrame
"""
def get_empatica_data(a3,keylogger=False) -> dict:
    if a3 == True:
        p_list = p_list_a3_eda if keylogger == False else p_list_a3_log_analysis
        path = a3_path
    else:
        p_list = p_list_a4_eda if keylogger == False else p_list_a4_log_analysis
        path = a4_path
    data = {}

    for p in p_list:
        data[p] = {}

        # EDA
        df_eda = pd.read_csv(os.path.join(path, p, "Empatica", "EDA.csv")).iloc[1:]
        start_time = int(df_eda.columns[0].split(".")[0])
        df_eda.columns = ['EDA']
        df_eda["Time"] = start_time + df_eda.index/eda_freq
        df_eda["Time"] = pd.to_datetime(df_eda["Time"], unit='s')

        # Read in tags (if any) - These are the markers created when the participant presses the Empatica
        dummy_df = pd.DataFrame({'Time': [0], 'Value': [0]})

        try:
            df_tags = pd.read_csv(os.path.join(path, p, "Empatica", "tags.csv"), header=None)
            if df_tags.empty:
                df_tags = dummy_df
            elif len(df_tags) < 2:
                df_tags = dummy_df
            else:
                df_tags.columns = ['Time']
                df_tags["Time"] = pd.to_datetime(df_tags["Time"], unit='s')
        except pd.errors.EmptyDataError:
            df_tags = dummy_df
        except FileNotFoundError:
            df_tags = dummy_df

        data[p]["EDA"] = df_eda
        data[p]["Tags"] = df_tags

    return data


'''
'''

def get_keylog_data(a3, keylogger=False) -> dict:
    if a3 == True:
        p_list = p_list_a3_eda if keylogger == False else p_list_a3_log_analysis
        path = a3_path
    else:
        p_list = p_list_a4_eda if keylogger == False else p_list_a4_log_analysis
        path = a4_path

    data_dict = {}
    

    print("Getting keylog data for a3 =",a3)

    for p in p_list:
        log_path = os.path.join(path, p, "extra_credit.log")
        print("\n Log path for ", p, " is ", log_path)

        # load data
        with open(log_path) as f:
            data = f.readlines()

        parsed_data = [json.loads(line) for line in data]
        df = pd.DataFrame(parsed_data)

        # drop useless col
        df.drop(columns=['type'], inplace=True)

        # time conversion
        df['Time'] = pd.to_datetime(df['t']).dt.tz_localize(None) # remove timezone for operation later
        df.drop(columns=['t'], inplace=True)
        df['T_s'] = (df['Time'] - df['Time'].min()).dt.total_seconds() # time since start

        # parse out content change values
        df['start_line'] = df['contentChanges'].apply(lambda x: x[0]['range'][0]['line'])
        df['start_character'] = df['contentChanges'].apply(lambda x: x[0]['range'][0]['character'])
        df['end_line'] = df['contentChanges'].apply(lambda x: x[0]['range'][1]['line'])
        df['end_character'] = df['contentChanges'].apply(lambda x: x[0]['range'][1]['character'])

        df['range_offset'] = df['contentChanges'].apply(lambda x: x[0]['rangeOffset'])
        df['range_length'] = df['contentChanges'].apply(lambda x: x[0]['rangeLength'])
        df['text'] = df['contentChanges'].apply(lambda x: x[0]['text'])

        df.drop(columns=['contentChanges'], inplace=True)
    
        data_dict[p] = df

        if p == "P9":
            print("Keylog data for P9, length = ", len(df))
            print(log_path)

    return data_dict


'''
For every log data find the successive words where "starting" first appears
Here's how we filter for the relevant log data, filtering in order:
Start:
- Must be after the watch is turned on (eda data first time value)
- Must be after the first button press(if any)
- Must be after typing starting (if any)
End:
- Must be before the watch is turned off (eda data last time value)
- Must be before the last button press
- Must be before typing ending (if any)

'''


def return_eda_times(empatica_data, keylogger=False, a3=False):
    if a3 == True:
        p_list = p_list_a3_eda if keylogger == False else p_list_a3_log_analysis
    else:
        p_list = p_list_a4_eda if keylogger == False else p_list_a4_log_analysis

    start_times = {}
    end_times = {}
    for p in p_list:
        start_times[p] = empatica_data[p]["EDA"]["Time"].iloc[0]
        end_times[p] = empatica_data[p]["EDA"]["Time"].iloc[-1]

    return start_times, end_times




def clip_for_start_end_times(empatica_data, keylog_data, a3, keylogger=False):
    if a3 == True:
        p_list = p_list_a3_eda if keylogger == False else p_list_a3_log_analysis
    else:
        p_list = p_list_a4_eda if keylogger == False else p_list_a4_log_analysis

    # Get the start and end time of the EDA data, clip the log data to match that 
    for p in p_list:
        # print(p)
        start_time = empatica_data[p]["EDA"]["Time"].iloc[0]
        end_time = empatica_data[p]["EDA"]["Time"].iloc[-1]
        keylog_data[p] = keylog_data[p][keylog_data[p]["Time"] > start_time]
        keylog_data[p] = keylog_data[p][keylog_data[p]["Time"] < end_time]

    start_locs = {}
    end_locs = {}
    # check for "Starting" string
    for p in keylog_data.keys():
        # create 7 time shifted columns for character
        for i in range(1, 8):
            keylog_data[p][f'text_{i}'] = keylog_data[p]['text'].shift(-i)
            
        keylog_data[p]["flag"] = keylog_data[p]["text"] + keylog_data[p]["text_1"] + keylog_data[p]["text_2"] + keylog_data[p]["text_3"] + keylog_data[p]["text_4"] + keylog_data[p]["text_5"] + keylog_data[p]["text_6"] + keylog_data[p]["text_7"]
        
        # Start: find the start of the experiment - first time the flag is "start", if there is such
        if "starting" in keylog_data[p]["flag"].values:
            start_locs[p] = keylog_data[p][keylog_data[p]["flag"] == "starting"].index[0]
        elif "Starting" in keylog_data[p]["flag"].values:
            start_locs[p] = keylog_data[p][keylog_data[p]["flag"] == "Starting"].index[0]
        else:
            start_locs[p] = -1

        # End: find the end of the experiment - first time the flag is "ending" or "finishing"
        if "ending" in keylog_data[p]["flag"].values:
            end_locs[p] = keylog_data[p][keylog_data[p]["flag"] == "ending"].index[0]
        elif "Ending" in keylog_data[p]["flag"].values:
            end_locs[p] = keylog_data[p][keylog_data[p]["flag"] == "Ending"].index[0]
        elif "finishing" in keylog_data[p]["flag"].values:
            end_locs[p] = keylog_data[p][keylog_data[p]["flag"] == "finishing"].index[0]
        elif "Finishing" in keylog_data[p]["flag"].values:
            end_locs[p] = keylog_data[p][keylog_data[p]["flag"] == "Finishing"].index[0]
        else:
            end_locs[p] = -1

    #remove lagged text columns
    for p in keylog_data.keys():
        keylog_data[p].drop(columns=['text_1', 'text_2', 'text_3', 'text_4', 'text_5', 'text_6', 'text_7', 'flag'], inplace=True)

    # If start_locs is not -1 then filter out for that
    for p in keylog_data.keys():
        if start_locs[p] != -1:
            keylog_data[p] = keylog_data[p][keylog_data[p].index >= start_locs[p]]
        if end_locs[p] != -1:
            keylog_data[p] = keylog_data[p][keylog_data[p].index <= end_locs[p]]

    # Do the final clipping
    for p in p_list:
        # print(p)
        start_time = max(empatica_data[p]["EDA"]["Time"].iloc[0], keylog_data[p]["Time"].iloc[0])
        end_time = min(empatica_data[p]["EDA"]["Time"].iloc[-1], keylog_data[p]["Time"].iloc[-1])

        empatica_data[p]["EDA"] = empatica_data[p]["EDA"][(empatica_data[p]["EDA"]["Time"] >= start_time) & (empatica_data[p]["EDA"]["Time"] <= end_time)]
        empatica_data[p]["EDA"].reset_index(drop=True, inplace=True)
        
        keylog_data[p] = keylog_data[p][(keylog_data[p]["Time"] >= start_time) & (keylog_data[p]["Time"] <= end_time)]
        keylog_data[p].reset_index(drop=True, inplace=True)

        # convert datetime to seconds since start
        empatica_data[p]["EDA"]["Time_s"] = (empatica_data[p]["EDA"]["Time"] - start_time).dt.total_seconds()

        keylog_data[p]["Time_s"] = (keylog_data[p]["Time"] - start_time).dt.total_seconds()

    
    return empatica_data, keylog_data
    

def print_start_end_times(empatica_data, keylog_data, a3):
    if a3 == True:
        p_list = p_list_a3_eda
    else:
        p_list = p_list_a4_eda

    for p in p_list:
        print(p)
        print(empatica_data[p]["EDA"]["Time"].iloc[0])
        print(keylog_data[p]["Time"].iloc[0])
        print(empatica_data[p]["EDA"]["Time"].iloc[-1])
        print(keylog_data[p]["Time"].iloc[-1])

'''
Upsample eda signal 
'''
def upsample_eda_signal(empatica_data, a3, eda_freq, upsampled_freq):
    if a3 == True:
        p_list = p_list_a3_eda
    else:
        p_list = p_list_a4_eda

    for p in p_list:
        eda = np.array(nk.signal_resample(empatica_data[p]["EDA"]["EDA"], sampling_rate=eda_freq, desired_sampling_rate=upsampled_freq, method="linear"))
        time  = empatica_data[p]["EDA"]["Time_s"]
        x = np.arange(0, len(time)*2, 2)
        x_new = np.arange(len(time)*2)
        time = np.interp(x_new, x, time)

        # do the same for Time - interpolate
        date_time = empatica_data[p_list[0]]["EDA"]
        middle_times = date_time['Time'][:-1].values + (date_time['Time'][1:].values - date_time['Time'][:-1].values) / 2
        df_middle = pd.DataFrame({'Time': middle_times})
        df_final = pd.concat([date_time, df_middle]).sort_values(by='Time').reset_index(drop=True)

        empatica_data[p]["EDA"] = pd.DataFrame({"Time_s": time, "EDA": eda})
        empatica_data[p]["EDA"]["Time"] = df_final["Time"]


def apply_butterforth_filter(empatica_data, a3):
    if a3 == True:
        p_list = p_list_a3_eda
    else:
        p_list = p_list_a4_eda
    for p in p_list:
        eda = empatica_data[p]["EDA"]["EDA"]
        empatica_data[p]["EDA"]["EDA"] = nk.bio_process(edadata=eda, sampling_rate=upsampled_freq, method="butterworth", cutoff=butterpass_cutoff, order=butterpass_order)
    return empatica_data

def process_eda_signal(empatica_data, a3):
    if a3 == True:
        p_list = p_list_a3_eda
    else:
        p_list = p_list_a4_eda
    for p in p_list:
        signals, info = nk.eda_process(empatica_data[p]['EDA']['EDA'], sampling_rate=eda_freq)
        date_time = empatica_data[p_list[0]]["EDA"]["Time"]
        signals["Time_s"] = empatica_data[p]["EDA"]["Time_s"]
        empatica_data[p]["EDA"] = signals
        empatica_data[p]["EDA"]["Time"] = date_time

        empatica_data[p]["EDA"]["EDA_Tonic_RollingAverage"] = empatica_data[p]["EDA"]["EDA_Tonic"].expanding(min_periods=1).mean()
        empatica_data[p]["EDA"].fillna(0, inplace=True)

    return empatica_data


'''
Optional to do, try to filter out potential motion artifacts etc
'''
def artifact_analysis():
    raise NotImplemented



def create_windowed_data(empatica_data, keylog_data, a3):
    if a3 == True:
        p_list = p_list_a3_eda
    else:
        p_list = p_list_a4_eda
    for p in p_list:
        # generate divider times
        start_time = min(empatica_data[p]["EDA"]['Time_s'].min(), keylog_data[p]['Time_s'].min())
        end_time = max(empatica_data[p]["EDA"]['Time_s'].max(), keylog_data[p]['Time_s'].max())
        divider_times = np.arange(start_time, end_time + window_length, window_length)  # +4 to ensure the last interval is included

        # assign to windows 
        empatica_data[p]["EDA"]['Window'] = pd.cut(empatica_data[p]["EDA"]['Time_s'], bins=divider_times, include_lowest=True, labels=False)
        keylog_data[p]['Window'] = pd.cut(keylog_data[p]['Time_s'], bins=divider_times, include_lowest=True, labels=False)

        # remove the last window as it may be incomplete
        empatica_data[p]["EDA"] = empatica_data[p]["EDA"][empatica_data[p]["EDA"]['Window'] != empatica_data[p]["EDA"]['Window'].max()]
        keylog_data[p] = keylog_data[p][keylog_data[p]['Window'] != keylog_data[p]['Window'].max()]

        # print(len(empatica_data[p]["EDA"]['Window'].unique())/len(log_data[p]['Window'].unique()))

    return empatica_data, keylog_data


'''
Here we commpute features for each window
- features that Anam used:  𝑖) mean and standard deviation of the raw EDA values, 𝑖𝑖) mean
and standard deviation of the phasic signal, 𝑖𝑖𝑖) mean and standard
deviation of the tonic signal, 𝑖𝑣) number of scr peaks,𝑣) average scr
peak amplitude, and 𝑣𝑖) average scr rise time.
'''
def create_aggregated_eda_window_features(empatica_data, a3):
    if a3 == True:
        p_list = p_list_a3_eda
    else:
        p_list = p_list_a4_eda

    for p in p_list:

        df = empatica_data[p]["EDA"].copy()
        grouped = df.groupby('Window')

        non_zero_mean = lambda x: x[x != 0].mean() if (x != 0).any() else 0

        aggregated_data = grouped.agg(
            number_of_scr_onsets=pd.NamedAgg(column='SCR_Onsets', aggfunc='sum'),
            number_of_scr_peaks=pd.NamedAgg(column='SCR_Peaks', aggfunc='sum'),
            number_of_scr_recoveries=pd.NamedAgg(column='SCR_Recovery', aggfunc='sum'),  # Assuming this column indicates recoveries
            mean_scr_height=pd.NamedAgg(column='SCR_Height', aggfunc=non_zero_mean),
            mean_scr_amplitude=pd.NamedAgg(column='SCR_Amplitude', aggfunc=non_zero_mean),
            mean_scr_rise_time=pd.NamedAgg(column='SCR_RiseTime', aggfunc=non_zero_mean),
            mean_scr_recovery_time=pd.NamedAgg(column='SCR_RecoveryTime', aggfunc=non_zero_mean)
        ).reset_index()

        empatica_data[p]["EDA_windowed"] = aggregated_data
    
    return empatica_data


'''
Here we compute a lot of potentially useful features for each 4s window of EDA data
- Use the LateralizationPhysiologicalWearables Paper's source code
'''
def create_extra_aggregated_eda_window_features(empatica_data, a3):
    if a3 == True:
        p_list = p_list_a3_eda
    else:
        p_list = p_list_a4_eda

    feature_names = ["min_feat","max_feat","mean_feat","std_feat","dynamic_range_feat","slope_feat","absolute_slope_feat","first_derivetive_mean_feat",
                  "first_derivative_std_feat","dc_term","sum_of_all_coefficients","information_entropy", "spectral_energy",]
    column_names = ['Window'] + ['mixed_' + name for name in feature_names] + ['tonic_' + name for name in feature_names] + ['phasic_' + name for name in feature_names]

    for p in p_list:
    # for p in ["P11"]: # for testing
        print("Now processing participant: ", p)
        eda_df = empatica_data[p]["EDA"]
        features_list = []

        for window in eda_df['Window'].unique():
        # for window in range(0,10): # for testing
            window_df = eda_df[eda_df['Window'] == window]

            # Get each EDA signal in numpy array format
            eda_clean = window_df["EDA_Clean"].to_numpy().reshape(-1, 1)
            eda_tonic = window_df["EDA_Tonic"].to_numpy().reshape(-1, 1)
            eda_phasic = window_df["EDA_Phasic"].to_numpy().reshape(-1, 1)
            
            # Get the features
            eda_clean_features  = get_eda_features(eda_clean, sampling_rate=eda_freq)
            eda_tonic_features = get_eda_features(eda_tonic, sampling_rate=eda_freq)
            eda_phasic_features = get_eda_features(eda_phasic, sampling_rate=eda_freq)

            features = [window] + [ val[0] for val in eda_clean_features] + [ val[0] for val in eda_tonic_features] + [val[0] for val in eda_phasic_features]

            # features = [window] + [ val[0] for val in eda_clean_features]

            features_list.append(features)
            
        features_df = pd.DataFrame(features_list, columns=column_names) 
        features_df = features_df.fillna(0)# scr peak amplitudes are null fi there is no peak

        # Merge the features with the existing windowed EDA data 
        existing_features_df = empatica_data[p]["EDA_windowed"]
        new_features_df = pd.merge(existing_features_df, features_df, on='Window')

        empatica_data[p]["EDA_windowed"] = new_features_df

        # standard scale
        df = empatica_data[p]["EDA_windowed"]
        columns_to_scale = df.columns.difference(['Window'])
        scaler = StandardScaler()
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

        # save scaled data
        empatica_data[p]["EDA_windowed"] = df.copy()

    return empatica_data


'''
Ideas for aggregation: 

- Number of characters changed
- Time since last change
- Is bigger than average character change across all windows?
- Is bigget than average character change across the past 10 windows?
'''
def create_windowed_keylogger_features(keylog_data, empatica_data, a3):
    if a3 == True:
        p_list = p_list_a3_eda
    else:
        p_list = p_list_a4_eda

    log_data_windows = {} # will store new dataframes here

    for p in p_list:
        log_df = keylog_data[p]
        windows  = empatica_data[p]["EDA_windowed"]["Window"].unique()

        # Feature engineering on log daa

        # Get length of text changes made
        log_df["text_length"] = log_df["text"].apply(lambda x: len(x))

        # Adjust the text length of new line characters since they count as 9 (count them as 1)
        rows_with_newline = log_df[log_df["text"].str.contains('\n\s*')]
        log_df['newline_count'] = log_df["text"].str.count('\n\s*')
        log_df['text_length'] = log_df['text_length'] - log_df['newline_count'] * 8

        # get the percentiles of text_length
        percentiles = log_df['text_length'].quantile([0.95])

        # create new column, mark anything above 95th percentile as 1
        log_df['text_length_flag'] = log_df['text_length'].apply(lambda x: 1 if x > percentiles.values[0] else 0)

        # create new column to record the time difference between two consecutive changes
        log_df['time_diff'] = log_df['Time_s'].diff().fillna(0)

        # Aggregation
        aggregated_df = log_df.groupby('Window').agg(
        unique_file_name=pd.NamedAgg(column='file_name', aggfunc=lambda x: x.nunique()),
        sum_text_length=pd.NamedAgg(column='text_length', aggfunc='sum'),
        count_text_length_flag=pd.NamedAgg(column='text_length_flag', aggfunc='sum'),  # Since flags are 1, sum will count them
        max_time_diff=pd.NamedAgg(column='time_diff', aggfunc='max'),
        earliest_time_diff=pd.NamedAgg(column='time_diff', aggfunc=lambda x: x[x > 0].min() if any(x > 0) else 0)
        ).reset_index()

        aggregated_df["activity"] = 1

        # create dummy df to fill in missing windows
        all_windows_df = pd.DataFrame(windows, columns=['Window'])
        for column in ['sum_text_length', 'count_text_length_flag', 'max_time_diff', 'earliest_time_diff', 'activity']:
            all_windows_df[column] = 0
        all_windows_df['unique_file_name'] = 1 # this is 1 by default
        
        # Before merging, set 'Window' as the index for both DataFrames
        all_windows_df.set_index('Window', inplace=True)
        aggregated_df.set_index('Window', inplace=True)

        # Align the data types of all_windows_df with aggregated_df to avoid dtype incompatibility
        for column in aggregated_df.columns:
            all_windows_df[column] = all_windows_df[column].astype(aggregated_df[column].dtype)
            
        # Use update to overwrite values in all_windows_df with those in aggregated_df where they exist
        all_windows_df.update(aggregated_df, overwrite=True)

        # Reset the index to bring 'Window' back as a column
        final_df = all_windows_df.reset_index()

        # FOR ASSIGNMENT 3 THIS IS ALWAYS JUST ONE FILE SO DROP
        final_df.drop(columns=['unique_file_name'], inplace=True)
                
        # Standard scale all columns except windows
        columns_to_scale = final_df.columns.difference(['Window', 'activity'])
        scaler = StandardScaler()   
        final_df[columns_to_scale] = scaler.fit_transform(final_df[columns_to_scale])

        # Store the result in your dictionary
        log_data_windows[p] = final_df.copy()
    
    return log_data_windows

