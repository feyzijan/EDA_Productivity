
from data_prep_helper import *
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

snapshots_dir_a3 = 'code_snapshots_a3'
snapshots_dir_a4 = 'code_snapshots_a4'

def get_keylog_dfs():
    empatica_data_a3 = get_empatica_data(a3=True)
    empatica_data_a4 = get_empatica_data(a3=False)
    keylog_data_a3 = get_keylog_data(a3=True)
    keylog_data_a4 = get_keylog_data(a3=False)
    _, keylog_data_a3 = clip_for_start_end_times(empatica_data_a3, keylog_data_a3, a3=True)
    _, keylog_data_a4 = clip_for_start_end_times(empatica_data_a4, keylog_data_a4, a3=False)

    # start times from 0
    for df in keylog_data_a3.values():
        df["T_s"] = df["T_s"] - df["T_s"].min()
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time').reset_index(drop=True)
        df['change_length'] = df['range_length'] + 1

    for df in keylog_data_a4.values():
        df["T_s"] = df["T_s"] - df["T_s"].min()
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time').reset_index(drop=True)
        df['change_length'] = df['range_length'] + 1


    # Remove participants without start data
    for p in a3_participants_to_remove_keylogs:
        keylog_data_a3.pop(p)
    for p in a4_participants_to_remove_keylogs:
        keylog_data_a4.pop(p)
    
    print(len(keylog_data_a3), len(keylog_data_a4))
    return keylog_data_a3, keylog_data_a4



def print_time_info(p,df,n):
    total_time = (df['T_s'].max() - df['T_s'].min()) /60
    temp_df = df.iloc[:n]
    max_diff = temp_df['T_s'].diff().max()
    max_diff_idx = temp_df['T_s'].diff().idxmax()
    max_diff_time = temp_df['T_s'][max_diff_idx]
    max_diff_time_prev = temp_df['T_s'][max_diff_idx - 1]
    print(f"{p}, total run time {round(total_time,1)}, Max time difference  between keystrokes: {round(max_diff/60,1)}m, at index: {max_diff_idx},  starting: {round(max_diff_time_prev)}, ending:{round(max_diff_time)}")


def apply_change(file_content, row):
    """
    Applies a change to the file_content string based on a single content change event.

    Parameters:
    - file_content: String representing the entire file content.
    - change: A dictionary representing a content change event.

    Returns:
    - Updated file content string.
    """
    # Extract change details
    range_offset = row['range_offset']
    range_length = row['range_length']
    new_text = row['text']
    
    # Ensure range_offset and range_length are valid
    file_length = len(file_content)
    # range_offset = min(max(range_offset, 0), file_length)
    range_offset = range_offset
    # end_offset = min(range_offset + range_length, file_length)
    end_offset = range_offset + range_length

   	# •	Insertion: RangeLength = 0 and text:non-empty. - seems ok
    if range_length == 0 and new_text != "":
        file_content = file_content[:range_offset] + new_text + file_content[end_offset:]
       
	# •	÷Deletion: rangeLength >0. text:empty - seems ok
    elif range_length > 0 and new_text == "":
        file_content = file_content[:range_offset] + file_content[end_offset:]

	# •	Replacement: rangeLength >0 and text:non-empty.
    elif range_length > 0 and new_text != "":
        file_content = file_content[:range_offset] + new_text + file_content[end_offset:]
    else:
        print("Invalid change event:", row)

    # check if we ever have range_length = 0 and text:3empty

    return file_content


def read_file_as_string(file_path):
    with open(file_path, 'r') as f:
        return f.read()


def create_snapshots(p, df, a3=True, snapshot_interval_minutes=10):
    counter = 0
    error_count = 0
    success_count = 0
    # Define snapshot interval
    snapshot_interval = pd.Timedelta(minutes=snapshot_interval_minutes)
    start_time = df['Time'].min()
    end_time = df['Time'].max()
    snapshot_times = pd.date_range(start=start_time, end=end_time, freq=f'{snapshot_interval_minutes}min')

    # Define file paths
    code_file_path_bustersagent = f"/Users/feyzjan/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/GatechCourses/CS 8903 Research/A3_DataFiles/{p}/Session_Start_{p}/tracking/busters.py"
    code_file_path_inference = f"/Users/feyzjan/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/GatechCourses/CS 8903 Research/A3_DataFiles/{p}/Session_Start_{p}/tracking/inference.py"
    
    code_file_path_neuralnet = f"/Users/feyzjan/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/GatechCourses/CS 8903 Research/A4_DataFiles/{p}/Session_Start_{p}/NeuralNet.py"
    code_file_path_neuralnetutil = f"/Users/feyzjan/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/GatechCourses/CS 8903 Research/A4_DataFiles/{p}/Session_Start_{p}/NeuralNetUtil.py"
    assignment = "a3" if a3 else "a4"
    snapshots_dir = snapshots_dir_a3 if a3 else snapshots_dir_a4

    try:
        if a3:
            original_files_content = {
                'bustersAgents.py': read_file_as_string(code_file_path_bustersagent),
                'inference.py': read_file_as_string(code_file_path_inference)
            }
        else:
            original_files_content = {
                'NeuralNet.py': read_file_as_string(code_file_path_neuralnet),
                'NeuralNetUtil.py': read_file_as_string(code_file_path_neuralnetutil)
            }
    except FileNotFoundError:
        print(f"Warning: Files not found for participant {p}.")
        return  # Exit the function if files are missing

    # Save the initial version of the files
    initial_snapshot_path = os.path.join(snapshots_dir, f'{assignment}_{p}_c{counter}')
    counter +=1
    os.makedirs(initial_snapshot_path, exist_ok=True)
    for file_name, content in original_files_content.items():
        with open(os.path.join(initial_snapshot_path, file_name), 'w') as f:
            f.write(content)
    print(f"Saved initial snapshot: {initial_snapshot_path}")

    # Iterate over each snapshot time
    for snapshot_counter, snapshot_time in enumerate(snapshot_times, start=1):
        print(f"Processing snapshot {snapshot_counter} for participant {p} at {snapshot_time}")

        # TODO: May just continue using the updated files_content
        files_content = original_files_content.copy()

        # Filter changes up to the current snapshot time
        changes_up_to_snapshot = df[df['Time'] <= snapshot_time]

        # Apply changes sequentially
        for index, row in changes_up_to_snapshot.iterrows():
            file_name = row['file_name']
            
            if file_name not in files_content:
                print(f"Warning: File {file_name} not recognized.")
                continue

            # Get the content changes
            try:
                files_content[file_name] = apply_change(files_content[file_name], row)
                success_count += 1
            except Exception as e:
                print(f"Error applying change for participant {p}, file {file_name} at index {index}: with error {e}")
                print("row details are ", row)
                error_count += 1
                continue

        # Save the current state of the files
        snapshot_filename = f'{assignment}_{p}_c{snapshot_counter}'
        counter +=1
        snapshot_path = os.path.join(snapshots_dir, snapshot_filename)
        os.makedirs(snapshot_path, exist_ok=True)

        for file_name, content in files_content.items():
            with open(os.path.join(snapshot_path, file_name), 'w') as f:
                f.write(content)

        print(f"Saved snapshot: {snapshot_filename}")
        print(f"Success count: {success_count}, Error count: {error_count}")

    print("success count", success_count, "error count", error_count)



def move_assignment_files(a3=True):
    # Define the directories
    if a3:
        code_snapshots_dir = 'code_snapshots_a3'
        files_to_copy_dir = 'a3_files_to_copy'
    else:
        files_to_copy_dir = 'a4_files_to_copy'
        code_snapshots_dir = 'code_snapshots_a4'

    # Check if the directories exist
    if not os.path.isdir(code_snapshots_dir):
        print(f"Error: Directory '{code_snapshots_dir}' does not exist.")
        return

    if not os.path.isdir(files_to_copy_dir):
        print(f"Error: Directory '{files_to_copy_dir}' does not exist.")
        return

    # Get a list of snapshot directories
    snapshot_dirs = [os.path.join(code_snapshots_dir, d) for d in os.listdir(code_snapshots_dir) if os.path.isdir(os.path.join(code_snapshots_dir, d))]

    # Process each snapshot directory
    for snapshot_dir in snapshot_dirs:
        print(f"Processing '{snapshot_dir}'...")

        # Step 1: Copy contents of a3_files_to_copy into the snapshot directory
        copy_files(files_to_copy_dir, snapshot_dir)

        # Step 2: Move 'inference.py' and 'bustersAgents.py' into the 'tracking' folder
        if a3:
            target_dir = os.path.join(snapshot_dir, 'tracking')
        else:
            target_dir = snapshot_dir

        # Ensure the 'tracking' directory exists
        if not os.path.isdir(target_dir):
            print(f"Error: Tracking directory '{target_dir}' does not exist after copying.")
            continue

        if a3:
            move_file_if_exists(os.path.join(snapshot_dir, 'inference.py'), target_dir)
            move_file_if_exists(os.path.join(snapshot_dir, 'bustersAgents.py'), target_dir)
        else:
            move_file_if_exists(os.path.join(snapshot_dir, 'NeuralNet.py'), target_dir)
            move_file_if_exists(os.path.join(snapshot_dir, 'NeuralNetUtil.py'), target_dir)

    print("All snapshots have been processed.")

def copy_files(src_dir, dest_dir):
    """
    Copy all files and directories from src_dir to dest_dir.
    """
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dest_dir, item)
        try:
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        except Exception as e:
            print(f"Error copying '{s}' to '{d}': {e}")

def move_file_if_exists(file_path, dest_dir):
    """
    Move file to dest_dir if it exists.
    """
    if os.path.isfile(file_path):
        try:
            shutil.move(file_path, dest_dir)
            print(f"Moved '{os.path.basename(file_path)}' to '{dest_dir}'.")
        except Exception as e:
            print(f"Error moving '{file_path}' to '{dest_dir}': {e}")
    else:
        print(f"Warning: '{file_path}' does not exist and cannot be moved.")


import os
import subprocess
import pandas as pd
import re

def run_autograder(a3=True):

    # Set assignment parameters
    if a3:
        snapshots_dir = 'code_snapshots_a3'
        files_to_measure = ['bustersAgents.py', 'inference.py']
    else:
        snapshots_dir = 'code_snapshots_a4'
        files_to_measure = ['NeuralNet.py', 'NeuralNetUtil.py']

    # Initialize a list to collect data
    data = []

    # Loop over snapshot directories
    for snapshot_dir in os.listdir(snapshots_dir):
        snapshot_path = os.path.join(snapshots_dir, snapshot_dir)
        if not os.path.isdir(snapshot_path):
            continue  # Skip if not a directory

        # Extract participant number and snapshot number from snapshot_dir
        # Snapshot_dir is in the format 'a3_P1_1' or 'a4_P1_1'
        parts = snapshot_dir.split('_')
        if len(parts) < 3:
            print(f"Skipping directory {snapshot_dir}: unexpected name format.")
            continue 

        assignment, participant_num, snapshot_num = parts[0], parts[1], parts[2]
        snapshot_num = int(snapshot_num[1:])# CHECK THIS
        print(f"Processing participant {participant_num}, snapshot {snapshot_num}..., assignment {assignment}")
        

        # Prepare paths to the files
        # Assuming the relevant files are in 'tracking' subdirectory
        if a3:
            tracking_dir = os.path.join(snapshot_path, 'tracking')
        else:
            tracking_dir = snapshot_path

        file_lengths = {}
        for file_name in files_to_measure:
            file_path = os.path.join(tracking_dir, file_name)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    file_length = len(content)  # Number of characters
                    # If you prefer number of lines:
                    # file_length = len(content.splitlines())
                    file_lengths[file_name] = file_length
            else:
                file_lengths[file_name] = 0  # File not found, length 0

        # Run autograder
        autograder_path = os.path.join(tracking_dir, 'autograder.py')
        if os.path.isfile(autograder_path):
            if a3:
                cmd = ['python', 'autograder.py', '--no-graphics']
            else:
                cmd = ['python', 'autograder.py']
            try:
                result = subprocess.run(cmd, cwd=tracking_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
                autograder_error = result.returncode != 0
            except subprocess.TimeoutExpired:
                print(f"Autograder timed out in {tracking_dir}")
                autograder_error = True
        else:
            print(f"autograder.py not found in {tracking_dir}")
            autograder_error = True  # autograder.py not found

        # Collect data
        row = {
            'participant': participant_num,
            'snapshot': snapshot_num,
            'autograder_error': autograder_error,
        }
        for file_name in files_to_measure:
            row[file_name + '_length'] = file_lengths.get(file_name, 0)
        data.append(row)

    df = pd.DataFrame(data)
    df.sort_values(by=['participant', 'snapshot'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['autograder_error_numeric'] = df['autograder_error'].astype(int)

    return df




def plot_keylogger_results(df, a3=True):
    participants = df['participant'].unique()
    for participant in participants:
        # Filter data for the participant
        participant_df = df[df['participant'] == participant]
        
        # Set up the figure
        plt.figure(figsize=(8, 6))

        file_1_length = 'bustersAgents.py_length' if a3 else 'NeuralNet.py_length'
        file_2_length = 'inference.py_length' if a3 else 'NeuralNetUtil.py_length'
        
        # Plot the lengths
        sns.lineplot(
            data=participant_df,
            x='snapshot',
            y=file_1_length,
            label= file_1_length
        )
        sns.lineplot(
            data=participant_df,
            x='snapshot',
            y= file_2_length,
            label= file_2_length
        )
        
        # Create a twin axis to plot autograder_error
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        sns.lineplot(
            data=participant_df,
            x='snapshot',
            y='autograder_error_numeric',
            label='Autograder Error',
            color='red',
            ax=ax2
        )
        ax2.set_ylabel('Autograder Error (1=Error, 0=No Error)')
        ax2.set_ylim(-0.1, 1.1)  # Set y-limits for binary data
        
        # Customize the plot
        a_label = "a3" if a3 else "a4"
        plt.title(f'Assignment:{a_label}, Participant {participant} - Code Lengths and Autograder Errors Over Snapshots')
        ax1.set_xlabel('Snapshot Number')
        ax1.set_ylabel('File Length (Characters)')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Show the plot
        plt.show()