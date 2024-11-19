
from data_prep_helper import *
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import subprocess
import pandas as pd
import concurrent.futures
import subprocess
import re

snapshots_dir_a3 = 'code_snapshots_a3'
snapshots_dir_a4 = 'code_snapshots_a4'
snapshots_dir_a3_unclipped = 'code_snapshots_a3_unclipped'
snapshots_dir_a4_unclipped = 'code_snapshots_a4_unclipped'
snapshots_dir_a3_unclipped_from_scratch = 'code_snapshots_a3_unclipped_from_scratch'
snapshots_dir_a4_unclipped_from_scratch = 'code_snapshots_a4_unclipped_from_scratch' 

def create_directories():
    os.makedirs(snapshots_dir_a3, exist_ok=True)
    os.makedirs(snapshots_dir_a4, exist_ok=True)
    os.makedirs(snapshots_dir_a3_unclipped, exist_ok=True)
    os.makedirs(snapshots_dir_a4_unclipped, exist_ok=True)
    os.makedirs(snapshots_dir_a3_unclipped_from_scratch, exist_ok=True)
    os.makedirs(snapshots_dir_a4_unclipped_from_scratch, exist_ok=True)

a3_file_1 = 'bustersAgents.py'
a3_file_2 = 'inference.py'
a4_file_1 = 'NeuralNet.py'
a4_file_2 = 'NeuralNetUtil.py'

a3_file_1_original_path = f"archive/IncompleteAssignments/{a3_file_1}"
a3_file_2_original_path = f"archive/IncompleteAssignments/{a3_file_2}"
a4_file_1_original_path = f"archive/IncompleteAssignments/{a4_file_1}"
a4_file_2_original_path = f"archive/IncompleteAssignments/{a4_file_2}"


n_snapshot_interval_minutes = 5

a3_autograder_commands = {
    'q1': ['python autograder.py -q q1 --no-graphics',3],
    'q2': ['python autograder.py -q q2 --no-graphics',4],
    'q3': ['python autograder.py -q q3 --no-graphics',3],
    'q4': ['python autograder.py -q q4 --no-graphics',3],
    'q5': ['python autograder.py -q q5 --no-graphics',4],
    'q6': ['python autograder.py -q q6 --no-graphics',1],
    'q7': ['python autograder.py -q q7 --no-graphics',2]
}

a4_autograder_commands = {
    'q1': ['python autograder.py -q q1',2],
    'q2': ['python autograder.py -q q2',2],
    'q3': ['python autograder.py -q q3',4],
    'q4': ['python autograder.py -q q4',4],
}


def get_keylog_dfs():
    empatica_data_a3 = get_empatica_data(a3=True,keylogger=True)
    empatica_data_a4 = get_empatica_data(a3=False, keylogger=True)
    keylog_data_a3 = get_keylog_data(a3=True, keylogger=True)
    keylog_data_a4 = get_keylog_data(a3=False, keylogger=True)
    keylog_data_a3_original = keylog_data_a3.copy()
    keylog_data_a4_original = keylog_data_a4.copy()
    _, keylog_data_a3 = clip_for_start_end_times(empatica_data_a3, keylog_data_a3, a3=True, keylogger=True)
    _, keylog_data_a4 = clip_for_start_end_times(empatica_data_a4, keylog_data_a4, a3=False, keylogger=True)

    # start times from 0
    for p in keylog_data_a3.keys():
        keylog_data_a3[p] = format_keylog_data(keylog_data_a3[p])
        keylog_data_a3_original[p] = format_keylog_data(keylog_data_a3_original[p])
    for p in keylog_data_a4.keys():
        keylog_data_a4[p] = format_keylog_data(keylog_data_a4[p])
        keylog_data_a4_original[p] = format_keylog_data(keylog_data_a4_original[p])

    
    print(len(keylog_data_a3), len(keylog_data_a4))
    return keylog_data_a3, keylog_data_a4 , keylog_data_a3_original, keylog_data_a4_original


def format_keylog_data(df):
    df["T_s"] = df["T_s"] - df["T_s"].min()
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    # df = df.drop(columns=['Time_s'])
    return df


'''
Print session length, max interval between no keystrokes
'''
def print_time_info(p,df,n):
    total_time = (df['T_s'].max() - df['T_s'].min()) /60
    temp_df = df.iloc[:n]
    max_diff = temp_df['T_s'].diff().max()
    max_diff_idx = temp_df['T_s'].diff().idxmax()
    max_diff_time = temp_df['T_s'][max_diff_idx]
    max_diff_time_prev = temp_df['T_s'][max_diff_idx - 1]
    print(f"{p}, total run time {round(total_time,1)}, Max time difference  between keystrokes: {round(max_diff/60,1)}m, at index: {max_diff_idx},  starting: {round(max_diff_time_prev)}, ending:{round(max_diff_time)}")


'''

'''
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


'''
Read file as single block of string
'''
def read_file_as_string(file_path):
    with open(file_path, 'r') as f:
        return f.read()


'''
Create snapshots every n minutes of the active code files for each student in the dataset
'''
def create_snapshots(p, df, snapshot_directory, a3=True, use_original_files=False):

    # Define snapshot intervals
    snapshot_interval = pd.Timedelta(minutes=n_snapshot_interval_minutes)
    start_time = df['Time'].min() 
    end_time = df['Time'].max() + snapshot_interval
    snapshot_times = pd.date_range(start=start_time + snapshot_interval, end=end_time, freq=f'{n_snapshot_interval_minutes}min')
    print(f"\nCreating snapshots for p {p}, snapshots start at {start_time}, and end at {end_time}")

    # Define file paths and read files
    if not use_original_files:
        code_file_path_bustersagent = f"/Users/feyzjan/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/GatechCourses/CS 8903 Research/A3_DataFiles/{p}/Session_Start_{p}/tracking/bustersAgents.py"
        code_file_path_inference = f"/Users/feyzjan/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/GatechCourses/CS 8903 Research/A3_DataFiles/{p}/Session_Start_{p}/tracking/inference.py"
        code_file_path_neuralnet = f"/Users/feyzjan/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/GatechCourses/CS 8903 Research/A4_DataFiles/{p}/Session_Start_{p}/NeuralNet.py"
        code_file_path_neuralnetutil = f"/Users/feyzjan/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/GatechCourses/CS 8903 Research/A4_DataFiles/{p}/Session_Start_{p}/NeuralNetUtil.py"
    else:
        code_file_path_bustersagent = a3_file_1_original_path
        code_file_path_inference = a3_file_2_original_path 
        code_file_path_neuralnet = a4_file_1_original_path 
        code_file_path_neuralnetutil = a4_file_2_original_path


    assignment = "a3" if a3 else "a4"
    snapshots_dir = snapshot_directory

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
        print("File paths are ", code_file_path_bustersagent, code_file_path_inference)
        raise FileNotFoundError

    # Save the initial version of the files
    counter = 0
    initial_snapshot_path = os.path.join(snapshots_dir, f'{assignment}_{p}_c{counter}')
    counter +=1
    os.makedirs(initial_snapshot_path, exist_ok=True)
    for file_name, content in original_files_content.items():
        with open(os.path.join(initial_snapshot_path, file_name), 'w') as f:
            f.write(content)
    print(f"Saved initial snapshot: {initial_snapshot_path}")

    # initialize snapshot column
    df['snapshot'] = 0
    last_snapshot_time = start_time -  snapshot_interval
    # Iterate over each snapshot time
    for snapshot_counter, snapshot_time in enumerate(snapshot_times, start=1):
        print(f"Processing snapshot {snapshot_counter} for participant {p} at {snapshot_time}")

        # TODO: May just continue using the updated files_content
        files_content = original_files_content.copy()

        # Filter changes up to the current snapshot time
        changes_up_to_snapshot = df[df['Time'] <= snapshot_time]
        changes_in_snapshot = df[(df['Time'] > last_snapshot_time) & (df['Time'] <= snapshot_time)]

        # Update the snapshot column
        df.loc[
            (df['Time'] > last_snapshot_time) &
            (df['Time'] <= snapshot_time) &
            (df['snapshot'] == 0),
            'snapshot'] = snapshot_counter
        
        if changes_in_snapshot.empty:
            print(f"No changes detected in snapshot {snapshot_counter}, skipping save.")
            last_snapshot_time = snapshot_time
            continue  # Skip to the next snapshot

        last_snapshot_time = snapshot_time

        print("Number of changes up to this snapshot time: ", len(changes_up_to_snapshot))

        # Apply changes sequentially
        for index, row in changes_up_to_snapshot.iterrows():
            file_name = row['file_name']

            if file_name not in files_content:
                print(f"error File {file_name} not recognized")
                continue

            # Get the content changes
            try:
                files_content[file_name] = apply_change(files_content[file_name], row)
            except Exception as e:
                print(f"Error applying change for participant {p}, file {file_name} at index {index}: with error {e}")
                print("row details are ", row)
                raise e

        # Save the current state of the files
        snapshot_filename = f'{assignment}_{p}_c{snapshot_counter}'
        counter +=1
        snapshot_path = os.path.join(snapshots_dir, snapshot_filename)
        os.makedirs(snapshot_path, exist_ok=True)

        for file_name, content in files_content.items():
            with open(os.path.join(snapshot_path, file_name), 'w') as f:
                f.write(content)

        print(f"Saved snapshot: {snapshot_filename}")

    # Assign snapshot number to any remaining rows
    if len(df[df['snapshot'] == 0]) > 0:
        snapshot_counter += 1
        df.loc[df['snapshot'] == 0, 'snapshot'] = snapshot_counter
        print(f"Assigned snapshot {snapshot_counter} to remaining {len(df[df['snapshot'] == snapshot_counter])} changes.")

    return df



'''
Move the necessary assignment files to each snapshot directory so we can run the autograder.
- May be more efficient to move the snapshots somewhere else and run the autograder from there, but this does the job.
'''
def move_assignment_files(a3=True, code_snapshots_dir=None):
    # Define the directories
    if a3:
        files_to_copy_dir = 'a3_files_to_copy'
    else:
        files_to_copy_dir = 'a4_files_to_copy'

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



""" 
Run autograder for a given question
"""
def run_autograder_for_question(tracking_dir, question, command):
    points_earned = 0
    try:
        print("\nRunning autograder for question", question)
        cmd = command[0].split()
        result = subprocess.run(cmd, cwd=tracking_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        autograder_error = result.returncode != 0
        # print("autograder result is:", result)
        # print("stdout is:", result.stdout)

        # Extract total points if the pattern exists in stdout
        pattern = r'Total:\s(\d+)/(\d+)'
        # print(str(result))
        match = re.search(pattern, str(result))
        if match:
            points_earned = int(match.group(1))
            print(f" Score match found - For {question} the student earned {points_earned} points.")
        else:
            points_earned = 0
            print(f"Score match not found For {question} the student earned {points_earned}? points.")

        return (question, points_earned, autograder_error)

    except subprocess.TimeoutExpired:
        print(f"For {question} autograder timed out in {tracking_dir}")
        return (question, 0, True)
    except Exception as e:
        print(f"For {question} error running autograder in {tracking_dir}: {e}")
        return (question, 0, True)
    


"""
Run the autograder on each snapshot directory for each question and save the results to a dataframe.
"""
def run_local_autograder(a3=True, snapshots_dir = None, start_idx=0, end_idx=-1):

    # Set assignment parameters
    if a3:
        files_to_measure = ['bustersAgents.py', 'inference.py']
        autograder_commands = a3_autograder_commands
    else:
        files_to_measure = ['NeuralNet.py', 'NeuralNetUtil.py']
        autograder_commands = a4_autograder_commands

    data = []

    # Loop over snapshot directories in alphabetical order
    for snapshot_dir in sorted(os.listdir(snapshots_dir))[start_idx:end_idx]:
        snapshot_path = os.path.join(snapshots_dir, snapshot_dir)
        tracking_dir = os.path.join(snapshot_path, 'tracking') if a3 else snapshot_path

        if not os.path.isdir(snapshot_path):
            continue 

        # Extract participant number and snapshot number from snapshot_dir
        # Snapshot_dir is in the format 'a3_P1_1' or 'a4_P1_1'
        parts = snapshot_dir.split('_')
        if len(parts) < 3:
            raise Exception(f"Directory {snapshot_dir}: unexpected name format.")

        assignment, participant_num, snapshot_num = parts[0], parts[1], parts[2]
        snapshot_num = int(snapshot_num[1:])# CHECK THIS
        print(f"\n ---- Processing participant {participant_num}, snapshot {snapshot_num}..., assignment {assignment}")
        

        file_lengths = {}
        for file_name in files_to_measure:
            file_path = os.path.join(tracking_dir, file_name)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    file_length = len(content)  # Number of characters
                    # file_length = len(content.splitlines())
                    file_lengths[file_name] = file_length
            else:
                file_lengths[file_name] = 0  # File not found, length 0


        # Initialize row to add to dataframe
        row = {
            'participant': participant_num,
            'snapshot': snapshot_num,
        }

        # Check syntax of the files
        for file_name in files_to_measure:
            print(f"Checking syntax for {file_name}")
            file_path = os.path.join(tracking_dir, file_name)
            compiles = check_syntax(file_path)
            row[file_name + '_compiles'] = compiles

        # add file lengths
        for file_name in files_to_measure:
            row[file_name + '_length'] = file_lengths.get(file_name, 0)
        

        # Run autograder
        autograder_path = os.path.join(tracking_dir, 'autograder.py')

        if os.path.isfile(autograder_path):
            # for question, command in autograder_commands.items():
            #     # cmd = command[0].split()
            #     cmd = command[0].split()
            #     max_points = command[1]
            #     print("\nRunning autograder for question" , question)
            #     try:
            #         result = subprocess.run(cmd, cwd=tracking_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            #         autograder_error = result.returncode != 0
            #         print("autograder result is:", result)
            #         print("stdout is:", result.stdout)
            #         # TODO: Read the point from Total: int/int
            #     except subprocess.TimeoutExpired:
            #         print(f"Autograder timed out in {tracking_dir}")
            #         autograder_error = True
            #     except Exception as e:
            #         print(f"Error running autograder in {tracking_dir}: {e}")
            #         autograder_error = True

            #     points = max_points if not autograder_error else 0
            #     # TODO: Check if there are partial points
            #     row[f"{question}_points"] = points
    

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_question = {
                    executor.submit(run_autograder_for_question, tracking_dir, question, command): question
                    for question, command in autograder_commands.items()
                }
                for future in concurrent.futures.as_completed(future_to_question):
                    question = future_to_question[future]
                    try:
                        question, points, autograder_error = future.result()
                        row[f"{question}_points"] = points
                    except Exception as e:
                        print(f"Error processing question {question}: {e}")
                        row[f"{question}_points"] = 0

        data.append(row)

    df = pd.DataFrame(data)
    df.sort_values(by=['participant', 'snapshot'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # create total_points column
    for question in autograder_commands.keys():
        df[f"{question}_points"] = df[f"{question}_points"].fillna(0)
        df['total_points'] = df[[f"{question}_points" for question in autograder_commands.keys()]].sum(axis=1)
        # create another sum that takes the max for each question up to that point
        df['total_points_max'] = df[[f"{question}_points" for question in autograder_commands.keys()]].cummax(axis=1).sum(axis=1)

    return df



import py_compile
def check_syntax(file_path):
    try:
        py_compile.compile(file_path, doraise=True)
        # print(f"{file_path} compiled successfully. No syntax errors found.")
        return 1
    except py_compile.PyCompileError as compile_error:
        print(f"Syntax error in {file_path}:\n{compile_error}")
        return 0
    except Exception as e:
        print(f"An unexpected error occurred while checking {file_path}:\n{e}")
        return 0



def plot_keylogger_results(df, a3=True, plot_lengths=False):
    participants = df['participant'].unique()
    for participant in participants:
        # Filter data for the participant
        participant_df = df[df['participant'] == participant]
        
        # Set up the figure with two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        file_1_length = 'bustersAgents.py_length' if a3 else 'NeuralNet.py_length'
        file_2_length = 'inference.py_length' if a3 else 'NeuralNetUtil.py_length'

        print(file_1_length, file_2_length)
        
        # Plot 1: File lengths over snapshots
        if plot_lengths:
            sns.lineplot(
                data=participant_df,
                x='snapshot',
                y=file_1_length,
                label=file_1_length,
                ax=axes[0]
            )
            sns.lineplot(
                data=participant_df,
                x='snapshot',
                y=file_2_length,
                label=file_2_length,
                ax=axes[0]
            )
            axes[0].set_title(f'File Lengths for Participant {participant}')
            axes[0].set_xlabel('Snapshot Number')
            axes[0].set_ylabel('File Length (Characters)')
            axes[0].legend(loc='upper left')

        # Plot 2: Total points earned over snapshots
        sns.lineplot(
            data=participant_df,
            x='snapshot',
            y='total_points_max',
            label='Total Points Max',
            color='red',
            ax=axes[1]
        )
        axes[1].set_title(f'Total Points Earned for Participant {participant}')
        axes[1].set_xlabel('Snapshot Number')
        axes[1].set_ylabel('Total Points')
        axes[1].set_ylim(0, participant_df['total_points_max'].max() + 1)  # Adjust Y-axis limits if needed
        axes[1].legend(loc='upper left')

        # Adjust layout for better visualization
        plt.suptitle(f'Assignment {"a3" if a3 else "a4"}, Participant {participant}')
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Add space for the main title
        
        # Show the plots
        plt.show()

