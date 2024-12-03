
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
import py_compile
import string
import copy


"""
- Notes on keylogger outputs:
- 	range:
	- Represents the span of text affected by the change.
	- Contains start and end positions, each with line and character properties (both zero-based indices).
- rangeOffset:
	- The offset of the change from the beginning of the document (in characters).
- rangeLength:
	- The length of the range that was replaced (in characters).
- text:
	- The new text inserted in place of the range.
- Line no's start from 0
- characters end at 200
"""


# Snapshot interval
n_snapshot_interval_minutes = 5


# Directories to store snapshots
snapshots_dir_a3 = 'code_snapshots_a3'
snapshots_dir_a4 = 'code_snapshots_a4'

def create_directories():
    os.makedirs(snapshots_dir_a3, exist_ok=True)
    os.makedirs(snapshots_dir_a4, exist_ok=True)

# File name and paths to their unedited versions
a3_file_1 = 'bustersAgents.py'
a3_file_2 = 'inference.py'
a4_file_1 = 'NeuralNet.py'
a4_file_2 = 'NeuralNetUtil.py'

a3_file_1_original_path = f"archive/IncompleteAssignments/{a3_file_1}"
a3_file_2_original_path = f"archive/IncompleteAssignments/{a3_file_2}"
a4_file_1_original_path = f"archive/IncompleteAssignments/{a4_file_1}"
a4_file_2_original_path = f"archive/IncompleteAssignments/{a4_file_2}"


# Commands to run the autograders for each question
a3_autograder_commands = {
    'q1': ['python autograder.py -q q1 --no-graphics',3],
    'q2': ['python autograder.py -q q2 --no-graphics',4],
    'q3': ['python autograder.py -q q3 --no-graphics',3],
    'q4': ['python autograder.py -q q4 --no-graphics',3],
    'q5': ['python autograder.py -q q5 --no-graphics',4],
    'q6': ['python autograder.py -q q6 --no-graphics',1],
    'q7': ['python autograder.py -q q7 --no-graphics',2]
}
a3_max_points = 17
a3_extra_credit_max_points = 20

a4_autograder_commands = {
    'q1': ['python autograder.py -q q1',2],
    'q2': ['python autograder.py -q q2',2],
    'q3': ['python autograder.py -q q3',4],
    'q4': ['python autograder.py -q q4',4],
}
a4_max_points = 12
a3_extra_credit_max_points = 12


"""
Get the keylog data for both assignments
"""
def get_keylog_dfs():
    keylog_data_a3 = get_keylog_data(a3=True, keylogger=True)
    keylog_data_a4 = get_keylog_data(a3=False, keylogger=True)

    for p in keylog_data_a3.keys():
        keylog_data_a3[p] = format_keylog_data(keylog_data_a3[p])
    for p in keylog_data_a4.keys():
        keylog_data_a4[p] = format_keylog_data(keylog_data_a4[p])
    
    return keylog_data_a3, keylog_data_a4 


# Normalize time and drop some columns
def format_keylog_data(df):
    df["T_s"] = df["T_s"] - df["T_s"].min()
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    return df


'''
Print session length, max interval between no keystrokes. For reference.
'''
def print_time_info(p,df,n):
    total_time = (df['T_s'].max() - df['T_s'].min()) /60
    temp_df = df.iloc[:n]
    max_diff = temp_df['T_s'].diff().max()
    max_diff_idx = temp_df['T_s'].diff().idxmax()
    max_diff_time = temp_df['T_s'][max_diff_idx]
    max_diff_time_prev = temp_df['T_s'][max_diff_idx - 1]
    print(f"\n{p}, total run time {round(total_time,1)} mins, Max time difference  between keystrokes: {round(max_diff/60,1)}m, at index: {max_diff_idx},  starting: {round(max_diff_time_prev)}, ending:{round(max_diff_time)}")



def read_file_as_lines(file_path):
    """
    Reads a file and returns its content as a list of lines, preserving line endings.
    Ensures that a trailing empty line is included if the file ends with a newline.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read the entire content
        content = f.read()
    # Normalize line endings to '\n' to ensure consistency
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    # Split into lines, keeping the line endings
    lines = content.splitlines(keepends=True)
    # Check if the file ends with a newline and append an empty line if necessary
    if content.endswith('\n') and (not lines or lines[-1] != '\n'):
        lines.append('')  # Add a final empty line
    return lines


def apply_change_by_lines(lines, row):
    """
    Applies a change to the lines list based on a single content change event.

    Parameters:
    - lines: List of strings representing the file content, one line per element.
    - row: A dictionary or Series representing a content change event, with fields:
        - 'start_line', 'start_character', 'end_line', 'end_character', 'text'

    Returns:
    - Updated list of lines.
    """
    # Extract change details
    start_line = int(row['start_line'])
    start_character = int(row['start_character'])
    end_line = int(row['end_line'])
    end_character = int(row['end_character'])
    new_text = row['text']

    # Ensure indices are within bounds
    if start_line < 0 or start_line >= len(lines):
        raise ValueError(f"start_line {start_line} is out of bounds")
    if end_line < 0 or end_line >= len(lines):
        raise ValueError(f"end_line {end_line} is out of bounds")

    # Extract the text before and after the change
    before_change = lines[start_line][:start_character]
    after_change = lines[end_line][end_character:]
    # after_change = lines[end_line][end_character:] if lines[end_line] else ""

    # Build the new content to replace the specified range
    new_text_lines = new_text.splitlines(keepends=True)

    if len(new_text_lines) == 0:
        # Deletion without replacement
        # Keep the line and remove the specified characters
        new_line = before_change + after_change
        new_lines = [new_line]
    else:
        # Merge the before_change with the first line of new_text
        new_text_lines[0] = before_change + new_text_lines[0]

        # Merge the after_change with the last line of new_text
        if not new_text.endswith('\n'):
            new_text_lines[-1] = new_text_lines[-1] + after_change
        else:
            # If new_text ends with a newline, include after_change in a new line
            new_text_lines.append(after_change)

        new_lines = new_text_lines

    # Replace the old lines with the new lines
    lines[start_line:end_line + 1] = new_lines

    return lines



'''
Create snapshots every n minutes of the active code files for each student in the dataset
'''
# TODO: Check snapshot 0, 1 and -1
def create_snapshots(p, df, eda_start_time, a3=True, use_original_files=False):

    # a3 or a4
    assignment = "a3" if a3 else "a4"
    snapshot_base_dir = snapshots_dir_a3 if a3 else snapshots_dir_a4

    # Define snapshot intervals
    snapshot_interval = pd.Timedelta(minutes=n_snapshot_interval_minutes)
    start_time = eda_start_time
    end_time = df['Time'].max() + snapshot_interval
    snapshot_times = pd.date_range(start=start_time , end=end_time, freq=f'{n_snapshot_interval_minutes}min')
    print(f"\nCreating snapshots for p {p}, snapshots start at {start_time}, and end at {end_time}, the min time was {df['Time'].min()}")

    # Determine file paths
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

    # Read files
    try:
        if a3:
            original_files_content = {
                'bustersAgents.py': read_file_as_lines(code_file_path_bustersagent),
                'inference.py': read_file_as_lines(code_file_path_inference)
            }
            count_syntax_passed = {
                'bustersAgents.py': 0,
                'inference.py': 0
            }
            # remove df entries that do not have either file
            df = df[(df['file_name'] == 'bustersAgents.py') | (df['file_name'] == 'inference.py')]
        else:
            original_files_content = {
                'NeuralNet.py': read_file_as_lines(code_file_path_neuralnet),
                'NeuralNetUtil.py': read_file_as_lines(code_file_path_neuralnetutil)
            }
            count_syntax_passed = {
                'NeuralNet.py': 0,
                'NeuralNetUtil.py': 0
            }
            # remove df entries that do not have either file
            df = df[(df['file_name'] == 'NeuralNet.py') | (df['file_name'] == 'NeuralNetUtil.py')]
    except FileNotFoundError:
        print(f"Warning: Files not found for participant {p}.")
        # print("File paths are ", code_file_path_bustersagent, code_file_path_inference)
        raise FileNotFoundError

    # print length of files for reference
    for file_name, content in original_files_content.items():
        print(f"Length of {file_name} is {len(content)}")

    # Save the initial version of the files : snapshot 0
    initial_snapshot_path = os.path.join(snapshot_base_dir, f'{assignment}_{p}_c0')
    counter = 1
    os.makedirs(initial_snapshot_path, exist_ok=True)
    for file_name, content in original_files_content.items():
        with open(os.path.join(initial_snapshot_path, file_name), 'w') as f:
            f.write("".join(content))
    print(f"Saved initial snapshot for participant {p}, represeting the start time of {df['Time'].min()}")

    # Iterate over the dataframe, saving snapshots in different folders
    df['snapshot'] = 0
    last_snapshot_time = start_time -  snapshot_interval

    for snapshot_counter, snapshot_time in enumerate(snapshot_times, start=1):
        print(f"\nProcessing snapshot {snapshot_counter} for participant {p} at {snapshot_time}")
        print("previous snapshot time is  ", last_snapshot_time)
                
        # Copy the original files content so you can apply all changes in order
        files_content = copy.deepcopy(original_files_content)

        # Filter changes up to the current snapshot time
        changes_up_to_snapshot = df[df['Time'] <= snapshot_time]
        changes_in_snapshot = df[(df['Time'] > last_snapshot_time) & (df['Time'] <= snapshot_time)]
        # print("Number of changes up to this snapshot time: ", len(changes_up_to_snapshot))

        # Update the snapshot column to the correct counter
        print(f"Assigning snapshot {snapshot_counter} to changes between {last_snapshot_time} and {snapshot_time}")
        print(f"Number of changes assigned: {len(df[(df['Time'] > last_snapshot_time) & (df['Time'] <= snapshot_time)])}")
       
        df.loc[
            (df['Time'] > last_snapshot_time) &
            (df['Time'] <= snapshot_time) &
            (df['snapshot'] == 0),
            'snapshot'] = snapshot_counter
        
        last_snapshot_time = snapshot_time
        
        # Skip saving if there are no changes
        if changes_in_snapshot.empty:
            # print(f"No changes detected in snapshot {snapshot_counter}")
            pass
        else:
            # print(f"Number of changes:", changes_in_snapshot.groupby('file_name').size().reset_index(name='counts'))
            pass

        # Apply changes sequentially for each file
        for index, row in changes_up_to_snapshot.iterrows():
            file_name = row['file_name']

            # Get the content changes
            try:
                files_content[file_name] = apply_change_by_lines(files_content[file_name], row)
            except Exception as e:
                print(f"Error applying change for participant {p}, file {file_name} at index {index}: with error {e}")
                print("files content length is  ", len(files_content[file_name]))
                # print("row details are ", row)
                raise e
            # files_content[file_name] = apply_change_by_lines(files_content[file_name], row)
          

        # Save the current state of the two files (even if one has not changed at all)
        snapshot_filename = f'{assignment}_{p}_c{snapshot_counter}'
        counter += 1
        snapshot_path = os.path.join(snapshot_base_dir, snapshot_filename)
        os.makedirs(snapshot_path, exist_ok=True)

        for file_name, content in files_content.items():
            with open(os.path.join(snapshot_path, file_name), 'w') as f:
                content_to_write = "".join(content)
                f.write(content_to_write)

        print(f"Saved snapshot: {snapshot_filename}")

        # check if the two files compile
        for file_name in files_content.keys():
            compiles = check_syntax(os.path.join(snapshot_path, file_name))
            if compiles:
                count_syntax_passed[file_name] += 1
            else:
                print(f"Syntax check for {file_name} in snapshot {snapshot_counter}: {'Success' if compiles else 'Failed'}")

    # Assign snapshot number to any remaining rows
    if len(df[df['snapshot'] == 0]) > 0:
        snapshot_counter += 1
        df.loc[df['snapshot'] == 0, 'snapshot'] = snapshot_counter
        # print(f"Assigned snapshot {snapshot_counter} to remaining {len(df[df['snapshot'] == snapshot_counter])} changes.")

    print(f"Syntax check results:", count_syntax_passed)

    return df


'''
Move the necessary assignment files to each snapshot directory so we can run the autograder.
- May be more efficient to move the snapshots somewhere else and run the autograder from there, but this does the job.
'''
def move_assignment_files(a3=True, code_snapshots_dir=None):

    files_to_copy_dir = 'a3_files_to_copy' if a3 else 'a4_files_to_copy'

    # Check if the directories exist
    if not os.path.isdir(code_snapshots_dir) or not os.path.isdir(files_to_copy_dir):
        print(f"Error: Directory '{code_snapshots_dir} or {files_to_copy_dir}' does not exist.")
        return

    # Get a list of snapshot directories
    snapshot_dirs = [os.path.join(code_snapshots_dir, d) for d in os.listdir(code_snapshots_dir) if os.path.isdir(os.path.join(code_snapshots_dir, d))]

    # Move files to each snapshot directory
    for snapshot_dir in snapshot_dirs:
        print(f"Processing '{snapshot_dir}'...")

        # Copy all the other code files needed for the autograder to run
        copy_files(files_to_copy_dir, snapshot_dir)

        # For a3  Move 'inference.py' and 'bustersAgents.py' into the 'tracking' folder
        target_dir = os.path.join(snapshot_dir, 'tracking') if a3 else snapshot_dir

        if a3:
            move_file_if_exists(os.path.join(snapshot_dir, 'inference.py'), target_dir)
            move_file_if_exists(os.path.join(snapshot_dir, 'bustersAgents.py'), target_dir)
        
    print("All snapshots have been processed.")


def copy_files(src_dir, dest_dir):
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
            raise e
    else:
        print(f"Warning: '{file_path}' does not exist and cannot be moved.")
        raise FileNotFoundError


""" 
Run autograder for a given question - used for parallel processing
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

        # Extract total points if the pattern exists in stdout via regex "eg. Total: 3/3"
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
    files_to_measure = ['bustersAgents.py', 'inference.py'] if a3 else ['NeuralNet.py', 'NeuralNetUtil.py']
    autograder_commands = a3_autograder_commands if a3 else a4_autograder_commands
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
        

        # Run autograder in parallel
        autograder_path = os.path.join(tracking_dir, 'autograder.py')

        if os.path.isfile(autograder_path):

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

    return df


def calc_total_cumulative_points(df, a3):
    participants = df['participant'].unique()

    for participant in participants:
        participant_df = df[df['participant'] == participant].copy()  # Use a copy to avoid warnings
        if a3:
            q_list = ['q1_points', 'q2_points', 'q3_points', 'q4_points', 'q5_points', 'q6_points', 'q7_points']
        else:
            q_list = ['q1_points', 'q2_points', 'q3_points', 'q4_points']
        # Create a new column for the rolling sum of the max values
        participant_df['total_points'] = participant_df[q_list].cummax(axis=0).sum(axis=1).cummax()
        
        # Update the original DataFrame
        df.loc[df['participant'] == participant, 'total_points'] = participant_df['total_points']

    return df

"""
Check if a file compiles , will add this info to the dataframe
"""
def check_syntax(file_path):
    try:
        py_compile.compile(file_path, doraise=True)
        # print(f"{file_path} compiled successfully. No syntax errors found.")
        return 1
    except py_compile.PyCompileError as compile_error:
        # print(f"Syntax error in {file_path}:")
        # print("Error details are ", compile_error)
        return 0
    except Exception as e:
        print(f"An unexpected error occurred while checking {file_path}:\n{e}")
        return 0
    

"""
Plot some stuff
"""
def plot_keylogger_results(df, a3, plot_lengths=False):
    participants = df['participant'].unique()
    file_1_length = 'bustersAgents.py_length' if a3 else 'NeuralNet.py_length'
    file_2_length = 'inference.py_length' if a3 else 'NeuralNetUtil.py_length'
    print(file_1_length, file_2_length)
    
    for participant in participants:
        # Filter data for the participant
        participant_df = df[df['participant'] == participant]
        # Set up the figure with two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

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
            y='total_points',
            label='Total Points Max',
            color='red',
            ax=axes[1]
        )
        axes[1].set_title(f'Total Points Earned for Participant {participant}')
        axes[1].set_xlabel('Snapshot Number')
        axes[1].set_ylabel('Total Points')
        max_points = a3_max_points if a3 else a4_max_points
        axes[1].set_ylim(0, max_points)  # Adjust Y-axis limits if needed
        axes[1].legend(loc='upper left')

        # Adjust layout for better visualization
        plt.suptitle(f'Assignment {"a3" if a3 else "a4"}, Participant {participant}')
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Add space for the main title
        
        # Show the plots
        plt.show()


def plot_syntax_checks(df, a3 ):
    participants = df['participant'].unique()
    file_1_compiles = 'bustersAgents.py_compiles' if a3 else 'NeuralNet.py_compiles'
    file_2_compiles = 'inference.py_compiles' if a3 else 'NeuralNetUtil.py_compiles'

    for participant in participants:
        # Filter data for the participant
        participant_df = df[df['participant'] == participant]
        # plot file_1

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: File lengths over snapshots
        sns.lineplot(
            data=participant_df,
            x='snapshot',
            y=file_1_compiles,
            label=file_1_compiles,
            ax=axes[0]
        )
        sns.lineplot(
            data=participant_df,
            x='snapshot',
            y=file_2_compiles,
            label=file_2_compiles,
            ax=axes[1]
        )
        axes[0].set_xlabel('Snapshot Number')
        axes[1].set_xlabel('Snapshot Number')
        axes[0].set_ylabel('File Syntax Check')
        axes[1].set_ylabel('File Syntax Check')
        axes[0].legend(loc='upper left')
        axes[1].legend(loc='upper left')

    plt.suptitle(f'Assignment {"a3" if a3 else "a4"}, Participant {participant}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Add space for the main title
    
    # Show the plots
    plt.show()

    