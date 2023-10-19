import os
import pandas as pd
import argparse
from matplotlib import pyplot as plt
import textwrap
import numpy as np
from natsort import natsorted


def split_title(title, max_line_length=30):
    # Use textwrap to split the title into lines
    lines = textwrap.wrap(title, width=max_line_length)
    # Join the lines with line breaks
    return '\n'.join(lines)

def compute_grouped_mean_std(df):
    # Group the DataFrame by the 'Iteration' column
    grouped = df.groupby('Iteration')

    # Compute the mean and standard deviation of 'MCD Mean' for each group
    result = grouped['MOS Mean'].agg(['mean', 'std']).reset_index()

    return result

def plot_mean_std_MCD(df, title=None):
    # Extract 'Iteration', 'mean', and 'std' columns from the DataFrame
    iterations = df['Iteration']
    mean_values = df['mean']
    std_values = df['std']

    # Set the width of the bars
    bar_width = 0.35

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar positions for 'mean' and 'std'
    bar_positions = range(len(iterations))
    bar_positions_std = [pos + bar_width for pos in bar_positions]

    # Create the bar plots
    plt.bar(bar_positions, mean_values, bar_width, label='MOS MCD', yerr=std_values)

    # Set the x-axis labels to 'Iteration' values
    ax.set_xticks([pos + bar_width / 2 for pos in bar_positions])
    ax.set_xticklabels(iterations, rotation=90)  # Rotate labels 90 degrees

    # Split the title into multiple lines with a maximum line length
    title = split_title(title, max_line_length=70)

    # Set labels and title
    plt.xlabel('Iteration')
    plt.ylabel('MOS Mean')
    plt.title('MOS Mean of ' + str(title))

    # Add a legend
    plt.legend()

    # Adjust margins to ensure labels are fully visible
    plt.subplots_adjust(bottom=0.2)

    # Use tight_layout to improve the plot layout
    plt.tight_layout()

    # Show the plot
    plt.show()

def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened


def plot_grouped_comparison(df_dict, keys, bar_width=0.7, group_spacing=0.7, savefig=False, savefig_path=None):
    # In case list contains lists
    keys = flatten_list(keys)

    # Create a list of unique Iteration values from the DataFrames
    iteration_values = sorted(set(iteration for df in df_dict.values() for iteration in df['Iteration']))

    # Calculate the number of keys and the number of groups
    num_keys = len(keys)
    num_groups = len(iteration_values)

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a color map for the bars
    colors = plt.cm.get_cmap('tab10', num_keys)

    for i, key in enumerate(keys):
        key_df = df_dict[key]
        mean_values_key = key_df['mean']

        # Calculate the bar positions for each group, accounting for spacing
        group_positions = np.arange(num_groups) * (num_keys * (bar_width + group_spacing)) + i * (bar_width + group_spacing)

        #TODO: Choose title to plot
        run_name = key

        if "sgv_v1" in run_name:
            sgv = "StarGAN-VC"
            topk_g = run_name.split("topk_g_")[-1].split("_")[0]
            topk_v = run_name.split("topk_v_")[-1].split("_")[0]
            topk_fi = run_name.split("topk_fi_")[-1].split("_")[0]

        elif "sgv_v2" in run_name:
            sgv = "StarGAN-VCv2"
            topk_g = run_name.split("topk_g_")[-1].split("_")[0]
            topk_v = run_name.split("topk_v_")[-1].split("_")[0]
            topk_fi = run_name.split("topk_fi_")[-1].split("_")[0]

        if "topk_False" in run_name:
            title = sgv

        else:
            title = sgv + " TopK f.i: " + topk_fi + " $\gamma$: " + topk_g + " $v$: " + topk_v

        plt.bar(group_positions, mean_values_key, bar_width, label=f'{title}', color=colors(i))

    # Set the x-axis labels to 'Iteration' values
    ax.set_xticks(np.arange(num_groups) * (num_keys * (bar_width + group_spacing)) + ((num_keys - 1) * (bar_width + group_spacing)) / 2)
    ax.set_xticklabels(iteration_values, rotation=90)

    # Set labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Mean Value MOS')

    # Add a legend outside the graph
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.15))

    # Set the title with an appropriate pad value
    #plt.title("Comparison betwen runs", pad=20)

    # Use tight_layout to improve the plot layout
    plt.tight_layout()

    # Save figure
    if savefig:
        plt.savefig(savefig_path)

    # Show the plot
    plt.show()


def plot_grouped_comparison_lines(df_dict, keys, group_spacing=0.7, savefig=False, savefig_path=None):
    # In case list contains lists
    keys = flatten_list(keys)

    # Create a list of unique Iteration values from the DataFrames
    iteration_values = sorted(set(iteration for df in df_dict.values() for iteration in df['Iteration']))

    # Calculate the number of keys and the number of groups
    num_keys = len(keys)
    num_groups = len(iteration_values)

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(15, 6))

    # Create a color map for the lines
    colors = plt.cm.get_cmap('jet', num_keys)

    for i, key in enumerate(keys):
        key_df = df_dict[key]
        mean_values_key = key_df['mean']

        # Calculate the line positions for each group, accounting for spacing
        group_positions = np.arange(num_groups) * (group_spacing + 1) + i

        # TODO: Choose title to plot
        run_name = key

        if "sgv_v1" in run_name:
            sgv = "StarGAN-VC"
            topk_g = run_name.split("topk_g_")[-1].split("_")[0]
            topk_v = run_name.split("topk_v_")[-1].split("_")[0]
            topk_fi = run_name.split("topk_fi_")[-1].split("_")[0]

        elif "sgv_v2" in run_name:
            sgv = "StarGAN-VCv2"
            topk_g = run_name.split("topk_g_")[-1].split("_")[0]
            topk_v = run_name.split("topk_v_")[-1].split("_")[0]
            topk_fi = run_name.split("topk_fi_")[-1].split("_")[0]

        if "topk_False" in run_name:
            title = sgv

        else:
            title = sgv + " TopK f.i: " + topk_fi + " $\gamma$: " + topk_g + " $v$: " + topk_v

        # Plot the mean values as a line
        plt.plot(iteration_values, mean_values_key, marker='.', label=f'{title}', color=colors(i))

    # Set the x-axis labels to 'Iteration' values
    ax.set_xticks(iteration_values)
    ax.set_xticklabels(iteration_values, rotation=90)

    # Set labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Mean Value MOS')

    y_min = 2.25
    y_max = 2.6

    plt.ylim(y_min, y_max)
    plt.grid()

    # Add a legend outside the graph
    plt.legend(loc='lower left')#, bbox_to_anchor=(1, 1.0)) #'lower center'

    # Use tight_layout to improve the plot layout
    plt.tight_layout()

    # Save figure
    if savefig:
        plt.savefig(savefig_path)

    # Show the plot
    plt.show()

def main(config):

    os.makedirs(config.figures_savedir, exist_ok=True)


    ev_results_csvs = os.listdir(config.experiment_ev_results_folder)

    dfs_dict ={}
    for ev_results_csv in ev_results_csvs:

        ev_results_csv_path = os.path.join(config.experiment_ev_results_folder, ev_results_csv)
        # Read the CSV file into a DataFrame
        df = pd.read_csv(ev_results_csv_path)
        df_grouped_by_n_iter = compute_grouped_mean_std(df)
        #plot_mean_std_MCD(df_grouped_by_n_iter, title=ev_results_csv.split("_metrics")[0])

        dfs_dict[ev_results_csv.split("_metrics")[0]] = df_grouped_by_n_iter

    # Remove original voice MOS
    original_voices_MOS = dfs_dict.pop("original_voices")

    # Sort the dictionary keys naturally
    sorted_keys = natsorted(dfs_dict.keys())

    # Create a new dictionary with sorted keys
    dfs_dict = {key: dfs_dict[key] for key in sorted_keys}

    plot_every = 4

    # Extract the keys from dfs_dict
    keys = list(dfs_dict.keys())

    # Loop to plot every 4 indices, including the first key
    for i in range(1, len(keys), plot_every):
        print(i)
        plot_name =str(i) + ".png"
        plot_grouped_comparison_lines(dfs_dict, keys=[keys[i:i + plot_every],keys[0]], savefig=True, savefig_path=os.path.join(config.figures_savedir, plot_name))



if __name__ == '__main__':

    # Folder storing the samples from the
    experiment_name = "[13_10_2023]_TopKv2_FINALEXP"
    experiment_ev_results_folder = "./evaluation_results"
    figures_savedir = "./figures_results"
    os.makedirs(figures_savedir, exist_ok=True)

    parser = argparse.ArgumentParser()

    # Directories.
    parser.add_argument('--experiment_ev_results_folder', type=str, default=os.path.join(experiment_ev_results_folder, experiment_name))
    parser.add_argument('--figures_savedir', type=str, default=os.path.join(figures_savedir, experiment_name))

    config = parser.parse_args()
    main(config)
