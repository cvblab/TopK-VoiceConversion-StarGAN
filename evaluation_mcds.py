import argparse
from mel_cepstral_distance import get_metrics_wavs, get_metrics_mels
import os
import csv
import glob
import numpy as np
from tqdm import tqdm
from utils import *
#import matplotlib.pyplot as plt


def get_wavfiles_dict(speakers_list, parent_directory):
    # Create an empty dictionary to store the results
    wav_files_dict = {}

    # Iterate through the list of speakers
    for speaker_id in speakers_list:
        # Construct the path to the speaker's subfolder
        speaker_folder = os.path.join(parent_directory, speaker_id)

        # Check if the folder exists
        if os.path.exists(speaker_folder) and os.path.isdir(speaker_folder):
            # List all the WAV files in the speaker's subfolder
            wav_files = [f for f in os.listdir(speaker_folder) if f.endswith(".wav")]

            # Store the list of WAV files in the dictionary with the speaker ID as the key
            wav_files_dict[speaker_id] = wav_files
        else:
            print(f"Folder not found for speaker {speaker_id}")

    return wav_files_dict

def get_convertedwavfiles_dict(speakers_list, parent_directory):
    converted_wav_files_dict = {}

    # List all the WAV files in the parent directory
    wav_files = [f for f in os.listdir(parent_directory) if f.endswith(".wav")]

    tmp_dict = {}
    # Iteratre over the WAV files
    for wav_file in wav_files:
        if "cpsyn" not in wav_file:
            wav_file_split = wav_file.split("-")
            iteration_n = wav_file_split[0]
            org_spk = wav_file_split[1][:4]
            target_spk = wav_file_split[-1][:4]

            # Check if iteration_n key exists in tmp_dict
            if iteration_n in tmp_dict:
                # Append the wav_file to the existing list
                tmp_dict[iteration_n].append(wav_file)
            else:
                # Create a new list and add the wav_file to it
                tmp_dict[iteration_n] = [wav_file]

            converted_wav_files_dict[org_spk + "-vcto-" + target_spk] = tmp_dict


    return  converted_wav_files_dict

def plot_metrics_dict(metrics_dict, title, show_plot=True):
    # Extract the iteration numbers and their corresponding mean MCD and STD values
    iterations = list(metrics_dict['p262-vcto-p272'].keys())
    mean_mcd_values = [value[0] for value in metrics_dict['p262-vcto-p272'].values()]
    std_values = [value[1] for value in metrics_dict['p262-vcto-p272'].values()]

    # Create a bar plot
    fig, ax = plt.subplots()
    width = 0.35
    x = range(len(iterations))

    # Plot the mean MCD values
    ax.bar(x, mean_mcd_values, width, label='Mean MCD')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Mean MCD')

    # Plot the STD values as error bars
    ax.errorbar(x, mean_mcd_values, yerr=std_values, fmt='.k', label='STD', color="black")

    # Rotate X-axis labels vertically
    plt.xticks(x, iterations, rotation='vertical')
    ax.legend()

    # Display the plot
    # Create a title and split it into multiple lines if too lone
    max_line_length = 60  # Adjust this value as needed
    title_lines = [title[i:i + max_line_length] for i in range(0, len(title), max_line_length)]
    title_text = '\n'.join(title_lines)
    plt.title(title_text)

    plt.tight_layout()
    plt.show()


def main(config):

    # Directories
    if config.where_exec == "slurm":
        config.gnrl_data_dir = '/workspace/NASFolder'
        output_directory = os.path.join(config.gnrl_data_dir, "output")
    elif config.where_exec == "local":
        config.gnrl_data_dir = "E:/TFM_EN_ESTE_DISCO_DURO/TFM_project/"

    # Get run name
    run_name = config.converted_samples_data_dir.split("/")[-1]

    # Get original data directory
    original_wavs_dir = os.path.join(config.gnrl_data_dir, "data/wav16") #)"data/mcs"
    original_wavs_files_dict = get_wavfiles_dict(speakers_list=config.speakers, parent_directory=original_wavs_dir)

    # Get converted data directory
    converted_wavs_dir = config.converted_samples_data_dir
    converted_wavs_files_dict = get_convertedwavfiles_dict(speakers_list=config.speakers, parent_directory=converted_wavs_dir)

    # Iterate over conversions
    metrics_dict = {}
    metrics_dict = dict.fromkeys(converted_wavs_files_dict.keys()) # create dict to store metrics
    for vcto_conversion in converted_wavs_files_dict.keys():
        # Order dict first

        print("Computing for vcto: " + str(vcto_conversion))
        metrics_dict[vcto_conversion] = dict.fromkeys(converted_wavs_files_dict[vcto_conversion].keys())

        for iteration_n, iteration_n_files in tqdm(converted_wavs_files_dict[vcto_conversion].items()):
            print("-- Iteration: " + iteration_n, iteration_n_files)
            metrics_iteration_n = []
            for converted_wav in iteration_n_files:
                wav_file_split = converted_wav.split("-")
                iteration_n = wav_file_split[0]
                org_spk = wav_file_split[1][:4]
                target_spk = wav_file_split[-1][:4]
                track_id =  wav_file_split[1][-3:]
                converted_wavfile_path = os.path.join(converted_wavs_dir, converted_wav) # converted wav filepath

                original_wavname = target_spk + "_" + track_id + ".wav"
                original_wavfile_path = os.path.join(original_wavs_dir, target_spk, original_wavname) # original wav fielepath

                # Compute MCD
                mcd, penalty, frames = get_metrics_wavs(wav_file_1=original_wavfile_path,
                                                        wav_file_2=converted_wavfile_path,
                                                        n_mfcc=36)
                # Append to compute mean MCD
                metrics_iteration_n.append(mcd)

                print("MCD between " + original_wavname + " and " + converted_wav)
                print(mcd, penalty, frames)

            #Compute mean metrics and add to metrics dict
            metrics_dict[vcto_conversion][iteration_n] = np.mean(np.array(metrics_iteration_n)), np.std(np.array(metrics_iteration_n))


    # Sort the keys numerically
    for vcto_conversion in metrics_dict.keys():
        metrics_dict_vcto_sorted_keys = sorted(metrics_dict[vcto_conversion].keys(), key=lambda x: "{:0>10}".format(x)) # sort keys numerically naturally
        metrics_dict[vcto_conversion] = {key: metrics_dict[vcto_conversion][key] for key in metrics_dict_vcto_sorted_keys}

    # Plot the metrics in a bar diagram
    plot_metrics_dict(metrics_dict, title=run_name, show_plot=True)
    print("hola")

    # Convereted da



if __name__ == '__main__':

    # Folder storing the samples from the
    #experiment_folder = "Z:/Shared_PFC-TFG-TFM/Claudio/TOPK_VC/output/samples/[5_10_2023]_TopK_v1"
    experiment_folder = "./converted/[8_10_2023]_TopK_v1"

    runs_folders = os.listdir(experiment_folder)

    for run_folder in runs_folders:

        parser = argparse.ArgumentParser()

        # Directories.
        parser.add_argument('--gnrl_data_dir', type=str, default='.')
        parser.add_argument('--converted_samples_data_dir', type=str, default=os.path.join(experiment_folder, run_folder))
        parser.add_argument('--where_exec', type=str, default='local', help="slurm or local") # "slurm", "local"
        parser.add_argument('--speakers', type=str, nargs='+', required=False, help='Speaker dir names.',
                            default=["p272", "p262", "p232", "p229"])

        config = parser.parse_args()
        main(config)

