import utils
import random
random.seed(1984)
import tensorflow as tf
from tensorflow import keras
from model import CNN_BLSTM
import os
import argparse
import csv
from tqdm import tqdm
import numpy as np
from natsort import natsorted



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

    # Find iter folders of the run
    iter_folders = os.listdir(parent_directory)
    # Iterate over them
    for iter_folder in iter_folders:
        converted_wav_files_dict[iter_folder] = []
        iter_folder_path = os.path.join(parent_directory, iter_folder)
        # Find voice conversion folders
        vc_folders = os.listdir(iter_folder_path)
        tmp_dict = {}
        for vc_folder in vc_folders:
            tmp_dict[vc_folder] = []
            vc_folder_path = os.path.join(iter_folder_path, vc_folder)
            wav_files = [f for f in os.listdir(vc_folder_path) if f.endswith(".wav") and "-vcto-" in f]
            for wav_file in wav_files:
                tmp_dict[vc_folder].append(wav_file)
        converted_wav_files_dict[iter_folder] = tmp_dict

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

def load_mosnet(pretrained_mosnet_model):
    # init model
    print("Loading model weights")
    MOSNet = CNN_BLSTM()
    model = MOSNet.build()
    model.load_weights(pretrained_mosnet_model)

    return model

def main(config):

    # # Load MOSNet
    mosnet_model = load_mosnet(pretrained_mosnet_model=config.pretrained_mosnet_model)

    # Directories
    if config.where_exec == "slurm":
        config.gnrl_data_dir = '/workspace/NASFolder'
        output_directory = os.path.join(config.gnrl_data_dir, "output")
    elif config.where_exec == "local":
        config.gnrl_data_dir = "E:/TFM_EN_ESTE_DISCO_DURO/TFM_project/"


    # Get run name
    run_name = config.converted_samples_data_dir.split("/")[-1]

    # Get experiment name
    experiment_name = config.experiment_folder.split("/")[-1]

    # Folder where evaluation metrics will be stored
    evaluation_results_dir = "./evaluation_results"
    os.makedirs(evaluation_results_dir, exist_ok=True)
    os.makedirs(os.path.join(evaluation_results_dir, experiment_name), exist_ok=True)

    # Folder where evaluation metrics will be stored
    evaluation_results_dir = "./evaluation_results"
    os.makedirs(evaluation_results_dir, exist_ok=True)
    os.makedirs(os.path.join(evaluation_results_dir, experiment_name), exist_ok=True)

    # Get original data directory
    original_wavs_dir = os.path.join(config.gnrl_data_dir, "data/wav16") #)"data/mcs"
    original_wavs_files_dict = get_wavfiles_dict(speakers_list=config.speakers, parent_directory=original_wavs_dir)

    # Get converted data directory
    converted_wavs_dir = config.converted_samples_data_dir
    converted_wavs_files_dict = get_convertedwavfiles_dict(speakers_list=config.speakers, parent_directory=converted_wavs_dir)

    # Iterate over conversions
    metrics_org_dict = {}
    metrics_converted_dict = {}
    metrics_org_dict = dict.fromkeys(converted_wavs_files_dict.keys()) # create dict to store metrics
    metrics_converted_dict = dict.fromkeys(converted_wavs_files_dict.keys()) # create dict to store metrics
    for iters_key in tqdm(converted_wavs_files_dict.keys()):
        metrics_org_iteration_n_dict = {}
        metrics_converted_iteration_n_dict = {}
        for vc_key in tqdm(converted_wavs_files_dict[iters_key].keys()):
            converted_wavs = converted_wavs_files_dict[iters_key][vc_key]
            metrics_vc_org = []
            metrics_vc_converted = []
            for converted_wav in converted_wavs:
                wav_file_split = converted_wav.split("-")
                iteration_n = iters_key
                org_spk = wav_file_split[0][:4]
                target_spk = wav_file_split[-1][:4]
                track_id =  wav_file_split[0][-3:]

                # Derive original wavname
                original_wavname = target_spk + "_" + track_id + ".wav"
                original_wavfile_path = os.path.join(original_wavs_dir, target_spk, original_wavname) # original wav fielepath
                # Derive converted wavname
                converted_wavfile_path = os.path.join(converted_wavs_dir, iters_key, vc_key, converted_wav)

                # MOS Original #
                # Get mel spectrogram for original wavfile
                mag_sgram_org = utils.get_spectrograms(original_wavfile_path)
                timestep_org = mag_sgram_org.shape[0]
                mag_sgram_org = np.reshape(mag_sgram_org, (1, timestep_org, utils.SGRAM_DIM))

                # Compute MOS of the original wavfiles
                mos_score_org, mos_frame_score_org = mosnet_model.predict(mag_sgram_org, verbose=0, batch_size=1)

                # MOS Converted #
                # Get mel spectrogram for converted wavfile
                mag_sgram_converted = utils.get_spectrograms(converted_wavfile_path)
                timestep_converted = mag_sgram_converted.shape[0]
                mag_sgram_converted = np.reshape(mag_sgram_converted, (1, timestep_converted, utils.SGRAM_DIM))

                # Compute MOS of the original wavfiles
                mos_score_converted, mos_frame_score_converted = mosnet_model.predict(mag_sgram_converted, verbose=0, batch_size=1)

                # Log MOS original and convrted
                metrics_vc_org.append(float(mos_score_org.squeeze()))
                metrics_vc_converted.append(float(mos_score_converted.squeeze()))

            # Compute mean metrics and add to metrics dict
            metrics_org_iteration_n_dict[vc_key] = np.mean(np.array(metrics_vc_org)), np.std(np.array(metrics_vc_org))
            metrics_converted_iteration_n_dict[vc_key] = np.mean(np.array(metrics_vc_converted)), np.std(np.array(metrics_vc_converted))

        metrics_org_dict[iters_key] = metrics_org_iteration_n_dict
        metrics_converted_dict[iters_key] = metrics_converted_iteration_n_dict

    print("hola")

    ############
    # Original #
    ###########
    # Sort n_iterations keys naturally
    sorted_keys_org = natsorted(metrics_org_dict.keys())
    # Create a new ordered dictionary using the sorted keys
    metrics_org_dict = dict((key, metrics_org_dict[key]) for key in sorted_keys_org)

    # Specify the CSV file path
    csv_file_org = os.path.join(evaluation_results_dir, experiment_name, 'original_voices_metrics.csv')

    # Open the CSV file in write mode
    with open(csv_file_org, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)

        # Write the header row
        header = ['Iteration', 'Conversion', 'MOS Mean', 'MOS STD']
        csv_writer.writerow(header)

        # Write data to the CSV file
        for iteration, metrics_data in metrics_org_dict.items():
            for conversion, (value1, value2) in metrics_data.items():
                row = [iteration, conversion, value1, value2]
                csv_writer.writerow(row)

        #############
        # Converted #
        #############

        # Sort n_iterations keys naturally
        sorted_keys_converted = natsorted(metrics_converted_dict.keys())
        # Create a new ordered dictionary using the sorted keys
        metrics_converted_dict = dict((key, metrics_converted_dict[key]) for key in sorted_keys_converted)

        # Specify the CSV file path
        csv_file_converted = os.path.join(evaluation_results_dir, experiment_name,
                                run_name.split("\\")[-1][:len(run_name.split("\\")[-1]) // 2] + '_metrics.csv')

        # Open the CSV file in write mode
        with open(csv_file_converted, 'w', newline='') as csvfile:
            # Create a CSV writer object
            csv_writer = csv.writer(csvfile)

            # Write the header row
            header = ['Iteration', 'Conversion', 'MOS Mean', 'MOS STD']
            csv_writer.writerow(header)

            # Write data to the CSV file
            for iteration, metrics_data in metrics_converted_dict.items():
                for conversion, (value1, value2) in metrics_data.items():
                    row = [iteration, conversion, value1, value2]
                    csv_writer.writerow(row)


    # # Sort the keys numerically
    # for vcto_conversion in metrics_dict.keys():
    #     metrics_dict_vcto_sorted_keys = sorted(metrics_dict[vcto_conversion].keys(), key=lambda x: "{:0>10}".format(x)) # sort keys numerically naturally
    #     metrics_dict[vcto_conversion] = {key: metrics_dict[vcto_conversion][key] for key in metrics_dict_vcto_sorted_keys}

    # Plot the metrics in a bar diagram
    # plot_metrics_dict(metrics_dict, title=run_name, show_plot=True)
    # print("hola")


if __name__ == "__main__":

    # Folder storing the samples from the
    #experiment_folder = "Z:/Shared_PFC-TFG-TFM/Claudio/TOPK_VC/output/samples/[5_10_2023]_TopK_v1"
    # experiment_folder = "./converted/[9_10_2023]_TopK_v1"
    experiment_folder = "C:/Users/clferma1/Documents/Investigacion_GIT/VoiceConversionTopK/converted/[13_10_2023]_TopKv1_FINALEXP"
    runs_folders = os.listdir(experiment_folder)

    for run_folder in runs_folders:

        parser = argparse.ArgumentParser()

        # Directories.
        parser.add_argument("--pretrained_mosnet_model", default="./pre_trained/cnn_blstm.h5", type=str,
                            help="pretrained MOSNet model file")
        parser.add_argument('--gnrl_data_dir', type=str, default='.')
        parser.add_argument('--experiment_folder', type=str, default=experiment_folder)
        parser.add_argument('--converted_samples_data_dir', type=str, default=os.path.join(experiment_folder, run_folder))
        parser.add_argument('--where_exec', type=str, default='local', help="slurm or local") # "slurm", "local"
        parser.add_argument('--speakers', type=str, nargs='+', required=False, help='Speaker dir names.',
                            default=["p272", "p262", "p232", "p229"])

        config = parser.parse_args()
        main(config)

    # Folder storing the samples from the
    #experiment_folder = "Z:/Shared_PFC-TFG-TFM/Claudio/TOPK_VC/output/samples/[5_10_2023]_TopK_v1"
    # experiment_folder = "./converted/[9_10_2023]_TopK_v1"
    experiment_folder = "C:/Users/clferma1/Documents/Investigacion_GIT/VoiceConversionTopK/converted/[13_10_2023]_TopKv2_FINALEXP"
    runs_folders = os.listdir(experiment_folder)

    for run_folder in runs_folders:

        parser = argparse.ArgumentParser()

        # Directories.
        parser.add_argument("--pretrained_mosnet_model", default="./pre_trained/cnn_blstm.h5", type=str,
                            help="pretrained MOSNet model file")
        parser.add_argument('--gnrl_data_dir', type=str, default='.')
        parser.add_argument('--experiment_folder', type=str, default=experiment_folder)
        parser.add_argument('--converted_samples_data_dir', type=str, default=os.path.join(experiment_folder, run_folder))
        parser.add_argument('--where_exec', type=str, default='local', help="slurm or local") # "slurm", "local"
        parser.add_argument('--speakers', type=str, nargs='+', required=False, help='Speaker dir names.',
                            default=["p272", "p262", "p232", "p229"])

        config = parser.parse_args()
        main(config)

    experiment_folder = "C:/Users/clferma1/Documents/Investigacion_GIT/VoiceConversionTopK/converted/[8_10_2023]_TopK_v2"
    runs_folders = os.listdir(experiment_folder)

    for run_folder in runs_folders:

        parser = argparse.ArgumentParser()

        # Directories.
        parser.add_argument("--pretrained_mosnet_model", default="./pre_trained/cnn_blstm.h5", type=str,
                            help="pretrained MOSNet model file")
        parser.add_argument('--gnrl_data_dir', type=str, default='.')
        parser.add_argument('--experiment_folder', type=str, default=experiment_folder)
        parser.add_argument('--converted_samples_data_dir', type=str, default=os.path.join(experiment_folder, run_folder))
        parser.add_argument('--where_exec', type=str, default='local', help="slurm or local") # "slurm", "local"
        parser.add_argument('--speakers', type=str, nargs='+', required=False, help='Speaker dir names.',
                            default=["p272", "p262", "p232", "p229"])

        config = parser.parse_args()
        main(config)