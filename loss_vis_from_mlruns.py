import os
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def resample_data(steps, values, target_length):
    """
    Resample data by taking the mean of adjacent data points to reduce the number of points.

    Parameters:
    - steps (list): List of iteration or epoch numbers.
    - values (list): List of metric values corresponding to each step.
    - target_length (int): The desired length of the resampled data.

    Returns:
    - resampled_steps (list): Resampled iteration or epoch numbers.
    - resampled_values (list): Resampled metric values.
    """
    factor = len(steps) // target_length
    if factor <= 1:
        return steps, values

    resampled_steps = []
    resampled_values = []

    for i in range(0, len(steps), factor):
        resampled_steps.append(np.mean(steps[i:i + factor]))
        resampled_values.append(np.mean(values[i:i + factor]))

    return resampled_steps, resampled_values

def get_run_name(experiment_dir, experiment_id, run_id):
    """
    Get the run name from the "mlflow.runName" file.

    Parameters:
    - experiment_id (str): The ID of the MLflow experiment.
    - run_id (str): The ID of the MLflow run.

    Returns:
    - run_name (str): The run name extracted from the "mlflow.runName" file.
    """
    # experiment_dir = os.path.join("\mlruns", experiment_id)
    run_dir = os.path.join(experiment_dir, run_id)
    tags_dir = os.path.join(run_dir, 'tags')
    run_name_file = os.path.join(tags_dir, 'mlflow.runName')

    with open(run_name_file, 'r') as file:
        run_name = file.read().strip()

    return run_name

def get_params(experiment_id, run_id):
    """
    Get parameter values from the "params" folder.

    Parameters:
    - experiment_id (str): The ID of the MLflow experiment.
    - run_id (str): The ID of the MLflow run.

    Returns:
    - params (dict): A dictionary containing parameter names and their values.
    """
    experiment_dir = os.path.join("\mlruns", experiment_id)
    run_dir = os.path.join(experiment_dir, run_id)
    params_dir = os.path.join(run_dir, 'params')

    params = {}
    for param_name in os.listdir(params_dir):
        param_file = os.path.join(params_dir, param_name)
        with open(param_file, 'r') as file:
            param_value = file.read().strip()
        params[param_name] = param_value

    return params

def load_metrics(experiment_dir, experiment_id, run_id, metric_name):
    """
    Load metric values from the specified metric file.

    Parameters:
    - experiment_id (str): The ID of the MLflow experiment.
    - run_id (str): The ID of the MLflow run.
    - metric_name (str): The name of the metric file to load.

    Returns:
    - steps (list): List of iteration or epoch numbers.
    - values (list): List of metric values corresponding to each step.
    """
    # experiment_dir = os.path.join("\mlruns", experiment_id)
    run_dir = os.path.join(experiment_dir, run_id)
    metrics_dir = os.path.join(run_dir, 'metrics')
    metric_file = os.path.join(metrics_dir, metric_name)

    steps, values = [], []
    try:
        with open(metric_file, 'r') as file:
            for line in file:
                # Parse the line and extract step and value
                parts = line.strip().split()
                if len(parts) >= 2:
                    step = int(parts[2])  # Assuming the third item is the step/epoch number
                    value = float(parts[1])  # Assuming the second item is the metric value
                    steps.append(step)
                    values.append(value)
    except FileNotFoundError:
        print("not Kvalue logged metric")
        steps = [0]
        values = [0]

    return steps, values

def plot_learning_curve_from_mlflow(experiment_dir, experiment_id, run_id, target_length, figures_save_dir, metrics=None, loss_terms=None, title='Learning Curve'):
    os.makedirs(os.path.join(figures_save_dir), exist_ok=True)
    
    # Split the run name into lines if it's too long
    max_title_length = 100  # Maximum characters per line
    split_run_name = [run_name[i:i + max_title_length] for i in range(0, len(run_name), max_title_length)]
    #title = "\n".join(split_run_name)  # Join lines with line breaks

    #plt.plot(steps, metric_values, marker='o', linestyle='-', label=metric.capitalize())

    if loss_terms:
        # Create the learning curve plot for the specified metric
        plt.figure(figsize=(10, 6))

        for loss_term in loss_terms:

            # Load and plot the loss term data
            loss_steps_org, loss_values_org = load_metrics(experiment_dir, experiment_id, run_id, loss_term)

            #if len(loss_steps_org)==40000: # Finished training
            # Resample the data to reduce the number of points
            loss_steps_r, loss_values_r = resample_data(loss_steps_org, loss_values_org, target_length)

            plt.plot(loss_steps_r, loss_values_r, marker='', linestyle='-', label=loss_term)

        if metrics:
            for metric in metrics:
                metric_steps_org, metric_values_org = load_metrics(experiment_dir, experiment_id, run_id, metric)
                plt.plot(metric_steps_org, metric_values_org, marker='', linestyle='-', label=metric)

        #title = "StarGAN-VC Training Losses"
        plt.title(title)
        plt.xlabel('Iteration')
        #plt.ylabel(metric.capitalize())
        plt.grid(True)
        plt.legend()


        os.makedirs(os.path.join(figures_save_dir, "losses"), exist_ok=True)
        plt.savefig(os.path.join(figures_save_dir, "losses", run_name + ".png"))

        plt.show()

        print("hola")

# Replace '12' with your experiment ID and 'Your_Run_Name' with the actual run name.
experiment_id = '1'

# Specify the experiment directory
experiment_dir = os.path.join("Z:/Shared_PFC-TFG-TFM/Claudio/TOPK_VC/output_final/mlruns/", experiment_id)  # Update this to your experiment directory path
#run_name = 'Your_Run_Name'
metrics_to_plot = ['K_value']  # Add other metrics if needed
#loss_terms_to_plot = ['G/loss']  # Add loss terms to plot
loss_terms_to_plot = ['G/loss']  # Add loss terms to plot
#loss_terms_to_plot = ['D/loss_fake', 'D/loss_']  # Add loss terms to plot
target_length = 500

run_ids_list = os.listdir(experiment_dir)

# Create a dictionary to store run names and their corresponding run IDs
run_name_to_ids = {}

for run_id in run_ids_list:
    if run_id != "meta.yaml":
        run_name = get_run_name(experiment_dir, experiment_id, run_id)

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
            plot_learning_curve_from_mlflow(experiment_dir=experiment_dir,
                                            experiment_id=experiment_id,
                                            run_id=run_id,
                                            metrics=None,
                                            target_length=target_length,
                                            loss_terms=loss_terms_to_plot,
                                            title=sgv + " Training Losses",
                                            figures_save_dir="./figures")
        elif "topk_True" in run_name:
            plot_learning_curve_from_mlflow(experiment_dir=experiment_dir,
                                            experiment_id=experiment_id,
                                            run_id=run_id,
                                            metrics=metrics_to_plot,
                                            target_length=target_length,
                                            loss_terms=loss_terms_to_plot,
                                            title=sgv + " TopK f.i: " + topk_fi + " $\gamma$: " + topk_g +  " $v$: " + topk_v  , #+ " Training Losses",
                                            figures_save_dir="./figures")

        print(run_name)