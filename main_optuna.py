import optuna
import os
import argparse
from optuna_solver import Solver, SolverCustom
from data_loader import get_loader, TestDataset
from torch.backends import cudnn
import mlflow

def str2bool(v):
    return v.lower() in 'true'

def objective(trial):
    # Define the search space for hyperparameters
    batch_size = trial.suggest_int("batch_size", 8, 128, log=True)  # Adjust the range as needed
    topk_gamma = trial.suggest_float("topk_gamma", 0.9, 1.0)  # Adjust the range as needed
    topk_v = trial.suggest_int("topk_v", 1, batch_size)  # Adjust the range as needed
    g_lr = trial.suggest_float("g_lr", 1e-5, 0.1, log=True)  # Adjust the range as needed
    d_lr = trial.suggest_float("d_lr", 1e-5, 0.1, log=True)  # Adjust the range as needed
    batch_size = trial.suggest_int("batch_size", 8, 128, log=True)  # Adjust the range as needed
    num_iters = trial.suggest_int("num_iters", 10000, 100000)  # Adjust the range as needed
    #num_iters = 5  # Adjust the range as needed
    num_iters_decay = trial.suggest_int("num_iters_decay",  10000, 80000)  # Adjust the range as needed
    topk_from_iter = trial.suggest_int("topk_from_iter", 10000, 90000)  # Adjust the range as needed
    lambda_rec = trial.suggest_float("lambda_rec", 1e-2, 10.0, log=True)  # Adjust the range as needed
    lambda_gp = trial.suggest_float("lambda_gp", 1e-2, 10.0, log=True)  # Adjust the range as needed
    lambda_id = trial.suggest_float("lambda_id", 1e-2, 10.0, log=True)  # Adjust the range as needed
    lambda_cls = trial.suggest_float("lambda_cls", 1e-2, 10.0, log=True)  # Adjust the range as needed
    beta1 = trial.suggest_float("beta1", 0.0, 1.0)  # Adjust the range as needed
    beta2 = trial.suggest_float("beta2", 0.0, 1.0)  # Adjust the range as needed
    lr_update_step = trial.suggest_int("lr_update_step", 5000, 20000)  # Adjust the range as needed

    # Train the model with the selected hyperparameters
    parser = argparse.ArgumentParser()

    # StarGAN version configuration.
    parser.add_argument('--stargan_version', type=str, default='v1', help='Version of StarGAN-VC')  # "v1", "v2"

    # TopK configuration.
    parser.add_argument('--topk_training', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="Enable TopK training.")
    parser.add_argument('--topk_gamma', type=float, default=topk_gamma, help='K decay in TopK')
    parser.add_argument('--topk_v', type=float, default=topk_v, help='minimum percentage of batch size for K')
    parser.add_argument('--topk_from_iter', type=int, default=topk_from_iter,
                        help='iteration for starting to apply TopK Training')

    # Model configuration.
    parser.add_argument('--lambda_rec', type=float, default=lambda_rec, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=lambda_gp, help='weight for gradient penalty')
    parser.add_argument('--lambda_id', type=float, default=lambda_id, help='weight for id mapping loss')
    parser.add_argument('--lambda_cls', type=float, default=lambda_cls,
                        help=' (StarGANv1)weight for domain classification loss')
    parser.add_argument('--sampling_rate', type=int, default=16000, help='sampling rate')

    # Training configuration.
    parser.add_argument('--mlflow_experiment_name', type=str, default="TopK HPSearch V2")
    parser.add_argument('--preload_data', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="Load data on RAM memory.")
    parser.add_argument('--batch_size', type=int, default=batch_size, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=num_iters, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=num_iters_decay, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=g_lr, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=d_lr, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=beta1, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=beta2, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=100000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # Directories.
    parser.add_argument('--gnrl_data_dir', type=str, default='.')
    parser.add_argument('--where_exec', type=str, default='local', help="slurm or local")  # "slurm", "local"
    parser.add_argument('--speakers', type=str, nargs='+', required=False, help='Speaker dir names.',
                        default=['p262', 'p272', 'p229', 'p232'])

    # Step size.
    parser.add_argument('--log_step', type=int, default=100) # 10
    parser.add_argument('--sample_step', type=int, default=10000)  # 10000
    parser.add_argument('--model_save_step', type=int, default=10000)  # 10000
    parser.add_argument('--lr_update_step', type=int, default=lr_update_step)

    config = parser.parse_args()

    # no. of spks
    config.num_speakers = len(config.speakers)

    if len(config.speakers) < 2:
        raise RuntimeError("Need at least 2 speakers to convert audio.")

    print(config)

    # For fast training.
    cudnn.benchmark = True

    print("Directories:", os.listdir(os.getcwd()))
    # Create needed directories
    if config.where_exec == "slurm":
        config.gnrl_data_dir = '/workspace/NASFolder'
        output_directory = os.path.join(config.gnrl_data_dir, "output")
    elif config.where_exec == "local":
        config.gnrl_data_dir = "E:/TFM_EN_ESTE_DISCO_DURO/TFM_project/"
        output_directory = os.path.join("./output")

    train_data_dir = os.path.join(config.gnrl_data_dir, "data/mc/train")
    test_data_dir = os.path.join( config.gnrl_data_dir, "data/mc/test")
    wav_dir = os.path.join( config.gnrl_data_dir, "data/wav16")

    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(os.path.join(output_directory, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_directory, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_directory, "samples"), exist_ok=True)


    log_dir = os.path.join(output_directory, "logs", config.mlflow_experiment_name)
    model_save_dir = os.path.join(output_directory, "models", config.mlflow_experiment_name)
    sample_dir = os.path.join(output_directory, "samples", config.mlflow_experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # MlFlow Parameters
    mlruns_folder = os.path.join(output_directory, "mlruns")# "./mlruns"
    mlflow_experiment_name = config.mlflow_experiment_name
    key_substitutions = {
        'stargan_version': 'sgv',
        'sampling_rate': 'sr',
        'topk_training': 'topk',
        'topk_gamma': 'topk_g',
        'topk_from_iter': 'topk_fi',
        'lambda_rec': 'lbd_rec',
        'lambda_gp': 'lbd_rec',
        'lambda_cls': 'lbd_cls',
        'batch_size':'bs',
        'num_iters': 'iters',
        'num_iters_decay': 'iters_dec',
        'beta1' : 'b1',
        'beta2': 'b2'
    }
    config_dict = vars(config)
    # mlflow_run_name = "_".join([f"{key_substitutions.get(key, key)}_{value}".replace(":", "_")
    #                             for key, value in config_dict.items()
    #                             if "dir" not in key and "speakers" not in key
    #                             and "preload_data" not in key and "mlflow_experiment_name" not in key]).split("_resume_iters")[0]

    mlflow_run_name = "_".join([f"{key_substitutions.get(key, key)}_{value:.3f}" if isinstance(value,
                                                                                               float) else f"{key_substitutions.get(key, key)}_{value}".replace(
        ":", "_")
                                for key, value in config_dict.items()
                                if "dir" not in key and "speakers" not in key
                                and "preload_data" not in key and "mlflow_experiment_name" not in key]).split(
        "_resume_iters")[0]

    mlflow.set_tracking_uri(mlruns_folder)
    experiment = mlflow.set_experiment(mlflow_experiment_name)
    mlflow.start_run(run_name=mlflow_run_name)
    # Log Parameters
    for key, value in vars(config).items():
        mlflow.log_param(key, value)

    # Create savedir subdirectories for current run
    config.log_dir = os.path.join(log_dir, mlflow_run_name)
    config.model_save_dir = os.path.join(model_save_dir, mlflow_run_name)
    config.sample_dir = os.path.join(sample_dir, mlflow_run_name)

    print(os.getcwd())
    print(train_data_dir)
    print(test_data_dir)
    print(wav_dir)

    # Create directories if not exist.
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)

    # TODO: remove hard coding of 'test' speakers
    src_spk = config.speakers[0]
    trg_spk = config.speakers[1]

    # Data loader.
    train_loader = get_loader(config.speakers, train_data_dir, config.batch_size, 'train',
                              num_workers=config.num_workers, preload_data=config.preload_data )
    # TODO: currently only used to output a sample whilst training
    test_loader = TestDataset(config.speakers, test_data_dir, wav_dir, src_spk=src_spk, trg_spk=trg_spk)

    # Solver for training and testing StarGAN.
    if config.preload_data:
        solver = SolverCustom(train_loader, test_loader, config, trial)
    else:
        solver = Solver(train_loader, test_loader, config)

    # Train
    if config.mode == 'train':
        score = solver.train()

    # Return the score to Optuna for optimization
    return score


# Create an Optuna study with pruning enabled (MedianPruner)

study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())

# Set the number of optimization trials
n_trials = 100  # Adjust the number of trials as needed

# Enable parallel execution by specifying the number of jobs (threads)
n_jobs = 1 # Adjust the number of parallel jobs as needed

# Run the optimization with parallelization
study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

# Get the best hyperparameters and their values
best_params = study.best_params
best_score = study.best_value

# Print the best hyperparameters and score
print("Best Hyperparameters:", best_params)
print("Best Score:", best_score)

# Save the study results as a JSON file
study.trials_dataframe().to_json("study_results.json", orient="split")

# Save the study results as a CSV file
study.trials_dataframe().to_csv("study_results.csv", index=False)
