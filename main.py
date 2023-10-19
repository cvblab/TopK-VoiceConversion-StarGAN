import os
import argparse
from solver import Solver, SolverCustom
from data_loader import get_loader, TestDataset
import torch
import mlflow
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def str2bool(v):
    return v.lower() in 'true'

def main(config):
    # Set seeds for reproducibility
    set_seed(45)

    print("Directories:", os.listdir(os.getcwd()))
    # Create needed directories
    if config.where_exec == "slurm":
        config.gnrl_data_dir = '/workspace/NASFolder'
        output_directory = os.path.join(config.gnrl_data_dir, "output_final")
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
    mlflow_run_name = "_".join([f"{key_substitutions.get(key, key)}_{value}".replace(":", "_")
                                for key, value in config_dict.items()
                                if "dir" not in key and "speakers" not in key
                                and "preload_data" not in key and "mlflow_experiment_name" not in key]).split("_resume_iters")[0]

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
        solver = SolverCustom(train_loader, test_loader, config)
    else:
        solver = Solver(train_loader, test_loader, config)

    # Train
    if config.mode == 'train':
        solver.train()

    # elif config.mode == 'test':
    #     solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # StarGAN version configuration.
    parser.add_argument('--stargan_version', type=str, default='v2', help='Version of StarGAM-VC') # "v1", "v2"

    # TopK configuration.
    parser.add_argument('--topk_training', default=True, type=lambda x: (str(x).lower() == 'true'), help="Enable TopK training.")
    parser.add_argument('--topk_gamma', type=float, default=0.9999, help='K decay in TopK')
    parser.add_argument('--topk_v', type=float, default=0.5, help='minimum percentage of batch size for K')
    parser.add_argument('--topk_from_iter', type=int, default=25000, help='iteration for starting to apply TopK Training')

    # Model configuration.
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=5, help='weight for gradient penalty')
    parser.add_argument('--lambda_id', type=float, default=5, help='weight for id mapping loss')
    parser.add_argument('--lambda_cls', type=float, default=10, help=' (StarGANv1)weight for domain classification loss')
    parser.add_argument('--sampling_rate', type=int, default=16000, help='sampling rate')

    # Training configuration.
    parser.add_argument('--mlflow_experiment_name', type=str, default="[10_10_2023]_TopKv2KPASA", help='Name for experiment in MLFlow')
    parser.add_argument('--preload_data', default=True, type=lambda x: (str(x).lower() == 'true'), help="Load data on RAM memory.")
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=50000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=100000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=200)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # Directories.
    parser.add_argument('--gnrl_data_dir', type=str, default='.')
    parser.add_argument('--where_exec', type=str, default='local', help="slurm or local") # "slurm", "local"
    parser.add_argument('--speakers', type=str, nargs='+', required=False, help='Speaker dir names.',
                        default= ['p262', 'p272', 'p229', 'p232', 'p292', 'p293', 'p360', 'p361', 'p248', 'p251'])

    # Step size.
    parser.add_argument('--log_step', type=int, default=5000) #10
    parser.add_argument('--sample_step', type=int, default=10000) #10000
    parser.add_argument('--model_save_step', type=int, default=10000) #10000
    parser.add_argument('--lr_update_step', type=int, default=10000)

    config = parser.parse_args()

    # no. of spks
    config.num_speakers = len(config.speakers)

    if len(config.speakers) < 2:
        raise RuntimeError("Need at least 2 speakers to convert audio.")

    print(config)
    main(config)

