import argparse
import os
from natsort import natsorted
import torch
import librosa
import soundfile as sf
from model_v1 import *
from model_v2 import *
import glob
from utils import world_decompose, pitch_conversion, world_encode_spectral_envelop, world_speech_synthesis, wav_padding
from shutil import copy
from tqdm import tqdm

def load_wav(wavfile, sr=16000):
    wav, _ = librosa.load(wavfile, sr=sr, mono=True)
    return wav_padding(wav, sr=sr, frame_period=5, multiple=4)

class ConvertDataset(object):
    """Dataset for conversion."""
    def __init__(self, config, src_spk, trg_spk, train_data_dir, wav_dir):
        speakers = config.speakers
        spk2idx = dict(zip(speakers, range(len(speakers))))
        assert trg_spk in speakers, f'The trg_spk should be chosen from {speakers}, but you choose {trg_spk}.'

        self.src_spk = src_spk
        self.trg_spk = trg_spk

        # Source speaker locations.
        self.src_spk_stats = np.load(os.path.join(train_data_dir, f'{self.src_spk}_stats.npz'))
        self.src_wav_dir = f'{wav_dir}/{self.src_spk}'
        self.trg_wav_dir = f'{wav_dir}/{self.trg_spk}'
        self.src_wav_files = sorted(glob.glob(os.path.join(self.src_wav_dir, '*.wav')))
        self.trg_wav_files = sorted(glob.glob(os.path.join(self.trg_wav_dir, '*.wav')))

        # Target speaker locations.
        self.trg_spk_stats = np.load(os.path.join(train_data_dir, f'{self.trg_spk}_stats.npz'))

        self.logf0s_mean_src = self.src_spk_stats['log_f0s_mean']
        self.logf0s_std_src = self.src_spk_stats['log_f0s_std']
        self.logf0s_mean_trg = self.trg_spk_stats['log_f0s_mean']
        self.logf0s_std_trg = self.trg_spk_stats['log_f0s_std']
        self.mcep_mean_src = self.src_spk_stats['coded_sps_mean']
        self.mcep_std_src = self.src_spk_stats['coded_sps_std']
        self.mcep_mean_trg = self.trg_spk_stats['coded_sps_mean']
        self.mcep_std_trg = self.trg_spk_stats['coded_sps_std']

        self.spk_idx_src, self.spk_idx_trg = spk2idx[src_spk], spk2idx[trg_spk]
        spk_cat_src = to_categorical([self.spk_idx_src], num_classes=len(speakers))
        spk_cat_trg = to_categorical([self.spk_idx_trg], num_classes=len(speakers))
        self.spk_c_org = spk_cat_src
        self.spk_c_trg = spk_cat_trg

    def get_batch_test_data(self, batch_size=4):
        batch_data = []
        i = 0

        while i != batch_size:
            wav_file = self.src_wav_files[i]
            filename = os.path.basename(wav_file)
            num = filename.split('.')[0].split('_')[1]

            for j in range(len(self.trg_wav_files)):
                trg_wav_file = self.trg_wav_files[j]
                trg_filename = os.path.basename(trg_wav_file)
                trg_num = trg_filename.split('.')[0].split('_')[1]

                if num == trg_num:
                    batch_data.append(wav_file)
                    break
                elif j == len(self.trg_wav_files) - 1:
                    batch_size += 1

            i += 1

        return batch_data

def convert(config):
    # Directories
    if config.where_exec == "slurm":
        config.gnrl_data_dir = '/workspace/NASFolder'
        output_directory = os.path.join(config.gnrl_data_dir, "output")
    elif config.where_exec == "local":
        config.gnrl_data_dir = "E:/TFM_EN_ESTE_DISCO_DURO/TFM_project/"

    train_data_dir = os.path.join(config.gnrl_data_dir, "data/mc/train")
    test_data_dir = os.path.join( config.gnrl_data_dir, "data/mc/test")
    wav_dir = os.path.join( config.gnrl_data_dir, "data/wav16")

    runs_models_dir = os.path.join(config.models_save_dir, config.experiment_name)
    runs_models = os.listdir(runs_models_dir)

    # Find runs from the experiment
    for run_model in runs_models:
        conversions_savedir_run = os.path.join(config.convert_dir, config.experiment_name, run_model)
        if not os.path.exists(conversions_savedir_run):
            # Find iteration models for the run
            iterations_models = os.listdir(os.path.join(runs_models_dir, run_model))
            # Sort the checkpoint filenames in natural numerical order
            iterations_models_s = natsorted(iterations_models)
            # Use a list comprehension to create pairs based on iteration numbers
            iterations_models_pairs = [(iterations_models_s[i], iterations_models_s[i + 1]) for i in
                                range(0, len(iterations_models_s), 2)]

            # Iterate over pairs of D and G models
            for (iter_D_model, iter_G_model) in tqdm(iterations_models_pairs):
                n_iter = iter_G_model.split("-")[0]

                # Create directories
                os.makedirs(os.path.join(conversions_savedir_run, n_iter), exist_ok=True)
                sampling_rate, num_mcep, frame_period = config.sampling_rate, 36, 5
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Restore model
                print(f'Loading the trained models from step {n_iter}...')
                if "v1" in run_model:
                    generator = Generator_v1(num_speakers=10).to(device)
                elif "v2" in run_model:
                    generator = Generator_v2(num_speakers=10).to(device)

                iter_G_model_path = os.path.join(runs_models_dir, run_model, iter_G_model) # find generator model path
                generator.load_state_dict(torch.load(iter_G_model_path, map_location=lambda storage, loc: storage)) # load model weights

                # for all possible speaker pairs in config.speakers
                for i in tqdm(range(0, len(config.speakers))):
                    if config.speakers[i] in ["p262","p272","p229","p232"]:
                        for j in range(0, len(config.speakers)):
                            if config.speakers[j] in ["p262", "p272", "p229", "p232"]:
                                if i != j:
                                    speakers_mappings = {"p262":"vf1",
                                                         "p272":"vm1",
                                                        "p229":"vf2",
                                                        "p232":"vm2"}

                                    conversion_name = speakers_mappings[config.speakers[i]] + "_" + speakers_mappings[config.speakers[j]] + "_" + config.speakers[i] + "_" + config.speakers[j]
                                    target_dir = os.path.join(conversions_savedir_run, n_iter, conversion_name)
                                    os.makedirs(target_dir, exist_ok=True)
                                    print("Converting for n iter " + str(n_iter) + " for " + str(conversion_name))

                                    # Load speakers
                                    data_loader = ConvertDataset(config, src_spk=config.speakers[i], trg_spk=config.speakers[j], train_data_dir=train_data_dir, wav_dir=wav_dir)
                                    print('---------------------------------------')
                                    print('Source: ', config.speakers[i], ' Target: ', config.speakers[j])
                                    print('---------------------------------------')

                                    # Read a batch of testdata
                                    src_test_wavfiles = data_loader.get_batch_test_data(batch_size=config.num_converted_wavs)
                                    src_test_wavs = [load_wav(wavfile, sampling_rate) for wavfile in src_test_wavfiles]

                                    with torch.no_grad():
                                        for idx, wav in enumerate(src_test_wavs):
                                            print(f'({idx}), file length: {len(wav)}')
                                            wav_name = os.path.basename(src_test_wavfiles[idx])

                                            # convert wav to mceps
                                            f0, _, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
                                            f0_converted = pitch_conversion(f0=f0,
                                                                            mean_log_src=data_loader.logf0s_mean_src,
                                                                            std_log_src=data_loader.logf0s_std_src,
                                                                            mean_log_target=data_loader.logf0s_mean_trg,
                                                                            std_log_target=data_loader.logf0s_std_trg)
                                            coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
                                            print("Before being fed into G: ", coded_sp.shape)
                                            coded_sp_norm = (coded_sp - data_loader.mcep_mean_src) / data_loader.mcep_std_src
                                            coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1).to(device)
                                            spk_conds = torch.FloatTensor(data_loader.spk_c_trg).to(device)

                                            # Include org_conds if using src and target domain codes.
                                            org_conds = torch.FloatTensor(data_loader.spk_c_org).to(device)

                                            # generate converted speech
                                            coded_sp_converted_norm = generator(coded_sp_norm_tensor, spk_conds).data.cpu().numpy()
                                            coded_sp_converted = np.squeeze(coded_sp_converted_norm).T * data_loader.mcep_std_trg + data_loader.mcep_mean_trg
                                            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                                            print("After being fed into G: ", coded_sp_converted.shape)

                                            # convert back to wav
                                            wav_transformed = world_speech_synthesis(f0=f0_converted,
                                                                                     coded_sp=coded_sp_converted,
                                                                                     ap=ap,
                                                                                     fs=sampling_rate,
                                                                                     frame_period=frame_period)
                                            wav_id = wav_name.split('.')[0]

                                            # SAVE TARGET SYNTHESIZED
                                            sf.write(os.path.join(target_dir, f'{wav_id}-vcto-{data_loader.trg_spk}.wav'), data=wav_transformed,
                                                     samplerate=sampling_rate, subtype='PCM_24')

                                            # librosa.output.write_wav(os.path.join(target_dir, f'{wav_id}-vcto-{data_loader.trg_spk}.wav'),
                                            #                          wav_transformed,
                                            #                          sampling_rate)

                                            # SAVE COPY OF TARGET REFERENCE
                                            wav_num = wav_name.split('.')[0].split('_')[1]
                                            copy(f'{wav_dir}/{config.speakers[j]}/{config.speakers[j]}_{wav_num}.wav', target_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Exec config.
    parser.add_argument('--where_exec', type=str, default='local', help="slurm or local")  # "slurm", "local"

    # Model configuration.
    parser.add_argument('--num_converted_wavs', type=int, default=8, help='Number of wavs to convert.')
    parser.add_argument('--resume_model', type=str, default=None, help='Model to resume for testing.')
    #parser.add_argument('--speakers', type=str, nargs='+', required=True, help='Speakers to be converted.')
    parser.add_argument('--speakers', type=str, nargs='+', required=False, help='Speaker dir names.',
                        default=['p262', 'p272', 'p229', 'p232', 'p292', 'p293', 'p360', 'p361', 'p248', 'p251'])
    # Directories.
    parser.add_argument('--models_save_dir', type=str, default='Z:/Shared_PFC-TFG-TFM/Claudio/TOPK_VC/output_final/models/', help='Path to model save directory.')
    parser.add_argument('--experiment_name', type=str, default='[13_10_2023]_TopKv1_FINALEXP', help='Experiment name.')
    parser.add_argument('--convert_dir', type=str, default='./converted', help='Path to converted wavs directory.')

    parser.add_argument('--sampling_rate', type=int, default=16000, help='Sampling rate for converted wavs.')

    config = parser.parse_args()

    # no. of spks
    config.num_speakers = len(config.speakers)

    print(config)

    # if config.resume_model is None:
    #     raise RuntimeError("Please specify the step number for resuming.")
    # if len(config.speakers) < 2:
    #     raise RuntimeError("Need at least 2 speakers to convert audio.")

    convert(config)