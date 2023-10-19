from model_v2 import Generator_v2, Discriminator_v2
from model_v1 import Generator_v1, Discriminator_v1
import torch
import torch.nn.functional as F
from os.path import join, basename
import time
import datetime
from data_loader import to_categorical
from utils import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import soundfile as sf
from evaluation_mcds import *
import optuna

class SolverCustom(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, train_loader, test_loader, config, trial):
        """Initialize configurations."""

        # Optuna trial
        self.trial = trial

        # Data loader.
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.sampling_rate = config.sampling_rate

        # Model configurations.
        self.stargan_version = config.stargan_version
        self.num_speakers = config.num_speakers
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_id = config.lambda_id
        self.lambda_cls = config.lambda_cls

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.logger = SummaryWriter(log_dir=config.log_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Training on: ", self.device)

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # TopK configurations.
        self.topk_training = config.topk_training
        self.topk_gamma = config.topk_gamma
        self.topk_v = config.topk_v
        self.topk_from_iter = config.topk_from_iter

        # Build the model and tensorboard.
        self.build_model(stargan_version=self.stargan_version)

    def build_model(self, stargan_version="v2"):
        """Create a generator and a discriminator."""

        if stargan_version == "v2":
            self.generator = Generator_v2(num_speakers=self.num_speakers)
            self.discriminator = Discriminator_v2(num_speakers=self.num_speakers)
        elif stargan_version == "v1":
            self.generator = Generator_v1(num_speakers=self.num_speakers)
            self.discriminator = Discriminator_v1(num_speakers=self.num_speakers)
        else:
            raise print("Specify the version of StarGAN-VC (v1 or v2).")

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), self.d_lr, [self.beta1, self.beta2])

        self.print_network(self.generator, 'Generator')
        self.print_network(self.discriminator, 'Discriminator')

        self.generator.to(self.device)
        self.discriminator.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def log_loss_tensorboard(self, loss_dict, step):
        for k, v in loss_dict.items():
            self.logger.add_scalar(k, v, step)

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        g_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        d_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))

        self.generator.load_state_dict(torch.load(g_path, map_location=lambda storage, loc: storage))
        self.discriminator.load_state_dict(torch.load(d_path, map_location=lambda storage, loc: storage))

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradientgradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def sample_spk_c(self, size):
        spk_c = np.random.randint(0, self.num_speakers, size=size)
        spk_c_cat = to_categorical(spk_c, self.num_speakers)
        return torch.LongTensor(spk_c), torch.FloatTensor(spk_c_cat)

    def classification_loss(self, logit, target):
        """Compute softmax cross entropy loss."""
        return F.cross_entropy(logit, target)

    def load_wav(self, wavfile, sr=16000):
        wav, _ = librosa.load(wavfile, sr=sr, mono=True)
        return wav_padding(wav, sr=16000, frame_period=5, multiple = 4)

    def topk_real_sigmoid(self, out_src, k):
        sigmoid_layer = torch.nn.Sigmoid()
        out_sigmoid = sigmoid_layer(out_src)

        try:
            kth_largest = torch.topk(torch.reshape(out_sigmoid, [-1]), k, sorted=True)[0][-1]

            topk_mask_out_src = torch.where(out_sigmoid < kth_largest, torch.zeros_like(out_sigmoid, dtype=torch.float32),
                                        torch.ones_like(out_sigmoid, dtype=torch.float32))
            topk_out_src = out_src * topk_mask_out_src  # vector solo con la probabilidad de los topk
        except RuntimeError:
            print("ERROR: FOUND A BATCH WITH ONLY 1 ELEMENT. ZEROING OUT THE GRADIENTS.")
            topk_out_src = out_src

        return topk_out_src


    def train(self):
        """Train StarGAN."""
        # Set data loader.
        train_loader = self.train_loader
        #data_iter = iter(train_loader)

        # Read a batch of testdata
        test_wavfiles = self.test_loader.get_batch_test_data(batch_size=4)
        test_wavs = [self.load_wav(wavfile) for wavfile in test_wavfiles]

        # Determine whether do copysynthesize when first do training-time conversion test.
        cpsyn_flag = [True, False][0]
        # f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters is not None:
            print("resuming step %d ..."% self.resume_iters)
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()

        #Initialize starting value of K and minimum value v
        k = self.batch_size  # k is initialized as the batch_size
        if self.topk_training == True:
            topk_v_value = int(self.topk_v * self.batch_size)

        for i in range(start_iters, self.num_iters):
            # =================================================================================== #
            #                             0. Computing the new value of K                                #
            # =================================================================================== #
            if self.topk_training and i >= self.topk_from_iter:
                # print("---epoch---", i)
              # nonsense to reduce k in the first iterations, we start
                tmp_k = self.topk_gamma * k  # reduce k by a decay factor gamma (here or inside the first if)
                # tmp_k = topk_gamma * k #reduce k by a decay factor gamma (here or inside the first if)
                k = tmp_k
                new_k = round(tmp_k)
                if new_k >= topk_v_value:
                    new_k = new_k  # make sure that k its equal or higher than half of the batch size
                else:
                    new_k = topk_v_value
            else:
                new_k = k

            #for batch_data, batch_spk2idxs, batch_spk_cat in train_loader:
            batch_data, batch_spk2idxs, batch_spk_cat = next(train_loader)
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            mc_real = torch.stack(batch_data, dim=0)
            spk_label_org = torch.stack(batch_spk2idxs, dim=0)
            spk_c_org = torch.stack(batch_spk_cat, dim=0)

            mc_real.unsqueeze_(1)  # (B, D, T) -> (B, 1, D, T) for conv2d

            # Generate target domain labels randomly.
            # spk_label_trg: int,   spk_c_trg:one-hot representation
            spk_label_trg, spk_c_trg = self.sample_spk_c(mc_real.size(0))

            # mc_real = mc_real.to(self.device)              # Input mc.
            # spk_label_org = spk_label_org.to(self.device)  # Original spk labels.
            # spk_c_org = spk_c_org.to(self.device)          # Original spk one-hot.
            spk_label_trg = spk_label_trg.to(self.device)  # Target spk labels.
            spk_c_trg = spk_c_trg.to(self.device)          # Target spk one-hot.

            # =================================================================================== #
            #                             2. Train the Discriminator                              #
            # =================================================================================== #

            # Compute loss with REAL mc feats.
            if self.stargan_version == "v2":
                d_out_src = self.discriminator(mc_real, spk_c_trg, spk_c_org)
                d_loss_real = torch.mean((1.0 - d_out_src) ** 2)

            elif self.stargan_version == "v1":
                d_out_src, d_out_cls_spks = self.discriminator(mc_real)
                d_loss_real = - torch.mean(d_out_src)
                d_loss_cls_spks = self.classification_loss(d_out_cls_spks, spk_label_org)

            # Compute loss with FAKE mc feats.
            mc_fake = self.generator(mc_real, spk_c_trg)

            # StarGAN-VCv2
            if self.stargan_version == "v2":
                #TODO: Check why detach mc_fake
                d_out_fake = self.discriminator(mc_fake.detach(), spk_c_org, spk_c_trg)
                d_loss_fake = torch.mean(d_out_fake ** 2)

                # Compute total D loss
                # TODO: Why comment that
                d_loss = d_loss_real + d_loss_fake  # + self.lambda_gp * d_loss_gp

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss'] = d_loss.item()

            # StarGAN-VCv1
            elif self.stargan_version == "v1":
                d_out_src, d_out_cls_spks = self.discriminator(mc_fake)
                d_loss_fake = torch.mean(d_out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(mc_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * mc_real.data + (1 - alpha) * mc_fake.data).requires_grad_(True)
                d_out_src, _ = self.discriminator(x_hat)
                d_loss_gp = self.gradient_penalty(d_out_src, x_hat)

                # Compute total D loss
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls_spks + self.lambda_gp * d_loss_gp

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls_spks'] = d_loss_cls_spks.item()
                loss['D/loss_gp'] = d_loss_gp.item()
                loss['D/loss'] = d_loss.item()


            # Backward and optimize.
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                mc_fake = self.generator(mc_real, spk_c_trg)

                # StarGAN-VCv2
                if self.stargan_version == "v2":
                    g_out_src = self.discriminator(mc_fake, spk_c_org, spk_c_trg)
                    g_loss_fake = torch.mean((1.0 - g_out_src) ** 2)

                    # Target-to-original domain. Cycle-consistent.
                    mc_reconst = self.generator(mc_fake, spk_c_org)
                    g_loss_rec = torch.mean(torch.abs(mc_real - mc_reconst))

                    # Original-to-original, Id mapping loss. Mapping
                    mc_fake_id = self.generator(mc_real, spk_c_org)
                    g_loss_id = torch.mean(torch.abs(mc_real - mc_fake_id))

                    # TopK Training of the Generator G
                    if self.topk_training and i >= self.topk_from_iter:  # TopK Real/Fake training
                        print("Starting TopK Real/Fake training with k = ", new_k)
                        topk_out_src = self.topk_real_sigmoid(g_out_src, new_k)
                        # g_loss_fake = - torch.mean(topk_out_src)
                        g_loss_fake = torch.mean((1.0 - topk_out_src) ** 2)

                        # Log value of K to MLFlow
                        mlflow.log_metric("K_value", new_k, step=i)

                    # Compute total G loss
                    g_loss = g_loss_fake \
                        + self.lambda_rec * g_loss_rec \
                        + self.lambda_id * g_loss_id

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_id'] = g_loss_id.item()
                    loss['G/loss'] = g_loss.item()

                elif self.stargan_version == "v1":
                    g_out_src, g_out_cls_spks = self.discriminator(mc_fake)
                    g_loss_fake = - torch.mean(g_out_src)
                    g_loss_cls_spks = self.classification_loss(g_out_cls_spks, spk_label_trg)

                    # Target-to-original domain.
                    mc_reconst = self.generator(mc_fake, spk_c_org)
                    g_loss_rec = torch.mean(torch.abs(mc_real - mc_reconst))

                    # TopK Training of the Generator G
                    if self.topk_training and i >= self.topk_from_iter:  # TopK Real/Fake training
                        print("Starting TopK Real/Fake training with k = ", new_k)
                        topk_out_src = self.topk_real_sigmoid(g_out_src, new_k)
                        # g_loss_fake = - torch.mean(topk_out_src)
                        g_loss_fake = torch.mean((1.0 - topk_out_src) ** 2)

                        # Log value of K to MLFlow
                        mlflow.log_metric("K_value", new_k, step=i)

                    # Compute total G loss
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls_spks

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_cls_spks'] = g_loss_cls_spks.item()
                    loss['G/loss'] = g_loss.item()

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

            for key, value in loss.items():
                mlflow.log_metric(key, value, step=i)  # Log results to MLFlow

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)
                self.log_loss_tensorboard(loss, i+1)

            if (i+1) % self.sample_step == 0:
                sampling_rate = 16000
                num_mcep = 36
                frame_period = 5
                with torch.no_grad():
                    metrics_i = []
                    for idx, wav in tqdm(enumerate(test_wavs)):
                        wav_name = basename(test_wavfiles[idx])
                        # print(wav_name)
                        f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
                        f0_converted = pitch_conversion(f0=f0,
                            mean_log_src=self.test_loader.logf0s_mean_src, std_log_src=self.test_loader.logf0s_std_src,
                            mean_log_target=self.test_loader.logf0s_mean_trg, std_log_target=self.test_loader.logf0s_std_trg)
                        coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)

                        coded_sp_norm = (coded_sp - self.test_loader.mcep_mean_src) / self.test_loader.mcep_std_src
                        coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1).to(self.device)
                        conds = torch.FloatTensor(self.test_loader.spk_c_trg).to(self.device)

                        # Include org_conds if using src and target domain codes.
                        # org_conds = torch.FloatTensor(self.test_loader.spk_c_org).to(self.device)

                        coded_sp_converted_norm = self.generator(coded_sp_norm_tensor, conds).data.cpu().numpy()
                        coded_sp_converted = np.squeeze(coded_sp_converted_norm).T * self.test_loader.mcep_std_trg + self.test_loader.mcep_mean_trg
                        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                        # decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
                        wav_transformed = world_speech_synthesis(f0=f0_converted, coded_sp=coded_sp_converted,
                                                                ap=ap, fs=sampling_rate, frame_period=frame_period)

                        # librosa.output.write_wav(
                        #     join(self.sample_dir, str(i+1)+'-'+wav_name.split('.')[0]+'-vcto-{}'.format(self.test_loader.trg_spk)+'.wav'), wav_transformed, sampling_rate)
                        sf.write(join(self.sample_dir, str(i+1)+'-'+wav_name.split('.')[0]+'-vcto-{}'.format(self.test_loader.trg_spk)+'.wav'), data=wav_transformed, samplerate=sampling_rate, subtype='PCM_24')

                        if cpsyn_flag:
                            wav_cpsyn = world_speech_synthesis(f0=f0, coded_sp=coded_sp,
                                                        ap=ap, fs=sampling_rate, frame_period=frame_period)
                            #librosa.output.write_wav(join(self.sample_dir, 'cpsyn-'+wav_name), wav_cpsyn, sampling_rate)
                            sf.write(join(self.sample_dir, 'cpsyn-'+wav_name), data=wav_cpsyn, samplerate=sampling_rate,
                                     subtype='PCM_24')
                        # TODO: Compute MCD and MOS from sample
                        # Compute MCD
                        mcd, penalty, frames = get_metrics_wavs(wav_file_1=test_wavfiles[idx],
                                                                wav_file_2=os.path.join(self.sample_dir, str(i+1)+'-'+wav_name.split('.')[0]+'-vcto-{}'.format(self.test_loader.trg_spk)+'.wav'),
                                                                n_mfcc=36)
                        # Append to compute mean MCD
                        metrics_i.append(mcd)

                    mean_iter_mcd = np.mean(metrics_i)
                    std_iter_mcd = np.std(metrics_i)

                    mlflow.log_metric("MCD_mean", mcd, step=i)

                    # Report the metric to Optuna for optimization
                    self.trial.report(mean_iter_mcd, i)

                    # Check if the trial should be pruned based on the metric
                    if self.trial.should_prune():
                        mlflow.end_run()
                        raise optuna.TrialPruned()

                    cpsyn_flag = False

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                g_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                d_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))

                torch.save(self.generator.state_dict(), g_path)
                torch.save(self.discriminator.state_dict(), d_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}'.format(g_lr, d_lr))

        mlflow.end_run()

        return mean_iter_mcd

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, train_loader, test_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.sampling_rate = config.sampling_rate

        # Model configurations.
        self.stargan_version = config.stargan_version
        self.num_speakers = config.num_speakers
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_id = config.lambda_id
        self.lambda_cls = config.lambda_cls


        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.logger = SummaryWriter(log_dir=config.log_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Training on: ", self.device)

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # TopK configurations.
        self.topk_training = config.topk_training
        self.topk_gamma = config.topk_gamma
        self.topk_v = config.topk_v
        self.topk_from_iter = config.topk_from_iter

        # Build the model and tensorboard.
        self.build_model(stargan_version=self.stargan_version)

        print("hola")

    def build_model(self,stargan_version="v2"):
        """Create a generator and a discriminator."""
        if stargan_version == "v2":
            self.generator = Generator_v2(num_speakers=self.num_speakers)
            self.discriminator = Discriminator_v2(num_speakers=self.num_speakers)
        elif stargan_version == "v1":
            self.generator = Generator_v1(num_speakers=self.num_speakers)
            self.discriminator = Discriminator_v1(num_speakers=self.num_speakers)
        else:
            raise print("Specify the version of StarGAN-VC (v1 or v2).")

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), self.d_lr, [self.beta1, self.beta2])

        self.print_network(self.generator, 'Generator')
        self.print_network(self.discriminator, 'Discriminator')

        self.generator.to(self.device)
        self.discriminator.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def log_loss_tensorboard(self, loss_dict, step):
        for k, v in loss_dict.items():
            self.logger.add_scalar(k, v, step)

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        g_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        d_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))

        self.generator.load_state_dict(torch.load(g_path, map_location=lambda storage, loc: storage))
        self.discriminator.load_state_dict(torch.load(d_path, map_location=lambda storage, loc: storage))

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradientgradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def sample_spk_c(self, size):
        spk_c = np.random.randint(0, self.num_speakers, size=size)
        spk_c_cat = to_categorical(spk_c, self.num_speakers)
        return torch.LongTensor(spk_c), torch.FloatTensor(spk_c_cat)

    def classification_loss(self, logit, target):
        """Compute softmax cross entropy loss."""
        return F.cross_entropy(logit, target)

    def load_wav(self, wavfile, sr=16000):
        wav, _ = librosa.load(wavfile, sr=sr, mono=True)
        return wav_padding(wav, sr=16000, frame_period=5, multiple = 4)

    def topk_real_sigmoid(self, out_src, k):
        sigmoid_layer = torch.nn.Sigmoid()
        out_sigmoid = sigmoid_layer(out_src)

        try:
            kth_largest = torch.topk(torch.reshape(out_sigmoid, [-1]), k, sorted=True)[0][-1]

            topk_mask_out_src = torch.where(out_sigmoid < kth_largest, torch.zeros_like(out_sigmoid, dtype=torch.float32),
                                        torch.ones_like(out_sigmoid, dtype=torch.float32))
            topk_out_src = out_src * topk_mask_out_src  # vector solo con la probabilidad de los topk
        except RuntimeError:
            print("ERROR: FOUND A BATCH WITH ONLY 1 ELEMENT. ZEROING OUT THE GRADIENTS.")
            topk_out_src = out_src

        return topk_out_src

    def train(self):
        """Train StarGAN."""
        # Set data loader.
        train_loader = self.train_loader
        data_iter = iter(train_loader)

        # Read a batch of testdata
        test_wavfiles = self.test_loader.get_batch_test_data(batch_size=4)
        test_wavs = [self.load_wav(wavfile) for wavfile in test_wavfiles]

        # Determine whether do copysynthesize when first do training-time conversion test.
        cpsyn_flag = [True, False][0]
        # f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sampling_rate, frame_period = frame_period)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters is not None:
            print("resuming step %d ..."% self.resume_iters)
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Initialize starting value of K and minimum value v
        k = self.batch_size  # k is initialized as the batch_size
        if self.topk_training == True:
            topk_v_value = int(self.topk_v * self.batch_size)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            # =================================================================================== #
            #                             0. Computing the new value of K                                #
            # =================================================================================== #
            if self.topk_training and i >= self.topk_from_iter:
                # print("---epoch---", i)
              # nonsense to reduce k in the first iterations, we start
                tmp_k = self.topk_gamma * k  # reduce k by a decay factor gamma (here or inside the first if)
                # tmp_k = topk_gamma * k #reduce k by a decay factor gamma (here or inside the first if)
                k = tmp_k
                new_k = round(tmp_k)
                if new_k >= topk_v_value:
                    new_k = new_k  # make sure that k its equal or higher than half of the batch size
                else:
                    new_k = topk_v_value
            else:
                new_k = k

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch labels.
            try:
                mc_real, spk_label_org, spk_c_org = next(data_iter)
            except:
                data_iter = iter(train_loader)
                mc_real, spk_label_org, spk_c_org = next(data_iter)

            mc_real.unsqueeze_(1)  # (B, D, T) -> (B, 1, D, T) for conv2d

            # Generate target domain labels randomly.
            # spk_label_trg: int,   spk_c_trg:one-hot representation
            spk_label_trg, spk_c_trg = self.sample_spk_c(mc_real.size(0))

            mc_real = mc_real.to(self.device)              # Input mc.
            spk_label_org = spk_label_org.to(self.device)  # Original spk labels.
            spk_c_org = spk_c_org.to(self.device)          # Original spk one-hot.
            spk_label_trg = spk_label_trg.to(self.device)  # Target spk labels.
            spk_c_trg = spk_c_trg.to(self.device)          # Target spk one-hot.

            # =================================================================================== #
            #                             2. Train the Discriminator                              #
            # =================================================================================== #

            # Compute loss with REAL mc feats.
            if self.stargan_version == "v2":
                d_out_src = self.discriminator(mc_real, spk_c_trg, spk_c_org)
                d_loss_real = torch.mean((1.0 - d_out_src) ** 2)

            elif self.stargan_version == "v1":
                d_out_src, d_out_cls_spks = self.discriminator(mc_real)
                d_loss_real = - torch.mean(d_out_src)
                d_loss_cls_spks = self.classification_loss(d_out_cls_spks, spk_label_org)

            # Compute loss with FAKE mc feats.
            mc_fake = self.generator(mc_real, spk_c_trg)

            # StarGAN-VCv2
            if self.stargan_version == "v2":
                #TODO: Check why detach mc_fake
                d_out_fake = self.discriminator(mc_fake.detach(), spk_c_org, spk_c_trg)
                d_loss_fake = torch.mean(d_out_fake ** 2)

                # Compute total D loss
                # TODO: Why comment that
                d_loss = d_loss_real + d_loss_fake  # + self.lambda_gp * d_loss_gp

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss'] = d_loss.item()

            # StarGAN-VCv1
            elif self.stargan_version == "v1":
                d_out_src, d_out_cls_spks = self.discriminator(mc_fake)
                d_loss_fake = torch.mean(d_out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(mc_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * mc_real.data + (1 - alpha) * mc_fake.data).requires_grad_(True)
                d_out_src, _ = self.discriminator(x_hat)
                d_loss_gp = self.gradient_penalty(d_out_src, x_hat)

                # Compute total D loss
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls_spks + self.lambda_gp * d_loss_gp

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls_spks'] = d_loss_cls_spks.item()
                loss['D/loss_gp'] = d_loss_gp.item()
                loss['D/loss'] = d_loss.item()


            # Backward and optimize.
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()


            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                mc_fake = self.generator(mc_real, spk_c_trg)

                # StarGAN-VCv2
                if self.stargan_version == "v2":
                    g_out_src = self.discriminator(mc_fake, spk_c_org, spk_c_trg)
                    g_loss_fake = torch.mean((1.0 - g_out_src) ** 2)

                    # Target-to-original domain. Cycle-consistent.
                    mc_reconst = self.generator(mc_fake, spk_c_org)
                    g_loss_rec = torch.mean(torch.abs(mc_real - mc_reconst))

                    # Original-to-original, Id mapping loss. Mapping
                    mc_fake_id = self.generator(mc_real, spk_c_org)
                    g_loss_id = torch.mean(torch.abs(mc_real - mc_fake_id))

                    # TopK Training of the Generator G
                    if self.topk_training and i >= self.topk_from_iter:  # TopK Real/Fake training
                        print("Starting TopK Real/Fake training with k = ", new_k)
                        topk_out_src = self.topk_real_sigmoid(g_out_src, new_k)
                        # g_loss_fake = - torch.mean(topk_out_src)
                        g_loss_fake = torch.mean((1.0 - topk_out_src) ** 2)

                        # Log value of K to MLFlow
                        mlflow.log_metric("K_value", new_k, step=i)

                    # Compute total G loss
                    g_loss = g_loss_fake \
                        + self.lambda_rec * g_loss_rec \
                        + self.lambda_id * g_loss_id

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_id'] = g_loss_id.item()
                    loss['G/loss'] = g_loss.item()

                elif self.stargan_version == "v1":
                    g_out_src, g_out_cls_spks = self.discriminator(mc_fake)
                    g_loss_fake = - torch.mean(g_out_src)
                    g_loss_cls_spks = self.classification_loss(g_out_cls_spks, spk_label_trg)

                    # Target-to-original domain.
                    mc_reconst = self.generator(mc_fake, spk_c_org)
                    g_loss_rec = torch.mean(torch.abs(mc_real - mc_reconst))

                    # TopK Training of the Generator G
                    if self.topk_training and i >= self.topk_from_iter:  # TopK Real/Fake training
                        print("Starting TopK Real/Fake training with k = ", new_k)
                        topk_out_src = self.topk_real_sigmoid(g_out_src, new_k)
                        g_loss_fake = - torch.mean(topk_out_src)
                        # g_loss_fake = torch.mean((1.0 - topk_out_src) ** 2)

                        # Log value of K to MLFlow
                        mlflow.log_metric("K_value", new_k, step=i)

                    # Compute total G loss
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls_spks

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_cls_spks'] = g_loss_cls_spks.item()
                    loss['G/loss'] = g_loss.item()

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

            for key, value in loss.items():
                mlflow.log_metric(key, value, step=i)  # Log results to MLFlow

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)
                self.log_loss_tensorboard(loss, i+1)

            if (i+1) % self.sample_step == 0:
                sampling_rate = 16000
                num_mcep = 36
                frame_period = 5
                with torch.no_grad():
                    for idx, wav in tqdm(enumerate(test_wavs)):
                        wav_name = basename(test_wavfiles[idx])
                        # print(wav_name)
                        f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
                        f0_converted = pitch_conversion(f0=f0,
                            mean_log_src=self.test_loader.logf0s_mean_src, std_log_src=self.test_loader.logf0s_std_src,
                            mean_log_target=self.test_loader.logf0s_mean_trg, std_log_target=self.test_loader.logf0s_std_trg)
                        coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)

                        coded_sp_norm = (coded_sp - self.test_loader.mcep_mean_src) / self.test_loader.mcep_std_src
                        coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1).to(self.device)
                        conds = torch.FloatTensor(self.test_loader.spk_c_trg).to(self.device)

                        # Include org_conds if using src and target domain codes.
                        # org_conds = torch.FloatTensor(self.test_loader.spk_c_org).to(self.device)

                        coded_sp_converted_norm = self.generator(coded_sp_norm_tensor, conds).data.cpu().numpy()
                        coded_sp_converted = np.squeeze(coded_sp_converted_norm).T * self.test_loader.mcep_std_trg + self.test_loader.mcep_mean_trg
                        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                        # decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sampling_rate)
                        wav_transformed = world_speech_synthesis(f0=f0_converted, coded_sp=coded_sp_converted,
                                                                ap=ap, fs=sampling_rate, frame_period=frame_period)

                        # librosa.output.write_wav(
                        #     join(self.sample_dir, str(i+1)+'-'+wav_name.split('.')[0]+'-vcto-{}'.format(self.test_loader.trg_spk)+'.wav'), wav_transformed, sampling_rate)
                        sf.write(join(self.sample_dir, str(i+1)+'-'+wav_name.split('.')[0]+'-vcto-{}'.format(self.test_loader.trg_spk)+'.wav'), data=wav_transformed, samplerate=sampling_rate, subtype='PCM_24')

                        if cpsyn_flag:
                            wav_cpsyn = world_speech_synthesis(f0=f0, coded_sp=coded_sp,
                                                        ap=ap, fs=sampling_rate, frame_period=frame_period)
                            #librosa.output.write_wav(join(self.sample_dir, 'cpsyn-'+wav_name), wav_cpsyn, sampling_rate)
                            sf.write(join(self.sample_dir, 'cpsyn-'+wav_name), data=wav_cpsyn, samplerate=sampling_rate,
                                     subtype='PCM_24')

                    cpsyn_flag = False

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                g_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                d_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))

                torch.save(self.generator.state_dict(), g_path)
                torch.save(self.discriminator.state_dict(), d_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}'.format(g_lr, d_lr))
