import math
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from einops import reduce
from tqdm.auto import tqdm
from functools import partial
from collections import namedtuple
from Lossfunction import DiceLoss
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def identity(t, *args, **kwargs):
    return t


# Check if variable is not None
def exists(x):
    return x is not None


# Return value if exists, otherwise return default
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# Linear beta schedule for diffusion (used in original DDPM)
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


# Cosine beta schedule from the Improved DDPM paper
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# Extract specific index values and reshape to match input tensor shape
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# The core Gaussian Diffusion class
# Define the forward process and reverse process of the entire Diffusion model
class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            model,
            image_size,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            objective='pred_noise',
            beta_schedule='cosine',
            activate='none',
            p2_loss_weight_gamma=0.,
            p2_loss_weight_k=1,
            ddim_sampling_eta=1.,
            seg_weight=1,
    ):
        """
        In the following code,
            'p_' refers to the reverse process
            'q_' refers to the forward process.
        """
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.learned_sinusoidal_cond
        self.model = model
        self.multitask = 'seg_head_conv' in str(self.model.__dict__) or 'seg_model_conv' in str(self.model.__dict__)
        self.seg_weight = seg_weight
        self.channels = self.model.channels
        self.condition = self.model.condition
        self.activate = activate
        self.image_size = image_size
        self.objective = objective
        assert objective in {'pred_noise',
                             'pred_x0'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start)'
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps)  # default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # P2 reweighting loss
        register_buffer('p2_loss_weight',
                        (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_cond=None, seg_cond=None, label=None, clip_x_start=False):
        if not self.multitask:
            model_output = self.model(x, t, x_self_cond=x_cond, seg_cond=seg_cond, label=label)
        else:
            (model_output, seg_output) = self.model(x, t, x_self_cond=x_cond, seg_cond=seg_cond, label=label)
        if self.activate == 'tanh':
            model_output = torch.tanh(model_output)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        # return ModelPrediction(pred_noise, x_start), feature
        if not self.multitask:
            return ModelPrediction(pred_noise, x_start)
        else:
            return ModelPrediction(pred_noise, x_start), seg_output

    def feature(self, x, t, x_cond=None, label=None, clip_x_start=False):

        model_output = self.model(x, t, x_cond, label)
        # feature = self.model.feature.detach()

        if self.activate == 'tanh':
            model_output = torch.tanh(model_output)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        # return ModelPrediction(pred_noise, x_start), feature
        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_cond=None, label=None, clip_denoised=True):
        preds = self.model_predictions(x, t, x_cond, label)
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_cond=None, label=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, x_cond=x_cond, label=label,
                                                                          clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        x_start = None
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.condition else None
            img, x_start = self.p_sample(img, t, self_cond)
        return img

    def pred_cond(self, cond, seg_cond=None, label=None):
        """
        Define reverse process.
        """
        device = self.betas.device
        shape = (cond.size(0), self.model.channels, self.image_size, self.image_size)
        img = torch.randn(shape, device=device)
        for t in reversed(range(0, self.num_timesteps)):
            x_cond = cond if self.condition else None
            img, x_start = self.p_sample(img, t, x_cond, label=label)
        return img

    def pred_cond_fast(self, cond, seg_cond=None, label=None, sampling_timesteps=250, clip_denoised=True):
        """
        Define reverse process. Use DDIM's sampling method to reduce the number of inference time steps (1000 to 100)
        """
        device = self.betas.device
        batch = cond.size(0)
        shape = (cond.size(0), self.model.channels, self.image_size, self.image_size)
        times = torch.linspace(-1, self.num_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        img = torch.randn(shape, device=device)

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            if not self.multitask:
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond, cond, seg_cond=seg_cond, label=label, clip_x_start=clip_denoised)
            else:
                if time < len(self.alphas_cumprod)*0.2 and seg_cond is not None:
                    seg_cond = seg_output
                (pred_noise, x_start, *_), seg_output = self.model_predictions(img, time_cond, cond, seg_cond=seg_cond, label=label, clip_x_start=clip_denoised)
            if time_next < 0:
                img = x_start
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
        if not self.multitask:
            return img
        else:
            return img, seg_output


    @torch.no_grad()
    def sample(self, batch_size=16):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size))

    def q_sample(self, x_start, t, noise=None):
        """
        Define forward process
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')


    def p_losses(self, x_start, t, cond=None, seg=None, seg_cond=None, label=None, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_cond = cond
        if seg is None:
            model_out = self.model(x, t, x_self_cond=x_cond, seg_cond=seg_cond, label=label)
        else:
            (model_out, seg_out) = self.model(x, t, x_self_cond=x_cond, seg_cond=seg_cond, label=label)
        if self.activate == 'tanh':
            model_out = torch.tanh(model_out)
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':  # In CBSI, we chose to predict x0, which can converge faster
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')
        # loss = self.loss_fn(model_out, target, reduction='none')
        if self.loss_type == 'vgg':
            loss = self.loss_fn(model_out, target)
        elif self.loss_type == 'mix':
            loss_l1 = self.loss_fn[0](model_out, target, reduction='none')
            loss_l1 = reduce(loss_l1, 'b ... -> b (...)', 'mean')
            loss_l1 = loss_l1 * extract(self.p2_loss_weight, t, loss_l1.shape)
            loss_vgg = self.loss_fn[1](model_out, target) # for pred_noise, Convert to x0 and then compute?
            loss = loss_l1.mean() + loss_vgg * self.vgg_weight
        else:
            loss = self.loss_fn(model_out, target, reduction='none')
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            loss = loss * extract(self.p2_loss_weight, t, loss.shape)
            loss = loss.mean()

        if seg is not None:
            lossfunction_seg = DiceLoss()
            loss_seg = lossfunction_seg(seg_out, seg)
            loss = loss + loss_seg * self.seg_weight
        return loss


    def forward(self, img, cond, seg=None, seg_cond=None, label=None, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size} not {h}&{w}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(img, t, cond=cond, seg=seg, seg_cond=seg_cond, label=label, *args, **kwargs)



class DDPM_Trainer(object):
    def __init__(
            self,
            diffusion_model,
            train_batch_size=16,
            train_lr=1e-4,
            adam_betas=(0.9, 0.99),
    ):
        super().__init__()
        self.model = diffusion_model

        # Check if the model supports multitask outputs (e.g., image + segmentation)
        self.multitask = self.model.multitask

        # Training hyperparameters
        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size

        # Optimizer initialization using Adam
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

    def calculate_loss(self, x, x_cond, label=None):
        """
        Compute the training loss for standard (single-task) conditional diffusion.
        Args:
            x: Ground truth images.
            x_cond: Conditional input (e.g., NCCT).
            label: Optional class label for conditional generation.
        Returns:
            Scalar loss tensor.
        """
        loss = self.model(x, x_cond, label=label)
        return loss

    def calculate_loss_multitask(self, x, x_cond, seg, seg_cond, label=None):
        """
        Compute the training loss for multitask diffusion (e.g., image + segmentation).
        Args:
            x: Ground truth image.
            x_cond: Conditional input image.
            seg: Ground truth segmentation.
            seg_cond: Conditional segmentation input (optional).
            label: Optional class label for conditional generation.
        Returns:
            Scalar multitask loss tensor.
        """
        loss = self.model(x, x_cond, seg, seg_cond, label=label)
        return loss


    def pred(self, x_cond, T, seg_cond=None, label=None, sample_type='ddim'):
        """
        Generate samples from the trained model using either DDPM or DDIM.
        Args:
            x_cond: Conditional input image.
            T: Number of diffusion steps (T < total steps for DDIM, T == total for DDPM).
            seg_cond: Optional conditional segmentation input.
            label: Optional class label.
            sample_type: 'ddpm' or 'ddim' to choose sampling strategy.
        Returns:
            Predicted image (and segmentation if multitask).
        """
        if not self.multitask:
            # Single-task generation
            if T == self.model.num_timesteps and sample_type.lower() == 'ddpm':
                pred_img = self.model.pred_cond(x_cond, seg_cond=seg_cond, label=label)
            elif T < self.model.num_timesteps or sample_type.lower() == 'ddim':
                pred_img = self.model.pred_cond_fast(x_cond, seg_cond=seg_cond, label=label, sampling_timesteps=T)
            else:
                raise ValueError(f'invalid sample type {sample_type}')
            return pred_img
        else:
            # Multitask generation
            if T == self.model.num_timesteps and sample_type.lower() == 'ddpm':
                pred_img, pred_seg = self.model.pred_cond(x_cond, seg_cond=seg_cond, label=label)
            elif T < self.model.num_timesteps or sample_type.lower() == 'ddim':
                pred_img, pred_seg = self.model.pred_cond_fast(x_cond, seg_cond=seg_cond, label=label, sampling_timesteps=T)
            return pred_img, pred_seg

    def save(self, save_path):
        """
        Save model and optimizer state to a file.
        Args:
            save_path: Path to save the checkpoint file.
        """
        data = {
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, save_path)

    def load(self, model_path):
        """
        Load model and optimizer state from a checkpoint file.
        Automatically handles missing segmentation modules for backward compatibility.
        Args:
            model_path: Path to the checkpoint file.
        """
        data = torch.load(model_path)
        try:
            self.model.load_state_dict(data['model'])
            self.opt.load_state_dict(data['opt'])
            print('Load successfully. Use strict load method!')
        except:
            try:
                # Attempt to load model excluding segmentation head (if incompatible)
                current_state_dict = self.model.state_dict()
                filtered_state_dict = {k: v for k, v in data['model'].items() if 'seg_head_conv' not in k}
                current_state_dict.update(filtered_state_dict)
                self.model.load_state_dict(current_state_dict)
                print('Load successfully except seg module!!!')
            except:
                # Fall back to non-strict load
                print('Can not load successfully. Use non-strict load method!')
                self.model.load_state_dict(data['model'], strict=False)
