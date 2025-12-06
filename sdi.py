from abc import ABC, abstractmethod
import torch
from tqdm import tqdm
import utils
import mriutils
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import os

class SDI_Base(ABC):
    def __init__(self, args, device, net, optimizer):
        self.args = args
        self.device = device
        self.net = net
        self.optimizer = optimizer
        self.tracking_data = []

    def train(self, x_hat, y, **kwargs):
        base_idx = 0
        for stage in range(self.args.progressive_params['stages']):
            factor = self.args.progressive_params['factor'][stage]
            steps = max(self.args.steps // factor, self.args.progressive_params['min_steps'])
            
            schedule_params = self.args.schedule_params.copy()
            schedule_params['num_steps'] = steps
            schedule_params['factor'] = factor
            schedule_params['device'] = self.device
            sigmas = utils.SigmaScheduler(**schedule_params)

            total_iters = steps * self.args.iter
            pbar = tqdm(total=total_iters, desc="Training Progress")

            for idx in range(steps):
                noise = torch.randn_like(x_hat)
                sigma_t = sigmas.compute_sigma(idx)
                x_t = x_hat + noise * sigma_t
                
                for epoch in range(self.args.iter):
                    self.optimizer.zero_grad()
                    x_star = self.net(x_t)
                    loss = self.compute_loss(x_star, y, **kwargs)
                    loss.backward()
                    self.optimizer.step()

                    pbar.update(1)
                    pbar.set_description(f"Step {idx}, Epoch {epoch}, Sigma {sigma_t.item():.4f}")
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                
                x_hat = x_star.detach()

                if self.args.verbose:
                    self.plot(base_idx + idx, x_hat, **kwargs)
            base_idx += steps
        return x_hat

    @abstractmethod
    def plot(self, idx, x_star, **kwargs):
        pass

    @abstractmethod
    def compute_loss(self, x_star, y, **kwargs):
        """Compute the main loss term."""
        pass

class General_SDI(SDI_Base):
    def __init__(self, args, device, net, optimizer):
        super().__init__(args, device, net, optimizer)
        self.loss_fn = torch.nn.MSELoss()
    
    def compute_loss(self, x_star, y, **kwargs):
        loss = self.loss_fn(kwargs['A'](x_star), y)
        if self.args.wavelet_l1_weight > 0:
            loss = loss + utils.wavelet_l1_loss_2d_abs(x_star, self.args.wavelet_l1_weight)
        if self.args.tv_weight > 0:
            loss = loss + utils.total_variation_2d(x_star) * self.args.tv_weight
        return loss

    def plot(self, idx, x_star, **kwargs):
        x_hat_copy = x_star.permute(0,2,3,1).detach().cpu().numpy()

        x_hat_avg = np.mean(x_hat_copy, axis=0)
        x_hat_avg = x_hat_avg / np.max(x_hat_avg)
        psnr = compute_psnr(kwargs['x_true'].permute(0,2,3,1).detach().cpu().numpy()[0], x_hat_avg, data_range=1)
        self.tracking_data.append(psnr)

        utils.save_img(x_hat_avg, os.path.join(self.args.logdir, f'recon_{idx:03d}'), title=f"PSNR: {psnr:.4f}")

class Sim_SDI(SDI_Base):
    def __init__(self, args, device, net, optimizer, A):
        super().__init__(args, device, net, optimizer)
        self.A = A

    def compute_loss(self, x_star, y, **kwargs):
        x_star_flat = x_star.view(-1)
        pred_y = self.A @ x_star_flat
        data_loss = torch.linalg.norm(pred_y - y)
        fourier_loss = utils.fourier_l1_loss(x_star, self.args.fourier_l1_weight)
        loss = data_loss + fourier_loss
        if self.args.l1_weight > 0:
            loss = loss + utils.l1_regularization(self.net, self.args.l1_weight, self.args.network.lower())
        if self.args.tv_weight > 0:
            loss = loss + utils.total_variation(x_star) * self.args.tv_weight
        if self.args.wavelet_l1_weight > 0:
            loss = loss + utils.wavelet_l1_loss(x_star, self.args.wavelet_l1_weight)
        return loss

    def plot(self, idx, x_star, **kwargs):
        self.tracking_data.append(torch.norm(x_star.view(-1).cpu() - kwargs['x_true']).item())
        utils.plot_spectrum(step_idx=idx, x_recovered=x_star.view(-1).cpu(), **kwargs)

class MRI_SDI(SDI_Base):
    
    def __init__(self, args, device, net, optimizer):
        super().__init__(args, device, net, optimizer)

    def compute_loss(self, x_star, y, **kwargs):
        loss = mriutils.mri_l2_norm(x_star, y, kwargs['mask'], kwargs['coilsen'])
        if self.args.wavelet_l1_weight > 0:
            loss = loss + utils.wavelet_l1_loss_2d_abs(x_star, self.args.wavelet_l1_weight)
        if self.args.tv_weight > 0:
            loss = loss + utils.total_variation_2d(x_star) * self.args.tv_weight
        return loss

    def plot(self, idx, x_star, **kwargs):
        x_hat_copy = torch.view_as_complex(x_star.permute(0,2,3,1).contiguous().detach().cpu()).numpy()

        x_hat_avg = np.mean(x_hat_copy, axis=0)
        x_hat_avg = np.abs(x_hat_avg)
        x_hat_avg = x_hat_avg / np.max(x_hat_avg)
        x_hat_avg = x_hat_avg[::-1]
        psnr = compute_psnr(kwargs['x_true'], x_hat_avg)

        self.tracking_data.append(psnr)

        utils.save_img(x_hat_avg, os.path.join(self.args.logdir, f'recon_{idx:03d}')) #, title=f"PSNR: {psnr:.4f}")

class MRI3D_SDI(SDI_Base):
    
    def __init__(self, args, device, net, optimizer):
        super().__init__(args, device, net, optimizer)

    def compute_loss(self, x_star, y, **kwargs):
        loss = mriutils.mri_3d_l2_norm(x_star, y, kwargs['mask'], kwargs['coilsen'])
        if self.args.wavelet_l1_weight > 0:
            loss = loss + utils.wavelet_l1_loss_3d_abs(x_star, self.args.wavelet_l1_weight)
        # if self.args.tv_weight > 0:
            # loss = loss + utils.total_variation_3d(x_star) * self.args.tv_weight
        return loss

    def plot(self, idx, x_star, **kwargs):
        x_hat_copy = torch.view_as_complex(x_star.permute(0,2,3,4,1).contiguous().detach().cpu()).numpy()

        x_hat_avg = np.mean(x_hat_copy, axis=0)
        x_hat_avg = np.abs(x_hat_avg)
        x_hat_avg = x_hat_avg / np.max(x_hat_avg)
        # x_hat_avg = x_hat_avg[::-1]
        # psnr = compute_psnr(abs(kwargs['x_true'][125].cpu().numpy()), x_hat_avg[125],data_range=abs(kwargs['x_true'][125].cpu().numpy()).max())
        # self.tracking_data.append(psnr)

        utils.save_img(x_hat_avg[125], os.path.join(self.args.logdir, f'recon_{idx:03d}'), title=f"PSNR")
