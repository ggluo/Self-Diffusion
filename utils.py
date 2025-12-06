import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch.nn as nn
import argparse
from net.simple import UNet1D, SimpleNet1D
from net.extra_deep_unet import ExtraDeepUNet
from torch.nn import init

def message(s, verbose):
    if verbose:
        print(s)

def plot_psnr_values(psnr_list, save_path="psnr.png", title="PSNR Values Over Iteration", 
                     xlabel="Iteration", ylabel="PSNR (dB)", figsize=(10, 6)):
    """
    Plot a list of PSNR values and save it as a figure.
    
    Args:
        psnr_list (list): List of PSNR values (e.g., numpy float64 values).
        save_path (str): Path to save the figure (e.g., 'psnr_plot.png', 'psnr_plot.pdf').
        title (str): Plot title.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        figsize (tuple): Figure size (width, height).
    """
    # Convert list to numpy array for easier handling
    psnr_array = np.array(psnr_list)
    
    # Create the plot
    plt.figure(figsize=figsize)
    plt.plot(range(len(psnr_array)), psnr_array, marker='o', linestyle='-', color='b', label='PSNR')
    
    # Add labels and title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend()
    
    # Save the plot to the specified file
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

def save_img(img, path, vmin=0., vmax=1., cmap='gray', interpolation=None, title=None):
    """
    print images to pdf and png without white margin

    Args:
    img: image arrays
    path: saving path
    """
    plt.imshow(img, cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches = 0)
    plt.savefig(path+'.pdf', bbox_inches='tight', pad_inches = 0)
    plt.close()

def plot_spectrum(x_true, x_recovered, i, signal_length, network_name, results_dir, step_idx=None, x_measured=None):
    """
    Plot time and frequency domain analysis of true vs recovered signals.
    
    Args:
        x_true: True signal tensor
        x_recovered: Recovered signal tensor
        signal_length: Length of the signal
        network_name: Name of the network/model
        results_dir: Directory to save the plot
        step_idx: Step index (None for final results)
        x_measured: Measured samples (A^T y) for final plot (optional)
    """
    
    # Frequency domain data
    fft_true = torch.fft.fft(x_true)
    fft_recovered = torch.fft.fft(x_recovered)
    freq = torch.fft.fftfreq(signal_length) * signal_length
    freq_np = freq[:signal_length//2].cpu().numpy()
    fft_magnitude_np = torch.abs(fft_true)[:signal_length//2].cpu().numpy()
    fft_recovered_np = torch.abs(fft_recovered)[:signal_length//2].cpu().numpy()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    error = torch.norm(x_recovered - x_true)

    # Time domain plot (top)
    ax1.plot(i, x_true, label="True Signal", color='blue')
    ax1.plot(i, x_recovered, label=f"Recovered Signal ({network_name})" if step_idx is None else "Recovered Signal", 
             color='red')
    
    # Add measured samples if provided (typically for final plot)
    if x_measured is not None:
        ax1.plot(i, x_measured, label="Measured Samples (A^T y)", linestyle='--', color='green')
    
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Amplitude")
    title_suffix = f"Final Time Domain with {network_name}, Recovery Error: {error.item():.4f}" if step_idx is None \
                  else f"Time Domain with {network_name}, Error: {error.item():.4f} (Step {step_idx})"
    ax1.set_title(title_suffix)
    ax1.legend()
    ax1.grid(True)

    # Frequency domain plot (bottom)
    ax2.stem(freq_np, fft_magnitude_np, label="True Signal FFT", markerfmt='bo', basefmt=" ", linefmt='b-')
    ax2.stem(freq_np, fft_recovered_np, label="Recovered Signal FFT", markerfmt='r^', basefmt=" ", linefmt='r-.')
    ax2.set_xlabel("Frequency (cycles per signal length)")
    ax2.set_ylabel("Magnitude")
    title_suffix = f"Frequency Spectrum with {network_name}" if step_idx is None \
                  else f"Frequency Spectrum with {network_name} (Step {step_idx})"
    ax2.set_title(title_suffix)
    ax2.grid(True)
    ax2.legend()

    # Adjust layout and save
    filename = f"{network_name}_final_results.png" if step_idx is None \
              else f"{network_name}_step_{step_idx:03d}_analysis.png"
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()

def generate_signal(freq_arr, signal_length):
    """
    Generate a signal from an array of frequency components
    
    Args:
        freq_arr: List of tuples [(magnitude1, frequency1), (magnitude2, frequency2), ...]
        signal_length: Integer length of the signal
    
    Returns:
        torch.Tensor: Generated signal
    """
    # Initialize the signal as zeros
    x_signal = torch.zeros(signal_length)
    
    # Time index
    i = torch.arange(signal_length, dtype=torch.float32)
    
    # Add each frequency component
    for magnitude, frequency in freq_arr:
        # Calculate the component and add it to the signal
        component = magnitude * torch.sin(2 * np.pi * frequency * i / signal_length)
        x_signal += component
    
    return x_signal, i

class SigmaSchedulerOld:
    """
    A class to compute sigma values based on a beta schedule
    
    Attributes:
        beta_start (float): Starting beta value
        beta_end (float): Ending beta value
        num_steps (int): Number of steps in the schedule
        device (torch.device): Device to perform computations on
        betas (torch.Tensor): Scheduled beta values
    """
    
    def __init__(self, beta_start, beta_end, num_steps, device='cpu'):
        """
        Initialize the SigmaScheduler
        
        Args:
            beta_start (float): Starting beta value
            beta_end (float): Ending beta value
            num_steps (int): Number of steps in the schedule
            device (str or torch.device): Device to perform computations on
        """
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_steps = num_steps
        self.device = device
        
        # Create beta schedule and convert to tensor
        betas = np.linspace(beta_start, beta_end, num_steps, dtype=np.float64)
        self.betas = torch.from_numpy(betas).float().to(device)
    
    def compute_sigma(self, t):
        """
        Compute sigma value at timestep t
        
        Args:
            t (int): Timestep (0 to num_steps-1)
            
        Returns:
            torch.Tensor: Sigma value at timestep t
        """
        if not 0 <= t < self.num_steps:
            raise ValueError(f"Timestep t must be between 0 and {self.num_steps-1}, got {t}")
            
        # Compute alpha (cumulative product of 1 - betas up to t)
        alpha = torch.cumprod(1 - self.betas[:t+1], dim=0)[-1]
        sigma_t = torch.sqrt(1 - alpha)
        return sigma_t

class SigmaScheduler:
    """
    A class to compute sigma values based on either DDPM or EDM noise schedule
    
    Attributes:
        schedule_type (str): Type of schedule ('ddpm' or 'edm')
        num_steps (int): Number of steps in the schedule
        device (torch.device): Device to perform computations on
    """
    
    def __init__(self, beta_start=0.0001, beta_end=0.02, num_steps=40,
                 sigma_min=0.002, sigma_max=0.5, rho=7.0, schedule_type='ddpm', device='cpu', factor=1):
        """
        Initialize the SigmaScheduler
        
        Args:
            schedule_type (str): 'ddpm' or 'edm' schedule type
            num_steps (int): Number of steps in the schedule
            device (str or torch.device): Device to perform computations on
            beta_start (float): Starting beta value (DDPM only)
            beta_end (float): Ending beta value (DDPM only)
            sigma_min (float): Minimum sigma value (EDM only)
            sigma_max (float): Maximum sigma value (EDM only)
            rho (float): Power parameter for schedule shaping (EDM only)
        """
        self.schedule_type = schedule_type.lower()
        self.num_steps = num_steps
        self.device = device
        
        if self.schedule_type not in ['ddpm', 'edm']:
            raise ValueError("schedule_type must be 'ddpm' or 'edm'")
            
        if self.schedule_type == 'ddpm':
            betas = np.linspace(beta_start/factor, beta_end/factor, num_steps, dtype=np.float64)
            self.betas = torch.from_numpy(betas).float().to(device)            
        elif self.schedule_type == 'edm':
            self.sigma_min = sigma_min / factor
            self.sigma_max = sigma_max / factor**2
            self.rho = rho
            # Create time points and pre-compute sigma schedule
            self.t_steps = torch.linspace(0, 1, num_steps, device=device)
            self.sigmas = self._compute_edm_schedule()
    
    def _compute_edm_schedule(self):
        """Compute EDM sigma schedule"""
        sigma_min_rho = self.sigma_min ** (1/self.rho)
        sigma_max_rho = self.sigma_max ** (1/self.rho)
        sigmas = (sigma_max_rho + self.t_steps * (sigma_min_rho - sigma_max_rho)) ** self.rho
        return sigmas
    
    def compute_sigma(self, t):
        """
        Compute sigma value at timestep t
        
        Args:
            t (int): Timestep (0 to num_steps-1)
            
        Returns:
            torch.Tensor: Sigma value at timestep t
        """
        if not 0 <= t < self.num_steps:
            raise ValueError(f"Timestep t must be between 0 and {self.num_steps-1}, got {t}")
            
        if self.schedule_type == 'ddpm':
            t = self.num_steps-1-t
            # Compute sigma from cumulative product of alphas
            alpha = torch.cumprod(1 - self.betas[:t+1], dim=0)[-1]
            sigma_t = torch.sqrt(1 - alpha)
            return sigma_t
        else:  # edm
            return self.sigmas[t]

def total_variation(x):
    diff = x[..., 1:] - x[..., :-1]
    return torch.sum(torch.abs(diff))

def total_variation_2d(x):
    diff_x = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]).sum()
    diff_y = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]).sum()
    return diff_x + diff_y

# L1 regularization on bottleneck weights (only for UNet1D) or final layer (SimpleNet1D)
def l1_regularization(model, l1_weight, network='unet'):
    l1_loss = 0
    for name, param in model.named_parameters():
        if network == 'unet':
            if 'bottleneck' in name and 'weight' in name:  # UNet1D-specific
                l1_loss += torch.sum(torch.abs(param))
        elif network == 'simple':
            if 'final' in name and 'weight' in name:  # SimpleNet1D-specific
                l1_loss += torch.sum(torch.abs(param))
        else:
            raise ValueError(f"Unknown network: {network}")
    return l1_weight * l1_loss

# L1 regularization in Fourier domain
def fourier_l1_loss(x, weight):
    x_flat = x.view(-1)
    fft_x = torch.fft.fft(x_flat)
    l1_norm = torch.sum(torch.sqrt(torch.abs(fft_x**2) + 1e-10))
    return weight * l1_norm

# L1 regularization in Wavelet domain
def wavelet_l1_loss(x, weight, wavelet='db1', level=3):
    x_flat = x.view(-1).detach().cpu().numpy()
    coeffs = pywt.wavedec(x_flat, wavelet=wavelet, level=level)
    coeffs_flat = np.concatenate([c.flatten() for c in coeffs])
    coeffs_tensor = torch.tensor(coeffs_flat, device=x.device, dtype=x.dtype)
    l1_norm = torch.sum(torch.abs(coeffs_tensor))
    return weight * l1_norm

def wavelet_l1_loss_2d(x, weight, wavelet='db1', level=3):
    """
    Compute the L1 norm of 2D wavelet coefficients for a complex-valued image tensor.
    
    Args:
        x (torch.Tensor): Input tensor of shape [batch, 2, height, width] (real-valued),
                          where channels=2 represents real and imag parts of 1 complex channel.
        weight (float): Scaling factor for the loss
        wavelet (str): Wavelet type (e.g., 'db1')
        level (int): Number of decomposition levels
    
    Returns:
        torch.Tensor: Weighted L1 norm of wavelet coefficients
    """
    # Convert to complex tensor
    x_ = torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous())  # [batch, height, width]

    # Initialize total L1 norm
    l1_norm_total = 0.0
    
    # Process each batch separately
    for b in range(x_.shape[0]):
        # Extract real and imaginary parts for the current batch
        x_real = x_[b, ...].real.detach().cpu().numpy()  # [height, width]
        x_imag = x_[b, ...].imag.detach().cpu().numpy()  # [height, width]

        # 2D wavelet decomposition for real and imaginary parts
        coeffs_real = pywt.wavedec2(x_real, wavelet=wavelet, level=level)
        coeffs_imag = pywt.wavedec2(x_imag, wavelet=wavelet, level=level)

        # Flatten coefficients, handling the nested tuple structure
        def flatten_coeffs(coeffs):
            flat = [coeffs[0].flatten()]  # Approximation coefficients
            for detail in coeffs[1:]:  # Detail coefficients (LH, HL, HH) tuples
                flat.extend([d.flatten() for d in detail])
            return np.concatenate(flat)

        coeffs_flat_real = flatten_coeffs(coeffs_real)
        coeffs_flat_imag = flatten_coeffs(coeffs_imag)

        # Convert to torch tensors
        coeffs_tensor_real = torch.tensor(coeffs_flat_real, device=x.device, dtype=x.dtype)
        coeffs_tensor_imag = torch.tensor(coeffs_flat_imag, device=x.device, dtype=x.dtype)

        # Compute L1 norm for this batch
        l1_norm = torch.sum(torch.abs(coeffs_tensor_real)) + torch.sum(torch.abs(coeffs_tensor_imag))
        l1_norm_total += l1_norm

    # Average over batch (since channels=1)
    l1_norm_total /= x_.shape[0]

    return weight * l1_norm_total

def wavelet_l1_loss_2d_abs(x, weight, wavelet='db1', level=3):
    """
    Compute the L1 norm of 2D wavelet coefficients for a complex-valued image tensor.
    
    Args:
        x (torch.Tensor): Input tensor of shape [batch, 2, height, width] (real-valued),
                          where channels=2 represents real and imag parts of 1 complex channel.
        weight (float): Scaling factor for the loss
        wavelet (str): Wavelet type (e.g., 'db1')
        level (int): Number of decomposition levels
    
    Returns:
        torch.Tensor: Weighted L1 norm of wavelet coefficients
    """
    # Convert to complex tensor
    x_ = torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous())  # [batch, height, width]

    # Initialize total L1 norm
    l1_norm_total = 0.0
    
    # Process each batch separately
    for b in range(x_.shape[0]):
        # Extract real and imaginary parts for the current batch
        x_abs = x_[b, ...].abs().detach().cpu().numpy()  # [height, width]

        # 2D wavelet decomposition for real and imaginary parts
        coeffs_abs = pywt.wavedec2(x_abs, wavelet=wavelet, level=level)

        # Flatten coefficients, handling the nested tuple structure
        def flatten_coeffs(coeffs):
            flat = [coeffs[0].flatten()]  # Approximation coefficients
            for detail in coeffs[1:]:  # Detail coefficients (LH, HL, HH) tuples
                flat.extend([d.flatten() for d in detail])
            return np.concatenate(flat)

        coeffs_flat_abs = flatten_coeffs(coeffs_abs)

        # Convert to torch tensors
        coeffs_tensor_abs = torch.tensor(coeffs_flat_abs, device=x.device, dtype=x.dtype)

        # Compute L1 norm for this batch
        l1_norm = torch.sum(torch.abs(coeffs_tensor_abs))
        l1_norm_total += l1_norm

    # Average over batch (since channels=1)
    l1_norm_total /= x_.shape[0]

    return weight * l1_norm_total

def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.normal_(m.weight, mean=0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def init_weights_unet2d(net, init_type='normal', init_gain=0.02, seed=40):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def parse_freq_arr(freq_str):
    """Convert a string like '1.0:1,0.5:10,1.0:6' into a list of (mag (float), freq (int)) tuples."""
    try:
        pairs = freq_str.split(',')
        freq_arr = []
        for pair in pairs:
            mag_str, freq_str = pair.split(':')
            mag = float(mag_str)  # Magnitude as float
            freq = int(freq_str)  # Frequency as int (convert to float first to handle "1.0" inputs)
            freq_arr.append((mag, freq))
        return freq_arr
    except (ValueError, IndexError) as e:
        raise argparse.ArgumentTypeError(f"Invalid frequency array format: {freq_str}. Use 'mag1:freq1,mag2:freq2,...'")

def create_network(network_type, in_channels, out_channels, filter_number=16):
    if network_type == 'unet':
        return UNet1D(in_channels, out_channels, filter_number), "U-Net"
    elif network_type == 'simple':
        return SimpleNet1D(in_channels, out_channels, filter_number), "SimpleNet"
    elif network_type == 'deepunet':
        return ExtraDeepUNet(in_channels, out_channels), "DeepUNet"
    else:
        raise ValueError(f"Unknown network type: {network_type}. Use 'unet' or 'simple'.")
    
def soft_threshold(x, threshold):
    """Soft-thresholding operator for L1 regularization."""
    return torch.sgn(x) * torch.maximum(torch.abs(x) - threshold, torch.tensor(0.0))

def admm_bpdn(A, y, lambda_reg, rho, num_iters, signal_length, verbose=False):
    # Initialize variables
    x = torch.zeros(signal_length)  # Signal estimate
    z = torch.zeros(signal_length, dtype=torch.complex64)  # Fourier coefficients
    u = torch.zeros(signal_length, dtype=torch.complex64)  # Dual variable

    # Precompute matrices for efficiency
    F = torch.fft.fft(torch.eye(signal_length)) / np.sqrt(signal_length)  # Fourier transform matrix
    F_inv = torch.fft.ifft(torch.eye(signal_length)) * np.sqrt(signal_length)  # Inverse Fourier transform matrix
    A_t_A = A.T @ A
    A_t_y = A.T @ y
    I = torch.eye(signal_length)
    # Solve (A^T A + rho I)^(-1) using Cholesky decomposition for efficiency
    L = torch.linalg.cholesky(A_t_A + rho * I)
    L_t = L.T

    for iter in range(num_iters):
        # x-update: min_x || A x - y ||_2^2 + (rho/2) || F x - z + u ||_2^2
        # This reduces to solving (A^T A + rho I) x = A^T y + rho F^H (z - u)
        rhs = A_t_y + rho * torch.fft.ifft(z - u).real * np.sqrt(signal_length)  # Shape: [64]
        rhs = rhs.view(-1, 1)  # Reshape to [64, 1] for solve_triangular
        x_temp = torch.linalg.solve_triangular(L_t, torch.linalg.solve_triangular(L, rhs, upper=False), upper=True)
        x = x_temp.view(-1)  # Reshape back to [64]

        # z-update: min_z lambda || z ||_1 + (rho/2) || F x - z + u ||_2^2
        # This reduces to soft-thresholding
        Fx = torch.fft.fft(x) / np.sqrt(signal_length)
        z = soft_threshold(Fx + u, lambda_reg / rho)

        # u-update: u = u + F x - z
        u = u + Fx - z

        # Optional: Print loss for monitoring
        if verbose and iter % 100 == 0:
            data_loss = torch.norm(A @ x - y) ** 2
            fourier_l1 = lambda_reg * torch.sum(torch.abs(z))
            print(f"Iter {iter}, Data Loss: {data_loss.item():.4f}, Fourier L1: {fourier_l1.item():.4f}")

    return x

def parse_admm_params(arg_string):
    """Parse ADMM parameters from a string like 'lambda_reg=0.1,num_iters=1000,rho=1.0'."""
    # Default values
    params = {
        "lambda_reg": 0.1,
        "admm_iters": 1000,
        "rho": 1.0
    }
    
    # Split the string by commas and then by equals signs
    if arg_string:
        for param in arg_string.split(","):
            key, value = param.split("=")
            if key == "lambda_reg":
                params["lambda_reg"] = float(value)
            elif key == "admm_iters":
                params["admm_iters"] = int(value)
            elif key == "rho":
                params["rho"] = float(value)
    return params

def parse_progressive_params(arg_string):
    params = {
        'stages': 1,
        'factor': [1],
        'min_steps': 1
    }

    if arg_string:
        for param in arg_string.split(','):
            key, value = param.split('=')
            if key == 'stages':
                params['stages'] = int(value)
            elif key == 'factor':
                factors = value.split(':')
                params['factor'] = [int(f) for f in factors]
                if len(params['factor']) != params['stages']:
                    raise ValueError(f"Expected {params['stages']} factors, got {len(params['factor'])}")
            elif key == 'min_steps':
                params['min_steps'] = int(value)
            else:
                raise ValueError(f"Unknown parameter: {key}")

    return params

def parse_schedule(schedule_str):
    """
    Parse a schedule string into parameters for SigmaScheduler
    
    Args:
        schedule_str (str): String in format 'type:key1=value1,key2=value2,...'
                          e.g., 'ddpm:beta_start=0.0001,beta_end=0.02'
                          or 'edm:sigma_min=0.002,sigma_max=80.0,rho=7.0'
    
    Returns:
        dict: Dictionary containing schedule_type and parameters
        
    Raises:
        ValueError: If the schedule string is malformed or has invalid parameters
    """
    try:
        # Split into type and parameters
        parts = schedule_str.split(':', 1)
        if len(parts) != 2:
            raise ValueError("Schedule string must contain type and parameters separated by ':'")
            
        schedule_type, params_str = parts
        schedule_type = schedule_type.lower()
        
        if schedule_type not in ['ddpm', 'edm']:
            raise ValueError("Schedule type must be 'ddpm' or 'edm'")
        
        # Default parameters
        params = {
            'schedule_type': schedule_type,
            'num_steps': 10, # not initialized through params_str
            'device': 'cpu',
            'factor': 1 # not initialized through params_str
        }
        
        # DDPM defaults
        if schedule_type == 'ddpm':
            params.update({
                'beta_start': 0.0001,
                'beta_end': 0.02
            })
        # EDM defaults
        else:  # edm
            params.update({
                'sigma_min': 0.002,
                'sigma_max': 5.0,
                'rho': 6.0
            })
        
        # Parse additional parameters if provided
        if params_str:
            param_pairs = params_str.split(',')
            for pair in param_pairs:
                key, value = pair.split('=')
                key = key.strip()
                
                # Convert value to appropriate type
                if key in ['beta_start', 'beta_end', 'sigma_min', 'sigma_max', 'rho']:
                    value = float(value)
                elif key == 'num_steps':
                    value = int(value)
                elif key == 'device':
                    value = value.strip()
                else:
                    raise ValueError(f"Unknown parameter: {key}")
                    
                params[key] = value
                
        # Validate parameters
        if schedule_type == 'ddpm':
            if params['beta_start'] >= params['beta_end']:
                raise ValueError("beta_start must be less than beta_end")
        else:  # edm
            if params['sigma_min'] >= params['sigma_max']:
                raise ValueError("sigma_min must be less than sigma_max")
            if params['rho'] <= 0:
                raise ValueError("rho must be positive")
                
        if params['num_steps'] <= 0:
            raise ValueError("num_steps must be positive")
            
        return params
        
    except Exception as e:
        raise ValueError(f"Invalid schedule string '{schedule_str}': {str(e)}")
    

if __name__ == "__main__":
    # Example usage
    schedule_str = "ddpm:beta_start=0.0001,beta_end=0.02"
    params = parse_schedule(schedule_str)
    scheduler = SigmaScheduler(**params)
    print(scheduler.betas, scheduler.compute_sigma(5))

    schedule2 = SigmaSchedulerOld(beta_start=0.0001, beta_end=0.02, num_steps=40)
    print(schedule2.betas, schedule2.compute_sigma(5))