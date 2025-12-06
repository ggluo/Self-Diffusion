import torch
import numpy as np
import os
import argparse
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr

import mriutils
import utils
from sdi import MRI3D_SDI
from net import ExtraDeep3DUNet
# torch.use_deterministic_algorithms(True)

def main(args):
    torch.manual_seed(args.seed)
    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")


    data = np.load('e13991s3_P01536.npz',allow_pickle=True)['data'].item()
    kspace  = data['kspace']
    coilsen = data['coilsen']
    mask    = abs(kspace) > 0.001

    coilsen = coilsen.transpose((3, 0, 1, 2)).squeeze()
    kspace  = kspace.transpose((3, 0, 1, 2)).squeeze()
    mask    = mask.transpose((3, 0, 1, 2)).squeeze()
    
    und_kspace  = torch.from_numpy(kspace.astype(np.complex64)).to(device)
    mask        = torch.from_numpy(mask.astype(np.complex64)).to(device)
    coilsen     = torch.from_numpy(coilsen.astype(np.complex64)).to(device)

    x_true = mriutils.A_adjoint_3d(und_kspace,coilsen)
    # x_true = x_true / x_true.max()

    net = ExtraDeep3DUNet.ExtraDeep3DUNet(2,2).to(device)


    with torch.no_grad():
        scale_factor = torch.linalg.norm(net(torch.randn([1, 2, 256, 224, 192]).to(device)))/torch.linalg.norm(mriutils.A_adjoint_3d(und_kspace, coilsen).to(device))
        y = scale_factor * und_kspace
    
    x_hat = torch.randn([1, 2, 256, 224, 192]).to(device)
    optimizer = optim.Adam(net.parameters(), lr = args.learning_rate)

    print("Perform HSR reconstruction...")
    if args.wavelet_l1_weight > 0:
        print(f"Wavelet_l1_weight: {args.wavelet_l1_weight:.4f}")
    if args.tv_weight > 0:
        print(f"Use tv_weight: {args.tv_weight:.4f}")
    print(f"Use learning_rate: {args.learning_rate:.4f}")
    
    solver = MRI3D_SDI(args, device, net, optimizer)
    x_hat  = solver.train(x_hat, y, x_true=x_true, mask=mask, coilsen=coilsen)
    utils.plot_psnr_values(solver.tracking_data, os.path.join(logdir, "psnr_curve"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRI reconstruction using HSR")
    parser.add_argument("--size", type=int, default=320, help="image size (default: 320)")
    parser.add_argument("--kspace", type=str, default="misc/kspace",help="path to k-space data, shape (ncoils, size, size)")
    parser.add_argument("--acc_factor", type=int, default=4, help="acceleration factor (default: 4)")
    parser.add_argument("--random", action="store_true", help="Use random mask (default: False)")
    parser.add_argument("--acs", type=int, default=20, help="center acs lines (default: 20)")
    parser.add_argument("--steps", type=int, default=40, help="Number of denoising steps (default: 40)")
    parser.add_argument("--iter", type=int, default=100, help="Number of iterations per step (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer (default: 1e-3)")
    parser.add_argument("--filter_number", type=int, default=16, help="Base filter number for network (default: 16)")
    parser.add_argument("--network", type=str, default="deepunet", choices=["unet", "simple", "deepunet"], help="Network type: 'unet' or 'simple' or 'deepunet' (default: deepunet)")
    parser.add_argument("--tv_weight", type=float, default=0.0, help="Total Variation regularization weight (default: 0.0)")
    parser.add_argument("--wavelet_l1_weight", type=float, default=0.0, help="Wavelet L1 regularization weight (default: 0.0)")
    parser.add_argument("--verbose", action="store_true", help="Print loss values at each step (default: False)")
    parser.add_argument("--seed", type=int, default=40, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--logdir", type=str, default="results", help="Directory to save results (default: results)")
    parser.add_argument("--schedule_params", type=utils.parse_schedule, default="ddpm:beta_start=0.0001,beta_end=0.02", help="Noise schedule as 'type:key1=value1,key2=value2,...'")
    parser.add_argument("--progressive_params", type=utils.parse_progressive_params, default="stages=1,factor=1,min_steps=1", help="Progressive parameters as 'stages=factor'")
    parser.add_argument("--samples", type=int, default=1, help="number of samples")

    args = parser.parse_args()
    main(args)
