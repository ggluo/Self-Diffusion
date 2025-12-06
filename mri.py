import torch
import numpy as np
import os
import argparse
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr

import mriutils
import utils
from sdi import MRI_SDI

torch.use_deterministic_algorithms(True)

def main(args):
    torch.manual_seed(args.seed)
    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")

    # load data and compute ground truth
    kspace = mriutils.readcfl(args.kspace)
    size    = args.size
    chns    = 2
    if args.random:
        mask = mriutils.random_mask(size, size, args.acc_factor)
    else:
        mask = mriutils.equal_mask(size, size, args.acc_factor)
    if args.acs != 0:
        mask[:,size//2-args.acs//2:size//2+args.acs//2] = 1
    mriutils.writecfl(os.path.join(logdir, 'mask'), mask)
    utils.save_img(mask, os.path.join(logdir, 'mask'))

    coilsen = mriutils.bart(1, 'ecalib  -m1 -W -c0', kspace.transpose((1, 2, 0))[None,...])
    l1_pics = mriutils.bart(1, 'pics -l1 -r 0.005', (mask[None]*kspace).transpose((1, 2, 0))[None], coilsen).squeeze()[::-1]

    # to torch tensor
    coilsen = coilsen.transpose((3, 1, 2, 0)).squeeze()
    kspace  = torch.from_numpy(kspace.astype(np.complex64)).to(device)
    mask    = torch.from_numpy(mask.astype(np.complex64)).to(device)
    coilsen = torch.from_numpy(coilsen.astype(np.complex64)).to(device)

    und_kspace = mask * kspace
    zero_filled = mriutils.A_adjoint(und_kspace, coilsen)
    zero_filled = zero_filled.abs().cpu().numpy()[::-1]
    zero_filled = zero_filled / np.max(zero_filled)
    utils.save_img(zero_filled, os.path.join(logdir, 'zero_filled'))

    x_true = mriutils.A_adjoint(kspace, coilsen)
    x_true = torch.abs(x_true)/torch.max(torch.abs(x_true))
    x_true = x_true.abs().cpu().numpy()[::-1]
    utils.save_img(x_true, os.path.join(logdir, 'x_true'))
    
    l1_pics = l1_pics / np.max(l1_pics)
    print(f"Bart l1 PSNR: {compute_psnr(x_true, abs(l1_pics)):.4f}")
    utils.save_img(abs(l1_pics), os.path.join(logdir, 'l1_pics'))

    net, _ = utils.create_network(network_type=args.network, in_channels=chns, out_channels=chns)
    net = net.to(device)
    utils.init_weights_unet2d(net, init_type='normal',init_gain=0.02, seed=args.seed)

    with torch.no_grad():
        scale_factor = torch.linalg.norm(net(torch.randn([args.samples,chns,size,size]).to(device)))/torch.linalg.norm(mriutils.A_adjoint(und_kspace, coilsen).to(device))
    y = scale_factor * und_kspace

    x_hat = torch.randn([args.samples,chns,size,size]).to(device)
    optimizer = optim.Adam(net.parameters(), lr = args.learning_rate)

    print("Perform HSR reconstruction...")
    if args.wavelet_l1_weight > 0:
        print(f"Wavelet_l1_weight: {args.wavelet_l1_weight:.4f}")
    if args.tv_weight > 0:
        print(f"Use tv_weight: {args.tv_weight:.4f}")
    print(f"Use learning_rate: {args.learning_rate:.4f}")
    
    solver = MRI_SDI(args, device, net, optimizer)
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