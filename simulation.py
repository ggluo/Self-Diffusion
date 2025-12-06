#!/usr/bin/env python3

import argparse
import torch
import torch.optim as optim
import os
import utils
from sdi import Sim_SDI

def main(args):
    torch.manual_seed(args.seed)
    device = "cpu"

    results_dir = args.logdir
    os.makedirs(results_dir, exist_ok=True)

    x_true, i = utils.generate_signal(args.freq_arr, args.signal_length)

    # Measurement matrix A and y = A x
    A = torch.randn(args.num_measurements, args.signal_length) * (1 / torch.sqrt(torch.tensor(args.num_measurements, dtype=torch.float32)))
    y = A @ x_true
    x_measured = (A.T @ y).cpu()

    if args.admm:
        print("Perform recovery using ADMM BP")
        x_recovered = utils.admm_bpdn(A, y, args.admm_params['lambda_reg'], args.admm_params['rho'], args.admm_params['admm_iters'], args.signal_length, args.verbose)
        utils.plot_spectrum(x_true, x_recovered, i, args.signal_length, "ADMM-BP", results_dir, step_idx=args.admm_params['admm_iters'], x_measured=x_measured)

    print("Perform recovery using HiReS")
    net, network_name = utils.create_network(args.network, 1, 1, args.filter_number)
    net.apply(utils.init_weights)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    x_hat = torch.randn(1, 1, args.signal_length).to(device)
    solver = Sim_SDI(args, device, net, optimizer, A)
    kwargs = {'x_true': x_true, 'x_measured': x_measured, 'i': i, 'results_dir': results_dir, 'network_name': network_name, 'signal_length': args.signal_length}

    x_hat = solver.train(x_hat, y, **kwargs if args.verbose else {})
    utils.plot_psnr_values(solver.tracking_data, os.path.join(results_dir, 'error_curve'), title='Error Over Iteration', ylabel='Error')
    utils.plot_spectrum(x_true, x_hat.view(-1).cpu(), i, args.signal_length, network_name, results_dir, x_measured=x_measured)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1D Signal Recovery with U-Net or SimpleNet")
    parser.add_argument("--signal_length", type=int, default=128, help="Length of the signal (default: 128)")
    parser.add_argument("--num_measurements", type=int, default=30, help="Number of measurements (default: 30)")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps (default: 50)")
    parser.add_argument("--iter", type=int, default=100, help="Number of epochs per step (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer (default: 1e-3)")
    parser.add_argument("--filter_number", type=int, default=16, help="Base filter number for network (default: 16)")
    parser.add_argument("--network", type=str, default="unet", choices=["unet", "simple"], help="Network type: 'unet' or 'simple' (default: unet)")
    parser.add_argument("--tv_weight", type=float, default=0.0, help="Total Variation regularization weight (default: 0.0)")
    parser.add_argument("--l1_weight", type=float, default=0.00, help="L1 regularization weight (default: 0.001)")
    parser.add_argument("--fourier_l1_weight", type=float, default=0.01, help="Fourier L1 regularization weight (default: 0.01)")
    parser.add_argument("--wavelet_l1_weight", type=float, default=0.0, help="Wavelet L1 regularization weight (default: 0.0)")
    parser.add_argument("--verbose", action="store_true", help="Print loss values at each step (default: False)")
    parser.add_argument("--seed", type=int, default=40, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--logdir", type=str, default="results", help="Directory to save results (default: results)")
    parser.add_argument("--admm", action="store_true", help="Use ADMM instead of U-Net (default: False)")
    parser.add_argument("--admm_params", type=utils.parse_admm_params, default="lambda_reg=0.1,admm_iters=1000,rho=1.0", help="ADMM parameters as 'lambda_reg=num_iters=rho'")
    parser.add_argument("--schedule_params", type=utils.parse_schedule, default="ddpm:beta_start=0.000001,beta_end=0.004", help="Noise schedule as 'type:key1=value1,key2=value2,...'")
    parser.add_argument("--progressive_params", type=utils.parse_progressive_params, default="stages=1,factor=1,min_steps=1", help="Progressive parameters as 'stages=factor'")
    parser.add_argument("--freq_arr", type=utils.parse_freq_arr, default="1.0:1,0.5:10,1.0:6,1.0:4,1.0:2,1.0:5,0.3:11",
                        help="Frequency array as 'mag1:freq1,mag2:freq2,...' (default: 1.0:1,0.5:10,1.0:6,1.0:4,1.0:2,1.0:5,0.3:11)")

    args = parser.parse_args()
    main(args)