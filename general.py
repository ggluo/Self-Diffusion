import torch
import numpy as np
import os
import argparse
from PIL import Image
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr

import mriutils
import utils
from sdi import General_SDI

# torch.use_deterministic_algorithms(True)

def main(args):
    torch.manual_seed(args.seed)
    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")

    chns  = 3

    # defined A and Ap
    ###### SR ######
    if args.task == 'SR':
        A     = torch.nn.AdaptiveAvgPool2d((args.size//args.scale,args.size//args.scale))
        Ap    = lambda z: MeanUpsample(z,args.scale)

    ###### Inpainting ######
    if args.task == 'Inpainting':
        mask = torch.ones([1,3,256,256]).to(device)
        pd = 10
        mask[:,:,128-pd:128+pd,128-pd:128+pd] = 0
        mask[:,:,100-pd:100+pd,100-pd:100+pd] = 0
        mask[:,:,80-pd:80+pd,80-pd:80+pd] = 0
        mask[:,:,80-pd:80+pd,140-pd:140+pd] = 0
        A = lambda z: z*mask
        Ap = A
    
    # load data and compute ground truth
    x_true = np.array(Image.open(args.data).convert("RGB").resize([256,256])).transpose(2,0,1)
    x_true = x_true / 255.0
    x_true = torch.from_numpy(x_true.astype(np.float32))[None].to(device)
    y = A(x_true)

    utils.save_img(y[0].cpu().permute(1,2,0),      os.path.join(logdir, 'y'))
    utils.save_img(x_true[0].cpu().permute(1,2,0), os.path.join(logdir, 'x_true'))
    
    net, _ = utils.create_network(network_type=args.network, in_channels=chns, out_channels=chns)
    net = net.to(device)
    utils.init_weights_unet2d(net, init_type='normal',init_gain=0.02, seed=args.seed)

    x_hat = torch.randn([args.samples,chns,args.size,args.size]).to(device)
    optimizer = optim.Adam(net.parameters(), lr = args.learning_rate)
 
    print("Perform HSR reconstruction...")
    if args.wavelet_l1_weight > 0:
        print(f"Wavelet_l1_weight: {args.wavelet_l1_weight:.4f}")
    if args.tv_weight > 0:
        print(f"Use tv_weight: {args.tv_weight:.4f}")
    print(f"Use learning_rate: {args.learning_rate:.4f}")
    
    solver = General_SDI(args, device, net, optimizer)
    x_hat  = solver.train(x_hat, y, A=A, x_true=x_true)
    utils.plot_psnr_values(solver.tracking_data, os.path.join(logdir, "psnr_curve"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRI reconstruction using HSR")
    parser.add_argument("--size", type=int, default=256, help="image size (default: 320)")
    parser.add_argument("--data", type=str, default="misc/kspace",help="path to k-space data, shape (ncoils, size, size)")
    parser.add_argument("--task", type=str, default="sr",help="path to k-space data, shape (ncoils, size, size)")
    
    parser.add_argument("--scale", type=int, default=2, help="acceleration factor (default: 4)")

    
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