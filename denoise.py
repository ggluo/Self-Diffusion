import torch
import numpy as np
import os
import argparse
from PIL import Image
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import utils
from sdi import General_SDI

torch.use_deterministic_algorithms(True)

def main(args):
    torch.manual_seed(args.seed)
    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")

    chns  = 3
    A = lambda z: z
    
    original_img = Image.open(os.path.join(args.dataset,"original_png", args.img))
    noisy_img = Image.open(os.path.join(args.dataset,f"noisy{args.sigma}", args.img))
    width, height = original_img.size

    new_width = width if width % 2 == 0 else width + 1
    new_height = height if height % 2 == 0 else height + 1

    if new_width != width or new_height != height:
        original_img = original_img.resize((new_width, new_height), Image.LANCZOS)
        noisy_img = noisy_img.resize((new_width, new_height), Image.LANCZOS)


    x_true = np.array(original_img.convert("RGB")).transpose(2,0,1)/255.0
    x_true = torch.from_numpy(x_true.astype(np.float32))[None].to(device)

    y = np.array(noisy_img.convert("RGB")).transpose(2,0,1)/255.0
    y = torch.from_numpy(y.astype(np.float32))[None].to(device)
    
    net, _ = utils.create_network(network_type=args.network, in_channels=chns, out_channels=chns)
    net = net.to(device)
    utils.init_weights_unet2d(net, init_type='normal',init_gain=0.02, seed=args.seed)

    x_hat = torch.randn([args.samples,chns,x_true.shape[2],x_true.shape[3]]).to(device)
    optimizer = optim.Adam(net.parameters(), lr = args.learning_rate)
 
    print("Perform SDI reconstruction...")
    if args.wavelet_l1_weight > 0:
        print(f"Wavelet_l1_weight: {args.wavelet_l1_weight:.6f}")
    if args.tv_weight > 0:
        print(f"TV_weight: {args.tv_weight:.8f}")
    print(f"Learning_rate: {args.learning_rate:.6f}")
    
    solver = General_SDI(args, device, net, optimizer)
    x_hat  = solver.train(x_hat, y, A=A, x_true=x_true).permute(0,2,3,1).detach().cpu().numpy().squeeze()
    np.save(os.path.join(logdir, args.img.split(".")[0]+"_denoised"), x_hat)
    utils.save_img(x_hat.clip(0,1), os.path.join(logdir, args.img))
    utils.plot_psnr_values(solver.tracking_data, os.path.join(logdir, "psnr_curve"))
    utils.write_floats(solver.tracking_data, os.path.join(logdir, args.img.split(".")[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoising benchmark on CBSD68") # https://github.com/clausmichele/CBSD68-dataset.git
    parser.add_argument("--logdir", type=str, default="results", help="Directory to save results (default: results)")
    parser.add_argument("--dataset", type=str, default="misc/CBSD68/",help="path to CBSD68 dataset")
    parser.add_argument("--img", type=str, default="0010.png",help="path to CBSD68 image")
    parser.add_argument("--sigma", type=int, default="25",help="noise level")
    parser.add_argument("--steps", type=int, default=40, help="Number of denoising steps (default: 40)")
    parser.add_argument("--iter", type=int, default=100, help="Number of iterations per step (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer (default: 1e-3)")
    parser.add_argument("--filter_number", type=int, default=16, help="Base filter number for network (default: 16)")
    parser.add_argument("--network", type=str, default="deepunet", choices=["unet", "simple", "deepunet"], help="Network type: 'unet' or 'simple' or 'deepunet' (default: deepunet)")
    parser.add_argument("--tv_weight", type=float, default=0.0, help="Total Variation regularization weight (default: 0.0)")
    parser.add_argument("--wavelet_l1_weight", type=float, default=0.0, help="Wavelet L1 regularization weight (default: 0.0)")
    parser.add_argument("--verbose", action="store_true", help="Print loss values at each step (default: False)")
    parser.add_argument("--seed", type=int, default=40, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--schedule_params", type=utils.parse_schedule, default="ddpm:beta_start=0.0001,beta_end=0.02", help="Noise schedule as 'type:key1=value1,key2=value2,...'")
    parser.add_argument("--progressive_params", type=utils.parse_progressive_params, default="stages=1,factor=1,min_steps=1", help="Progressive parameters as 'stages=factor'")
    parser.add_argument("--samples", type=int, default=1, help="number of samples") # not test at the moment

    args = parser.parse_args()
    main(args)