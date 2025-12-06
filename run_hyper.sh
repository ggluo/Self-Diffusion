export BART_PATH=bart
export OMP_NUM_THREADS=1

dip()
{
    python mri.py --verbose --kspace misc/kspace --progressive_params=stages=1,factor=1 --learning_rate $1 --iter $2 --wavelet_l1_weight 0.0 --tv_weight 0.0001 --steps 1 --acc_factor 4 --logdir=results/hyper/dip_$3 --schedule_params ddpm:beta_start=0.0001,beta_end=0.01
}

dip 0.001 1000 1
dip 0.001 2000 2
dip 0.001 3000 3
dip 0.001 4000 4
dip 0.001 5000 5
dip 0.001 6000 6


sdi()
{
    python mri.py --verbose --kspace misc/kspace --progressive_params=stages=1,factor=1 --learning_rate $1 --iter $2 --wavelet_l1_weight 0.0 --tv_weight 0.0001 --steps $3 --acc_factor 4 --logdir=results/hyper/sdi_$4 --schedule_params ddpm:beta_start=0.0001,beta_end=0.01
}

# total iterations = 1000
sdi 0.001 100 10 100x10
sdi 0.001 50 20 50x20
sdi 0.001 25 40 25x40

# total iterations = 2000
sdi 0.001 200 10 200x10
sdi 0.001 100 20 100x20
sdi 0.001 50 40 50x40

# total iterations = 3000
sdi 0.001 300 10 300x10
sdi 0.001 150 20 150x20
sdi 0.001 75 40 75x40

# total iterations = 4000
sdi 0.001 400 10 400x10
sdi 0.001 200 20 200x20
sdi 0.001 100 40 100x40

# total iterations = 6000
sdi 0.001 600 10 600x10
sdi 0.001 300 20 300x20
sdi 0.001 150 40 150x40



#python mri.py --verbose --kspace misc/kspace --progressive_params=stages=3,factor=1:4:4 --learning_rate 0.001 --iter 150 --wavelet_l1_weight 0.0 --tv_weight 0.0001 --steps 40 --acc_factor 4 --logdir=results/mri1 --schedule_params ddpm:beta_start=0.0001,beta_end=0.01


#python mri.py --verbose --kspace misc/kspace --progressive_params=stages=3,factor=1:4:4 --learning_rate 0.0005 --iter 150 --wavelet_l1_weight 0.0 --tv_weight 0.0001 --steps 40 --acc_factor 4 --logdir=results/mri2 --schedule_params ddpm:beta_start=0.0001,beta_end=0.0