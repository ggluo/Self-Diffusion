export BART_PATH=bart
export OMP_NUM_THREADS=1

python mri.py --verbose --kspace misc/ksp --size 396 --progressive_params=stages=1,factor=1 --learning_rate 0.001 --iter 150 --wavelet_l1_weight 0.0 --tv_weight 0.0001 --steps 40 --acc_factor 6 --logdir=results/mri3 --schedule_params ddpm:beta_start=0.0001,beta_end=0.01

python mri.py --verbose --kspace misc/kspace --progressive_params=stages=3,factor=1:4:4 --learning_rate 0.0001 --iter 500 --wavelet_l1_weight 0.0 --tv_weight 0.0001 --steps 40 --acc_factor 4 --logdir=results/mri2 --schedule_params ddpm:beta_start=0.0001,beta_end=0.01