export BART_PATH=bart
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

sdi()
{
python mri.py --verbose --kspace misc/ksp --size 396 --progressive_params=stages=1,factor=1 --learning_rate $1 --iter $2 --wavelet_l1_weight 0.0 --tv_weight 0.0001 --steps $3 --acc_factor 6 --logdir=results/hyper2_acc6/sdi_$4 --schedule_params ddpm:beta_start=0.0001,beta_end=0.01
}


sdi 0.001 25 10 25x10
sdi 0.001 25 20 25x20
sdi 0.001 25 40 25x40

sdi 0.001 50 10 50x10
sdi 0.001 50 40 50x40
sdi 0.001 50 20 50x20

sdi 0.001 75 10 75x10
sdi 0.001 75 20 75x20
sdi 0.001 75 40 75x40

sdi 0.001 100 10 100x10
sdi 0.001 100 20 100x20
sdi 0.001 100 40 100x40

sdi 0.001 150 10 150x40
sdi 0.001 150 20 150x20
sdi 0.001 150 40 150x40

sdi 0.001 200 10 200x10
sdi 0.001 200 20 200x20
sdi 0.001 200 40 200x40

sdi 0.001 300 10 300x10
sdi 0.001 300 20 300x20
sdi 0.001 300 40 300x40

sdi 0.001 400 10 400x10
sdi 0.001 400 20 400x20
sdi 0.001 400 40 400x40

sdi 0.001 500 10 500x10
sdi 0.001 500 20 500x20
sdi 0.001 500 40 500x40

sdi 0.001 600 10 600x10
sdi 0.001 600 20 600x20
sdi 0.001 600 40 600x40

dip()
{
    python mri.py --verbose --kspace misc/ksp --size 396 --progressive_params=stages=1,factor=1 --learning_rate $1 --iter $2 --wavelet_l1_weight 0.0 --tv_weight 0.0001 --steps 1 --acc_factor 6 --logdir=results/hyper2_acc6/dip_$2 --schedule_params ddpm:beta_start=0.0001,beta_end=0.01
}


dip 0.001 250
dip 0.001 500
dip 0.001 750
dip 0.001 1000 
dip 0.001 2000
dip 0.001 3000
dip 0.001 4000
dip 0.001 5000
dip 0.001 6000
dip 0.001 7000
dip 0.001 8000
dip 0.001 9000
dip 0.001 10000

dip_wo_tv()
{
    python mri.py --verbose --kspace misc/ksp --size 396 --progressive_params=stages=1,factor=1 --learning_rate $1 --iter $2 --wavelet_l1_weight 0.0 --tv_weight 0.00 --steps 1 --acc_factor 6 --logdir=results/hyper2_acc6/dip_wo_tv_$2 --schedule_params ddpm:beta_start=0.0001,beta_end=0.01
}


dip_wo_tv 0.001 250
dip_wo_tv 0.001 500
dip_wo_tv 0.001 750
dip_wo_tv 0.001 1000 
dip_wo_tv 0.001 2000
dip_wo_tv 0.001 3000
dip_wo_tv 0.001 4000
dip_wo_tv 0.001 5000
dip_wo_tv 0.001 6000
dip_wo_tv 0.001 7000
dip_wo_tv 0.001 8000
dip_wo_tv 0.001 9000
dip_wo_tv 0.001 10000

# mri reconstruction under noise
recon_mri_noise()
{
python mri.py --verbose --kspace misc/ksp --size 396 --noise $1 --progressive_params=stages=1,factor=1 --learning_rate 0.001 --iter 200 --wavelet_l1_weight 0.0 --tv_weight 0.0001 --steps 20 --acc_factor 6 --logdir=results/noise/$1 --schedule_params ddpm:beta_start=0.0001,beta_end=0.01
}

recon_mri_noise 0.00005
recon_mri_noise 0.00004
recon_mri_noise 0.00003
recon_mri_noise 0.00002
recon_mri_noise 0.00001


# large image inpainting
python general.py --task Inpainting --data misc/_DSC1039.jpg  --size_x 3000 --size_y 5000 --mask misc/_DSC1039_coordinates.txt --logdir=results/sagres_factor_5 --factor 5 --progressive_params=stages=1,factor=1 --learning_rate 0.001 --iter 300 --wavelet_l1_weight 0.0 --tv_weight 0.0 --steps 40 --schedule_params ddpm:beta_start=0.0001,beta_end=0.01 --verbose

python general.py --task Inpainting --data misc/_DSC1039.jpg  --size_x 3000 --size_y 5000 --mask misc/_DSC1039_coordinates.txt --logdir=results/sagres_factor_2 --factor 2 --progressive_params=stages=1,factor=1 --learning_rate 0.001 --iter 500 --wavelet_l1_weight 0.0 --tv_weight 0.0 --steps 40 --schedule_params ddpm:beta_start=0.0001,beta_end=0.01 --verbose

python general.py --task Inpainting --data misc/_DSC1039.jpg  --size_x 3000 --size_y 5000 --mask misc/_DSC1039_coordinates.txt --logdir=results/sagres_factor_1 --factor 1 --progressive_params=stages=1,factor=1 --learning_rate 0.001 --iter 500 --wavelet_l1_weight 0.0 --tv_weight 0.0 --steps 40 --schedule_params ddpm:beta_start=0.0001,beta_end=0.01 --verbose

# denoising

denoise()
{
    python denoise.py --sigma $1 --dataset /home/gluo/Documents/CBSD68-dataset/CBSD68 --img $6 --logdir=/home/gluo/Documents/CBSD68-dataset/CBSD68/denoised"$1" --progressive_params=stages=1,factor=1 --learning_rate $4 --iter $3 --tv_weight $5 --steps 30 --schedule_params ddpm:beta_start=$2,beta_end=0.01 --verbose
}
echo "Running denoising tasks"
for i in {0000..0067}
do
    echo "Denoising image $i.png"
    denoise 25 0.0008 100 0.002 0.00000015 $i.png 
done 

DIP()
{
    python denoise.py --sigma $1 --dataset /home/gluo/Documents/CBSD68-dataset/CBSD68 --img $6 --logdir=/home/gluo/Documents/CBSD68-dataset/CBSD68/denoised_dip"$1" --progressive_params=stages=1,factor=1 --learning_rate $4 --iter 3000 --tv_weight $5 --steps 1 --schedule_params ddpm:beta_start=$2,beta_end=0.01 --verbose
}
echo "Running denoising tasks"
for i in {0000..0067}
do
    echo "Denoising image $i.png"
    DIP 25 0.0008 100 0.002 0.0000005 $i.png 
done 
