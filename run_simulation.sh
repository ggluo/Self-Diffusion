fourier_l1_weight=0.01
learning_rate=0.001

# experiment 1
python simulation.py --verbose --admm --admm_params rho=0.5 --network unet --filter_number 20 --steps 40 --learning_rate 0.00001 --fourier_l1_weight 0.01 --tv_weight 0.01 --logdir results/high_freq_prog --freq_arr "1.0:1,0.5:15,0.3:20,1.0:6,0.8:3,0.6:4,0.7:5" --progressive_params stages=4,factor=1:2:4:8,min_steps=10 --num_measurements 35 --iter 200

python simulation.py --verbose --admm --network simple --filter_number 300 --steps 50 \
--learning_rate $learning_rate --fourier_l1_weight $fourier_l1_weight --logdir results/low_freq_dominate
ffmpeg -framerate 5 -i results/low_freq_dominate/SimpleNet_step_%03d_analysis.png -c:v libx264 -crf 5 -pix_fmt yuv444p low_freq_dominate.mp4

: '
python simulation.py --verbose --network simple --filter_number 300 --steps 1 --num_epochs 1000 \
--learning_rate $learning_rate --fourier_l1_weight $fourier_l1_weight \
--beta_start 0.0 --beta_end 0.0 --logdir results/low_freq_dominate
'

# experiment 2
python simulation.py --verbose --admm --network simple --filter_number 300 --steps 40 \
--learning_rate $learning_rate --fourier_l1_weight $fourier_l1_weight \
--progressive_params stages=5,factor=1:2:4:8:10 --logdir results/low_freq_dominate_prog  
ffmpeg -framerate 5 -i results/low_freq_dominate_prog/SimpleNet_step_%03d_analysis.png -c:v libx264 -crf 5 -pix_fmt yuv444p low_freq_dominate_prog.mp4

# experiment 3
python simulation.py --verbose --admm --network simple --filter_number 300 --steps 40 \
--learning_rate $learning_rate --fourier_l1_weight $fourier_l1_weight \
--logdir results/high_freq_sparse --freq_arr "1.0:1,0.5:30,0.3:40,1.0:6"
ffmpeg -framerate 5 -i results/high_freq_sparse/SimpleNet_step_%03d_analysis.png -c:v libx264 -crf 5 -pix_fmt yuv444p high_freq_sparse.mp4

# experiment 4
python simulation.py --verbose --admm --network simple --filter_number 300 --steps 40 \
--learning_rate $learning_rate --fourier_l1_weight $fourier_l1_weight \
--logdir results/high_freq_sparse_prog --freq_arr "1.0:1,0.5:30,0.3:40,1.0:6" \
--progressive_params stages=7,factor=1:2:4:8:20:40:80,min_steps=5
ffmpeg -framerate 5 -i results/high_freq_sparse_prog/SimpleNet_step_%03d_analysis.png -c:v libx264 -crf 5 -pix_fmt yuv444p high_freq_sparse_prog.mp4

#--freq_arr "1.0:1,0.5:30,1.0:6" 