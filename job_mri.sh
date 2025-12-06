#!/bin/bash

export BART_PATH=bart
export OMP_NUM_THREADS=1

# 获取所有包含 T2 的文件路径
file_list=(/public_data/public_dataset/public_data/multicoil_val_80/*T2*)

# 获取总文件数
total_files=${#file_list[@]}
num_gpus=8

# 函数：每张卡处理一部分文件
run_on_gpu() {
    local gpu_id=$1
    local -n files=$2  # 使用 nameref 引用数组

    for ((i=gpu_id; i<total_files; i+=num_gpus)); do

        # 🛑 检查是否请求中断
        if [ -f "STOP" ]; then
            echo "GPU $gpu_id: User interruption detected. Stopping..."
            exit 1
        fi
    
        filename="${files[$i]}"
        if [ -f "$filename" ]; then
            filename_without_extension="${filename%.h5}"
            filename_without_extension=$(basename "$filename_without_extension")
            for idx in {0..11}; do
                echo "GPU $gpu_id | file: $filename_without_extension | slice: $idx"
                CUDA_VISIBLE_DEVICES=$gpu_id python mri.py \
                    --kspace "$filename" \
                    --slice_idx "$idx" \
                    --progressive_params=stages=1,factor=1 \
                    --learning_rate 0.001 \
                    --iter 50 \
                    --wavelet_l1_weight 0.0 \
                    --tv_weight 0.0001 \
                    --steps 40 \
                    --acc_factor 6 \
                    --logdir=results/"$filename_without_extension"/"$idx" \
                    --schedule_params ddpm:beta_start=0.0001,beta_end=0.01
            done
        fi
    done
}

# 启动每张卡一个进程
for ((gpu=0; gpu<num_gpus; gpu++)); do
    run_on_gpu "$gpu" file_list &
done

# 等待所有子进程完成
wait
