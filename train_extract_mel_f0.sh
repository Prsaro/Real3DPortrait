#!/bin/bash

# 创建日志目录
mkdir -p logs

# GPU列表
gpus=(1 2 3)

# 每个GPU运行的进程数
processes_per_gpu=8

# 启动12个并行进程
for ((i=0; i<=11; i++))
do
    # 计算当前进程使用的GPU
    gpu_index=$(( (i) / processes_per_gpu ))
    gpu=${gpus[$gpu_index]}
    
    # 每个进程输出到单独的日志文件
    CUDA_VISIBLE_DEVICES=$gpu /opt/conda/envs/real3dportrait/bin/python data_gen/utils/process_audio/extract_mel_f0.py --process_id $i > logs/cl_audio_process_$i.log 2>&1 &
done

# 等待所有后台进程完成
wait

echo "所有32个进程已完成，分布在GPU 1,2,3 上"
