#!/bin/bash

# 创建日志目录
mkdir -p logs

# GPU列表
gpus=(1 2 3)

# 每个GPU运行的进程数
processes_per_gpu=3

# 启动32个并行进程
for ((i=0; i<=17; i++))
do
    # 计算当前进程使用的GPU
    gpu_index=$(( (i) / processes_per_gpu ))
    gpu=${gpus[$gpu_index]}
    
    # 每个进程输出到单独的日志文件
    CUDA_VISIBLE_DEVICES=$gpu /opt/conda/envs/real3dportrait/bin/python data_gen/utils/process_audio/extract_hubert.py --process_id $i > logs/cl_hubert_process_$i.log 2>&1 &
done

# 等待所有后台进程完成
wait

echo "所有32个进程已完成，分布在GPU 1,2,3上"