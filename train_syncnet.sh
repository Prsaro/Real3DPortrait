#!/bin/bash
CUDA_VISIBLE_DEVICES=1,2,3 python tasks/run.py --config=egs/os_avatar/audio_lm3d_syncnet.yaml --exp_name=audio_lm3d_syncnet_new --reset > logs/syncnet.log 2>&1 &