#!/bin/bash
CUDA_VISIBLE_DEVICES=1,2,3 python tasks/run.py --config=egs/os_avatar/audio2motion_vae.yaml --exp_name=audio2motion_vae_new --hparams=syncnet_ckpt_dir=checkpoints/audio_lm3d_syncnet_new --reset > logs/vae.log 2>&1 &
