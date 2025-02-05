#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2,3 /opt/conda/envs/real3dportrait/bin/python data_gen/utils/process_video/extract_lm2d.py --process_id 0 > logs/2dprocess_0.log 2>&1 &