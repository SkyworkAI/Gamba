#! /bin/bash
accelerate launch --config_file acc_configs/gpu1.yaml main.py gamba --workspace /workspace_train --token_pnum 1
