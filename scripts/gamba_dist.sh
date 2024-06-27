#! /bin/bash
wandb enabled && wandb online
# wandb disabled && wandb offline
export WANDB_API_KEY=your_wandb_api_key
echo 'MASTER_ADDR: '$MASTER_ADDR
echo 'MASTER_PORT: '$MASTER_PORT
echo 'RANK: '$RANK
echo 'LOCAL_RANK: '$LOCAL_RANK
echo 'WORLD_SIZE: '$WORLD_SIZE
accelerate launch --config_file acc_configs/gpu8x2.yaml \
    --machine_rank $RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    main.py gamba \
    --workspace /workspace_train \
