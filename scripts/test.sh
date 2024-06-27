#! /bin/bash
python gamba_infer.py --model-type gamba --resume ./checkpoint/gamba_ep399.pth \
    --workspace workspace_test \
    --test_path ./data_test
