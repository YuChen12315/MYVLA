"""Main script for training and testing."""

import pdb
import os
# 只在调试模式下启动 pdb
if os.environ.get('DEBUG', '0') == '1':
    pdb.set_trace()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse

from pathlib import Path
import sys

import torch

from datasets import fetch_dataset_class
from model.policy.factory import make_model, make_policy_config
from utils.common_utils import str2bool, str_none
from utils.trainers import fetch_train_tester


def parse_arguments():
    parser = argparse.ArgumentParser("Parse arguments for main.py")
    # Tuples: (name, type, default)
    main_dir="Peract2"

    DATA_PATH="/home/eai/3d_flowmatch_actor/datasets"  # change this to your dataset path

    train_data_dir=f"{DATA_PATH}/Peract2_zarr/train.zarr"
    eval_data_dir=f"{DATA_PATH}/Peract2_zarr/val.zarr"
    train_instructions="instructions/peract2/instructions.json"
    val_instructions="instructions/peract2/instructions.json"

    dataset="Peract2_3dfront_3dwrist"
    num_workers=1
    B=64  # we used 64 but you can use as low as 16 without much performance drop - it's much faster
    B_val=64
    chunk_size=1
    memory_limit=8  # this means 8GB CPU RAM per worker per GPU,
    # but it will never reach that, because these datasets are small
    # reduce this if you can't allocate more than 96GB of CPU memory

    # Training/testing arguments
    val_freq=4000
    eval_only=False
    lr=1e-4
    backbone_lr=1e-6  # doesn't matter when we don't finetune
    lr_scheduler="constant"
    wd=1e-10
    train_iters=300000
    use_compile=False  # much faster, but sometimes unstable
    use_ema=False
    lv2_batch_size=1  # you can increase this and divide B equally, speed/accuracy tradeoff

    # Model arguments, change (some of) these for new architectures
    model_type="VLTM"
    bimanual=True
    pre_tokenize=True
    workspace_normalizer_buffer=0.05   
    
    num_shared_attn_layers=4
    relative_action=False
    rotation_format="quat_xyzw"
    denoise_timesteps=5
    denoise_model="rectified_flow"
    num_history=5
    run_log_dir=f"{model_type}-{dataset}-B{B}-lr{lr}-{lr_scheduler}-H{num_history}-{denoise_model}"
    checkpoint=f"train_logs/{main_dir}/{run_log_dir}/last.pth"

    ngpus=1  # we used 4

    arguments = [
        # Dataset/loader arguments
        ('train_data_dir', Path, train_data_dir),
        ('eval_data_dir', Path, eval_data_dir),
        ('train_instructions', Path, train_instructions),
        ('val_instructions', Path, val_instructions),
        ('dataset', str, dataset),
        ('num_workers', int, num_workers),
        ('batch_size', int, B),
        ('batch_size_val', int, B_val),
        ('num_history', int, num_history),
        ('chunk_size', int, chunk_size),
        ('memory_limit', float, memory_limit),  # cache limit in GB
        # Logging arguments
        ('base_log_dir', Path, Path(__file__).parent / "train_logs"),
        ('exp_log_dir', Path, "exp"),
        ('run_log_dir', Path, run_log_dir),
        # Training and testing arguments
        ('checkpoint', str_none, checkpoint),
        ('val_freq', int, val_freq),
        ('interm_ckpt_freq', int, 1000000),
        ('eval_only', str2bool, False),
        ('lr', float, lr),
        ('backbone_lr', float, backbone_lr),
        ('lr_scheduler', str, lr_scheduler),
        ('wd', float, wd),
        ('train_iters', int, train_iters),
        ('use_compile', str2bool, use_compile),
        ('use_ema', str2bool, use_ema),
        ('relative_action', str2bool, relative_action),
        # Model arguments: general policy type
        ('model_type', str, model_type),
        ('bimanual', str2bool, bimanual),
        ('pre_tokenize', str2bool, pre_tokenize),
        ('custom_img_size', int, None),
        ('workspace_normalizer_buffer', float, workspace_normalizer_buffer),
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()


def suppress_output_on_non_main():
    if int(os.environ.get("RANK", 0)) != 0:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # Arguments
    args = parse_arguments()
    print("Arguments:")
    print(args)
    print("-" * 100)

    log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    print(
        "Available devices (CUDA_VISIBLE_DEVICES):",
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    print("Device count:", torch.cuda.device_count())
    import torch.distributed as dist
    # 设置环境变量（在调用初始化之前）
    os.environ['LOCAL_RANK'] = '0'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 然后初始化进程组
    dist.init_process_group(
        backend='gloo',
        init_method='env://',  # 使用环境变量中的设置
        world_size=int(os.environ['WORLD_SIZE']),
        rank=int(os.environ['RANK'])
    )
    args.local_rank = int(os.environ["LOCAL_RANK"])
    # suppress_output_on_non_main()

    # # DDP initialization
    # torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
   
    # Select dataset and model classes
    dataset_class = fetch_dataset_class(args.dataset)
    model_name = args.model_type
    config = make_policy_config(
        policy_type=model_name,
        args=args
    )

    # Run
    TrainTester = fetch_train_tester(args.dataset)
    train_tester = TrainTester(args, dataset_class, model_name, model_config=config)
    train_tester.main()

    # Safe program termination
    if torch.distributed.is_initialized():
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
