import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import socket
import os

def get_master_ip():
    """
    Automatically fetches the IP address of the master node.
    """
    return socket.gethostbyname(socket.gethostname())  # Gets local machine IP


import socket

def find_free_port():
    """
    Finds an available port on the system dynamically.
    Returns:
        int: Available port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to an available port
        return s.getsockname()[1]

# Example Usage
free_port = find_free_port()
print(f"Using free port: {free_port}")

def setup_ddp(rank, world_size):
    """
    Initializes Distributed Data Parallel (DDP) setup.

    Args:
        rank (int): Process ID.
        world_size (int): Total number of GPUs to use.
    """
    port = find_free_port()
    master_ip = get_master_ip()  # Automatically get master IP
    dist_url=f"tcp://{master_ip}:{port}"

    # dist_url = f"tcp://10:0:62:168:29500"  # Use chosen port

    print(f"Rank {rank}: Initializing DDP with init_method={dist_url}")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(world_size)))  # Limit visible GPUs
    dist.init_process_group(backend="nccl", init_method=dist_url, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def train_ddp(rank, world_size, train_fn, source_dir, target_dir, batch_size, num_epochs):
    """
    Trains the CycleGAN model using Distributed Data Parallel (DDP).
    
    Args:
        rank (int): The current GPU/process ID.
        world_size (int): Total number of GPUs to use.
        train_fn (function): Training function to execute.
        source_dir (str): Path to source images.
        target_dir (str): Path to target images.
        batch_size (int): Batch size.
        num_epochs (int): Total training epochs.
    """
    setup_ddp(rank, world_size)

    from utils.dataset import load_images
    from utils.train import train

    # Load datasets for each process
    source_dataloader = load_images(source_dir, batch_size)
    target_dataloader = load_images(target_dir, batch_size)

    # Run training on the assigned GPU
    train(
        source=source_dataloader,
        target=target_dataloader,
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=f"cuda:{rank}",
        rank=rank,
        world_size=world_size
    )

    dist.destroy_process_group()  # Clean up process group


def launch_ddp_training(train_fn, num_gpus, source_dir, target_dir, batch_size, num_epochs):
    """
    Spawns multiple processes for DDP training.

    Args:
        train_fn (function): Training function to execute.
        num_gpus (int): Number of GPUs to use.
        source_dir (str): Path to source images.
        target_dir (str): Path to target images.
        batch_size (int): Batch size.
        num_epochs (int): Total training epochs.
    """
    world_size = num_gpus  # Total GPUs to use
    mp.spawn(train_ddp, args=(world_size, train_fn, source_dir, target_dir, batch_size, num_epochs), nprocs=world_size, join=True)
