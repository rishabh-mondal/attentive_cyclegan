import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def ddp_test(rank, world_size):
    """Initialize DDP and test GPU communication."""
    print(f"Rank {rank} starting DDP...")

    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:29500",
                            rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Create a tensor on each process
    tensor = torch.ones(1).to(f"cuda:{rank}") * rank
    print(f"Before: Rank {rank} has tensor {tensor.item()}")

    # Reduce sum across all GPUs
    dist.all_reduce(tensor)
    print(f"After: Rank {rank} has tensor {tensor.item()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(ddp_test, args=(world_size,), nprocs=world_size, join=True)
