import os

import torch
import torch.distributed as dist


def init_distributed_mode():
    """
    Initializes the distributed training environment.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        gpu = rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        return

    torch.cuda.set_device(gpu)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    dist.barrier()


def is_main_process():
    """
    Checks if the current process is the main process (rank 0).
    """
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_rank():
    """
    Gets the rank of the current process.
    """
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    """
    Gets the total number of processes.
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return 1
