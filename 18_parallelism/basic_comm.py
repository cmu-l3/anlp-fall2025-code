# http://docs.pytorch.org/tutorials/intermediate/dist_tuto.html
import os
from pdb import run
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


"""Non-blocking point-to-point communication."""
def run_non_blocking(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
        req.wait()
    elif rank == 1:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
        req.wait()
    else:
        # Do nothing
        pass
    print('Rank ', rank, ' has data ', tensor[0])


"""All-Reduce example."""
def run_all_reduce(rank, size):
    group = dist.new_group(list(range(size)))
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--example', type=str, choices=['non_blocking', 'all_reduce'], default='non_blocking',
    )
    parser.add_argument('--world_size', type=int, default=2)
    args = parser.parse_args()

    if args.example == 'non_blocking':
        run = run_non_blocking
    elif args.example == 'all_reduce':
        run = run_all_reduce

    world_size = args.world_size
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()