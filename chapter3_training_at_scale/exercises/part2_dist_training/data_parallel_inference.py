import tqdm
import argparse
import os
import logging
import time
import random
import string

import torch.distributed as dist
from torch.distributed import ReduceOp

import torch as t
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
from torchvision.io import read_image

assert torch.cuda.device_count() > 0  # make sure we have GPUs

import sys

CLUSTER_SIZE = 2  # the number of separate compute nodes we have
# WORLD_SIZE = int(sys.argv[1])  # the number of processes we want to launch - this is usually equal to the number of GPUs we have on this machine
WORLD_SIZE = 1  # the number of processes we want to launch - this is usually equal to the number of GPUs we have on this machine
TOTAL_RANKS = CLUSTER_SIZE * WORLD_SIZE
UNIGPU = torch.cuda.device_count() == 1  # remember to use the patched NCCL binary if you are using colab/practicing on a single GPU. You might need to compile https://github.com/pranavgade20/nccl-unigpu if you aren't using colab

def main(args):
    rank = args.rank
    torch.manual_seed(0)

    # world_size = args.world_size
    logging.basicConfig(format=f'[rank {args.rank}] %(message)s')
    logging.getLogger().setLevel(logging.DEBUG)
    logging.warning(f'hello')
    logging.warning(f"======={WORLD_SIZE=}")
    # you can use a variety of backends - nccl is the fastest but requires you to have a GPU; gloo is slower but also sometimes works on CPUs (see https://pytorch.org/docs/stable/distributed.html#backends)
    # dist.init_process_group(backend='nccl', init_method=f'file:///tmp/{"".join(random.choice(string.ascii_letters) for _ in range(10))}', world_size=WORLD_SIZE, rank=args.rank)
    # TODO: change init_method to the ip of cluster 0
    dist.init_process_group(backend='nccl', init_method=f'tcp://138.2.231.24:12345', world_size=WORLD_SIZE, rank=args.rank)  # this should be a globally accessible IP
    logging.warning(f'distributed.is_initialized {torch.distributed.is_initialized()}')
    logging.warning(f'distributed.is_mpi_available {torch.distributed.is_mpi_available()}')
    logging.warning(f'distributed.is_nccl_available {torch.distributed.is_nccl_available()}')
    logging.warning(f'distributed.is_gloo_available {torch.distributed.is_gloo_available()}')
    logging.warning(f'distributed.is_torchelastic_launched {torch.distributed.is_torchelastic_launched()}')

    data_root = "/root/dataset"

    device = f"cuda:{0 if UNIGPU else rank}"
    resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(device).eval()
    file_mappings = json.load(open(f'{data_root}/file_mappings_imagenet.json'))
    logging.warning("Loading Data:")

    imagenet_valset = list((lambda k=k: read_image(f'{data_root}/val/{k}.JPEG'), int(v)) for k, v in file_mappings.items())
    # imagenet_valset = Subset(imagenet_valset, indices=range(rank, len(imagenet_valset), TOTAL_RANKS))
    imagenet_valset = Subset(imagenet_valset, indices=range(rank, 1000, TOTAL_RANKS))
    imagenet_valset = [(x(), y) for x, y in tqdm.tqdm(imagenet_valset, desc=f'[rank {rank}]')]
    imagenet_valset = [(torch.cat([x,x,x],0) if x.shape[0] == 1 else x, y) for x, y in imagenet_valset]
    transform = torch.jit.script(torch.nn.Sequential(transforms.ConvertImageDtype(torch.float32),transforms.Resize((224, 224)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])))
    logging.warning("Transforming Data:")
    imagenet_valset = [(transform(x), y) for x, y in tqdm.tqdm(imagenet_valset, desc=f'[rank {rank}]')]

    time.sleep(1)

    # your code starts here - everything before this is setup code
    imagenet_dataloader = DataLoader(imagenet_valset, batch_size=32, shuffle=False)
    loss = []
    accuracy = []
    print(f"{torch.cuda.device_count()=}")
    loss_fn = t.nn.CrossEntropyLoss()

    losses = []
    accuracies = []

    batch = 0
    start_time = time.time()
    for data, labels in imagenet_dataloader:
        data, labels = data.to(device), labels.to(device)
        batch_size = labels.size(0)
        print(f"[rank {rank}] | batch {batch} | {data.shape}")

        logits = resnet34(data)

        loss = loss_fn(logits, labels)
        logging.warning(f"{loss=}")

        preds = logits.argmax(dim=-1)
        accuracy = (preds == labels).sum() / batch_size

        dist.reduce(loss, dst=0, op=ReduceOp.SUM)
        dist.reduce(accuracy, dst=0, op=ReduceOp.SUM)

        losses.append(loss / TOTAL_RANKS)
        accuracies.append(accuracy / TOTAL_RANKS)

        batch += 1

    if rank == 0:
        mean_loss = t.stack(losses).mean()
        mean_accuracy = t.stack(accuracies).mean()

        print(f"{losses=}, {len(losses)=}")
        print(f"{mean_loss=}, {mean_accuracy=}")


    print("--- %s seconds ---" % (time.time() - start_time))
    logging.warning("===== END OF MY CODE")

    # your code ends here - this is followed by teardown code
    dist.barrier()  # wait for all process to reach this point
    dist.destroy_process_group()


if __name__ == '__main__':
    args = argparse.Namespace(cluster_id=0, rank=-1, world_size=WORLD_SIZE)
    if args.rank == -1:
        # we are the parent process, spawn children
        for rank in range(args.cluster_id, TOTAL_RANKS, CLUSTER_SIZE):
            pid = os.fork()
            if pid == 0:
                # child process
                args.rank = rank
                main(args=args)
                break
    # wait for all children to finish
    if args.rank == -1:
        os.waitid(os.P_ALL, 0, os.WEXITED)
# %%
