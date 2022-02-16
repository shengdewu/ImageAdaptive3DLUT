import torch.utils.data
import multiprocessing
from engine.samplers.distributed_sampler import TrainingSampler
from engine.data.common import ToIterableDataset


def create_iterable_data_loader(dataset, batch_size, num_workers, collate_fn=None):
    sampler = TrainingSampler(len(dataset))
    dataset = ToIterableDataset(dataset, sampler)

    max_workers = multiprocessing.cpu_count()
    num_workers = num_workers if num_workers < max_workers else max_workers

    return torch.utils.data.DataLoader(
                                        dataset=dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        pin_memory=True,
                                        collate_fn=collate_fn,
                                        drop_last=True)


def create_distribute_iterable_data_loader(dataset, batch_size, rank, world_size, num_workers, collate_fn=None):
    sampler = TrainingSampler(len(dataset), rank=rank, world_size=world_size)
    dataset = ToIterableDataset(dataset, sampler)

    max_workers = multiprocessing.cpu_count()
    num_workers = num_workers if num_workers < max_workers else max_workers

    return torch.utils.data.DataLoader(
                                        dataset=dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        pin_memory=True,
                                        collate_fn=collate_fn,
                                        drop_last=True)


def create_data_loader(dataset, batch_size, num_workers, collate_fn=None):

    sampler = torch.utils.data.RandomSampler(dataset)

    max_workers = multiprocessing.cpu_count()
    num_workers = num_workers if num_workers < max_workers else max_workers

    return torch.utils.data.DataLoader(
                                       dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       sampler=sampler,
                                       num_workers=num_workers,
                                       drop_last=True,
                                       pin_memory=True,
                                       collate_fn=collate_fn)


def create_distribute_data_loader(dataset, batch_size, rank, world_size, num_workers, collate_fn=None):

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    max_workers = multiprocessing.cpu_count()
    num_workers = num_workers if num_workers < max_workers else max_workers

    return torch.utils.data.DataLoader(
                                       dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=num_workers,
                                       pin_memory=True,
                                       drop_last=True,
                                       sampler=sampler,
                                       collate_fn=collate_fn)
