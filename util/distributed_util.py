import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, Dataset, DistributedSampler
from .misc import get_rank, get_world_size
from typing import Iterable, Iterator, Optional, List, TypeVar, Union
from torch.utils.data.sampler import Sampler
import math


# __all__ = ["DistributedSampler", ]
T_co = TypeVar('T_co', covariant=True)


class Custom_DistributedSampler(DistributedSampler):
    """
    Customized DistributedSampler for the DataLoader.
    Mostly copied from torch.utils.data.distributed.DistributedSampler
    Just change the __iter__ function to repeat the dataset for each epoch multiple times.
    """
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, 
                 extend_factor: int = 20) -> None:
        super(Custom_DistributedSampler, self).__init__(dataset, 
                                                        num_replicas, rank, shuffle, seed, drop_last)

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if drop_last and len(self.dataset) % self.num_replicas:  
            # type: ignore[arg-type]

            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples_extend = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
                # type: ignore[arg-type]
            ) * extend_factor
        else:
            self.num_samples_extend = math.ceil(
                len(self.dataset)
                / self.num_replicas) * extend_factor
            # type: ignore[arg-type]
        self.total_size_extend = self.num_samples_extend * self.num_replicas
        self.extend_factor = extend_factor
    
    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices_extend = [torch.randperm(len(self.dataset), generator=g).tolist()
                       for _ in range(self.extend_factor)]  # type: ignore[arg-type]
        else:
            indices_extend = [list(range(len(self.dataset)))
                       for _ in range(self.extend_factor)]  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices_extend[0])   # type: ignore[arg-type]
            if padding_size <= len(indices_extend[0]):
                indices_extend = [indices + indices[:padding_size] for indices in indices_extend]
            else:
                indices_extend = [indices + (indices * 
                                             math.ceil(padding_size / 
                                                       len(indices)))[:padding_size]
                                  for indices in indices_extend]
        else:
            # remove tail of data to make it evenly divisible.
            indices_extend = [indices[:self.total_size] for indices in indices_extend] 
        assert all(len(indices) == self.total_size for indices in indices_extend)

        # subsample
        indices_extend = [indices[self.rank:self.total_size:self.num_replicas]
                          for indices in indices_extend]
        assert all(len(indices) == self.num_samples for indices in indices_extend)
        return iter([item for sublist in indices_extend for item in sublist])
    
    def __len__(self) -> int:
        return self.num_samples_extend


class Custom_BatchSampler(BatchSampler):
    """
    Customized BatchSampler for the DataLoader.
    Mostly copied from torch.utils.data.sampler.BatchSampler
    Just change the __iter__ function to repeat the dataset for each epoch multiple times.
    """
    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool) -> None:
        super(BatchSampler, self).__init__(sampler, batch_size, drop_last)
    
    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

