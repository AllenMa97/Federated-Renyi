import torch
import numpy as np

from torch.utils.data import DataLoader, random_split

np.random.seed(42)


def get_FL_dataloader(dataset, num_clients, split_strategy="Uniform",
                      do_train=True, need_validation=True, batch_size=64,
                      do_shuffle=True, num_workers=0,
                      ):
    # Split training set into serval partitions to simulate the individual dataset
    partition_size = len(dataset) // num_clients
    lengths = [partition_size] * num_clients

    remainder = len(dataset) - (partition_size * num_clients)
    lengths[-1] += remainder

    if split_strategy == "Dirichlet":
        pass
    elif split_strategy == "Pathological":
        pass
    elif split_strategy == "Uniform":
        if do_train:
            client_datasets = random_split(dataset, lengths, torch.Generator().manual_seed(42))
            if need_validation:
                # Split each partition into train/val and create DataLoader
                trainloaders = []
                valloaders = []
                for ds in client_datasets:
                    len_val = len(ds) // 10  # 10 % validation set
                    len_train = len(ds) - len_val
                    lengths = [len_train, len_val]
                    ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
                    trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=do_shuffle, num_workers=num_workers))
                    valloaders.append(DataLoader(ds_val, batch_size=batch_size, num_workers=num_workers))
                return trainloaders, valloaders, client_datasets
            else:
                trainloaders = []
                for ds in client_datasets:
                    trainloaders.append(DataLoader(ds, batch_size=batch_size, shuffle=do_shuffle, num_workers=num_workers))
                return trainloaders, None, client_datasets
        else:
            testloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
            return testloader
    else:
        pass
