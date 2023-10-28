import torch
import random

class ManualDataLoader:
    def __init__(self, dataset, preds_train, targets, batch_size, shuffle=False):
        self.dataset = dataset
        self.targets = targets
        self.preds_train = preds_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.last_batch = len(dataset)%batch_size
        self.indices = list(range(len(dataset)))
        self.current_index = 0

        if self.shuffle:
            random.shuffle(self.indices)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.dataset) - self.last_batch and self.current_index < len(self.dataset):

            batch_indices = self.indices[self.current_index:self.current_index + self.last_batch]
            batch_data = torch.stack([torch.from_numpy(self.dataset[i]) for i in batch_indices])
            batch_targets = torch.stack([torch.from_numpy(self.targets[i]) for i in batch_indices])
            batch_preds = torch.stack([torch.from_numpy(self.preds_train[i]) for i in batch_indices])
            self.current_index += self.last_batch
            return batch_data, batch_preds, batch_targets
        if self.current_index == len(self.dataset):
            self.current_index = 0
            raise StopIteration

        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        batch_data = torch.stack([torch.from_numpy(self.dataset[i]) for i in batch_indices])
        batch_targets = torch.stack([torch.from_numpy(self.targets[i]) for i in batch_indices])
        batch_preds = torch.stack([torch.from_numpy(self.preds_train[i]) for i in batch_indices])
        self.current_index += self.batch_size
        return batch_data, batch_preds, batch_targets