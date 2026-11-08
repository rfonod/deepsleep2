"""PyTorch dataset for the preprocessed PhysioNet/CinC 2018 PSG recordings.

Defined in a module (rather than in the notebook) so that DataLoader worker processes
can unpickle it on platforms that use the ``spawn`` start method (macOS, Windows).
"""

import os

from torch.utils.data import Dataset

from utils import get_record


class CustomDataset(Dataset):
    """Custom dataset for multichannel PSG recordings and sleep arousal annotations"""

    def __init__(self, data_path, records, channels = [*range(13)], z_norm = False, transforms = None):
        self.data_path = data_path
        self.records = records
        self.channels = channels
        self.n_channels = len(channels)
        self.normalization = 2 if z_norm else 1
        self.transforms = transforms

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.records[idx])
        recording, arousal = get_record(file_path, self.channels, self.normalization)

        if self.transforms is not None:
            recording, arousal = self.transforms(recording, arousal)

        return recording, arousal, idx

    def get_record_name(self, idx):
        return self.records[idx]
