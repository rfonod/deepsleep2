"""On-the-fly signal-space augmentations for multi-channel PSG recordings.

Every transform is a callable ``(recording, arousal) -> (recording, arousal)`` where
``recording`` is a ``(n_channels, n_samples)`` tensor and ``arousal`` a ``(1, n_samples)``
tensor. Apart from ``ToTensor``, they are applied to the training set only.

These classes live in a module (rather than in the notebook) so that DataLoader worker
processes can unpickle them on platforms that use the ``spawn`` start method (macOS,
Windows).
"""

import torch
import torch.nn.functional as F

# Canonical order of the 13 PhysioNet/CinC 2018 PSG channels:
#  0-5: EEG (F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1)
#    6: EOG (E1-M2)
#    7: EMG (Chin)
#  8-9: respiratory effort (ABD, Chest)
#   10: Airflow
#   11: SaO2
#   12: ECG
ALL_CHANNELS = tuple(range(13))


class MagScale(object):
    """Rescale the magnitude of all PSG channels with the same random scale factor"""

    def __init__(self, low = 0.8, high = 1.25):
        self.low = low
        self.high = high

    def __call__(self, recording, arousal):
        scale = self.low + torch.rand(1)*(self.high - self.low)
        recording = scale*recording

        return recording, arousal


class MagScaleRandCh(object):
    """Rescale the magnitude of each PSG channel with its own random scale factor"""

    def __init__(self, low = 0.8, high = 1.25):
        self.low = low
        self.high = high

    def __call__(self, recording, arousal):
        n_channels = recording.shape[0]
        scales = self.low + torch.rand(n_channels).view(-1,1)*(self.high - self.low)
        recording = scales*recording

        return recording, arousal


class RandShuffle(object):
    """Randomly reshuffle related PSG channels: the EEG channels among themselves and
       the two respiratory-effort channels (ABD, Chest) among themselves. All other
       channels keep their position.

       `channels` are the original PhysioNet channel indices that were selected (the
       `CHANNELS` hyperparameter), so the shuffle also works on channel subsets."""

    GROUPS = ((0, 1, 2, 3, 4, 5),  # EEG
              (8, 9))              # respiratory effort

    def __init__(self, channels = ALL_CHANNELS):
        channels = list(channels)
        self.n_channels = len(channels)
        groups = [torch.LongTensor([pos for pos, ch in enumerate(channels) if ch in group])
                  for group in self.GROUPS]
        self.groups = [g for g in groups if len(g) > 1]

    def __call__(self, recording, arousal):
        order = torch.arange(self.n_channels)
        for positions in self.groups:
            order[positions] = positions[torch.randperm(len(positions))]

        return recording.index_select(0, order), arousal


class AddRandGaussian2All(object):
    """Add zero-mean Gaussian noise to all PSG channels (std 0.1 for z-normalized data,
       otherwise 10% of each channel's standard deviation)"""

    def __init__(self, z_norm = True):
        self.z_norm = z_norm

    def __call__(self, recording, arousal):
        if self.z_norm:
            std_dev = 0.1
        else:
            std_dev = 0.1*torch.std(recording, 1, keepdim = True)
        recording = recording + std_dev*torch.randn(recording.shape)

        return recording, arousal


class InjectRandGaussian(object):
    """Replace a randomly selected PSG channel with a standard Gaussian noise sequence"""

    def __call__(self, recording, arousal):
        n_channels = recording.shape[0]
        ri = torch.randint(0, n_channels, (1,)).type(torch.long)
        recording[ri] = torch.normal(mean = 0, std = 1, size = (1, recording.shape[1]))

        return recording, arousal


def _fit_length(x, length, pad_value):
    """Center-crop or center-pad the last dimension of `x` to `length` samples
       (padding convention matches `utils.preprocess`)"""
    current = x.shape[-1]
    if current > length:
        start = (current - length)//2
        return x[..., start:start + length]
    padd = length - current
    return F.pad(x, (padd//2 + padd%2, padd//2), value = pad_value)


class TimeScale(object):
    """Stretch/shrink the recording and arousal signals with a random time scale factor
       drawn uniformly from [1 - interval/2, 1 + interval/2], then center-crop/pad them
       back to the original length (recording padded with 0, arousal with -1 = unscored)"""

    def __init__(self, interval = 0.2):
        self.interval = interval

    def __call__(self, recording, arousal):
        length = recording.shape[-1]
        scale = float(1 + self.interval*(torch.rand(1) - 0.5))
        recording = F.interpolate(recording.unsqueeze(0), scale_factor = scale,
                                  mode = 'linear', align_corners = False,
                                  recompute_scale_factor = True).squeeze(0)
        arousal = F.interpolate(arousal.reshape(1, 1, -1), scale_factor = scale,
                                mode = 'nearest', recompute_scale_factor = True).reshape(1, -1)

        return _fit_length(recording, length, 0.), _fit_length(arousal, length, -1.)


class ToTensor(object):
    """Convert the recording and arousal signals to Tensors"""

    def __call__(self, recording, arousal):
        return torch.Tensor(recording), torch.Tensor(arousal)


class Compose:
    """Stack multiple transforms together"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, recording, arousal):
        for t in self.transforms:
            recording, arousal = t(recording, arousal)

        return recording, arousal


def build_train_transforms(names, channels = ALL_CHANNELS, z_norm = True):
    """Build the training transform pipeline from the `TRANSFORMS` hyperparameter list"""
    factories = {
        'MagScale'            : lambda: MagScale(),
        'MagScaleRandCh'      : lambda: MagScaleRandCh(),
        'TimeScale'           : lambda: TimeScale(),
        'RandShuffle'         : lambda: RandShuffle(channels),
        'AddRandGaussian2All' : lambda: AddRandGaussian2All(z_norm),
        'InjectRandGaussian'  : lambda: InjectRandGaussian(),
    }
    unknown = [name for name in names if name not in factories]
    if unknown:
        raise ValueError('Unknown transform(s) {}; available: {}'.format(unknown, list(factories)))

    return Compose([ToTensor()] + [factories[name]() for name in names])
