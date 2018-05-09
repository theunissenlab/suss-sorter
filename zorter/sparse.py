"""
Code for fitting a sparse representations of spike waveforms.

(Want it to be independent of sample rate)
"""

import numpy as np
import torch
from scipy.interpolate import interp1d
from sklearn.decomposition import SparseCoder
from sklearn.preprocessing import normalize
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataset import random_split


def resample_data(data,
        old_sample_rate=None, old_center_bin=None, old_n_bins=None,
        new_sample_rate=None, new_center_bin=None, new_n_bins=None
    ):

    t = np.linspace(
            0,
            (old_n_bins - 1) / old_sample_rate,
            old_n_bins
    ) - (old_center_bin / old_sample_rate)

    t_new = np.linspace(
            0,
            (new_n_bins - 1) / new_sample_rate,
            new_n_bins
    ) - (new_center_bin / new_sample_rate)

    if np.min(t_new) < np.min(t) or np.max(t_new) > np.max(t):
        raise Exception("Resample boundaries exceeded")

    dtype = data.dtype
    return interp1d(t, data, kind="cubic", axis=1)(t_new).astype(dtype)



class AutoEncoder(nn.Module):
    def __init__(self, dictionary_size, features):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Linear(features, dictionary_size)

    def forward(self, x): 
        x = self.encoder(x)
        return torch.matmul(x, self.encoder.weight)


class SparseSpikeModel(object):
    def __init__(self, n_components):
        self.n_components = n_components
        
    def train(self, data, batch_size=1000, max_iters=100, norm=1,
             learning_rate=1e-3, weight_decay=1e-5):
        self.data = data
        self.model = AutoEncoder(self.n_components, data.shape[1]).cuda()
        
        self._mse = nn.MSELoss()
        self._optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self._loss = []
        
        for iter_idx in range(max_iters):
            for batch_start in range(0, len(data), batch_size):
                selector = slice(batch_start, batch_start + batch_size)
            
                x = data[selector]
                x = Variable(torch.from_numpy(x)).cuda()
                output = self.model(x)
                
                loss = self._mse(output, x) + self.model.encoder.weight.norm(norm)
                
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

            self._loss.append(loss)
            
        return range(max_iters), self._loss

    @property
    def dictionary(self):
        return self.model.encoder.weight.detach().cpu().numpy()


class Dictionary(object):

    def __init__(self, components, sample_rate, center_bin):
        self.components = components
        self.sample_rate = sample_rate
        self.center_bin = center_bin
        self.n_times = components.shape[1]

    def resample(self, sample_rate=None, center_bin=None, n_bins=None):
        """
        if t_before and t_after are not specified, try to
        resample over the same time range
        """
        args = [sample_rate, center_bin, n_bins]

        if any(args) and not all(args):
            raise Exception("Must specify sample_rate, center_bin, and n_bins together")

        if not any(args):
            return self.components

        return resample_data(self.components,
                old_sample_rate=self.sample_rate,
                old_center_bin=self.center_bin,
                old_n_bins=self.n_times,
                new_sample_rate=sample_rate,
                new_center_bin=center_bin,
                new_n_bins=n_bins)

    def encode(self, data, sample_rate=None, center_bin=None, n_bins=None):
        return SparseCoder(
            normalize(
                self.resample(sample_rate, center_bin, n_bins),
                axis=1
            )
        ).transform(data)


def train(
        waveforms,
        sample_rate,
        center_bin=None,
        n_components=30,
        batch_size=10000,
        max_iters=30
    ):
    """Train sparse coder on waveforms, return Dictionary"""
    ssm = SparseSpikeModel(n_components=n_components)

    n_bins = waveforms.shape[1]
    if center_bin is None:
        center_bin = n_bins // 2

    iters, loss = ssm.train(
        waveforms,
        batch_size=batch_size,
        max_iters=max_iters
    )

    return Dictionary(ssm.dictionary, sample_rate, center_bin), iters, loss
