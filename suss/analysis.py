"""
Stimuli are defined by arrival times and stimulus properties

This module provides convenience methods to look at responses
of ClusterDatasets (i.e. putative clusters) to stimuli by aligning to
stimulus arrival times and filtering stimuli by their properties
"""

import numpy as np


def align(cluster, stimulus_times, t_start, t_stop):
    aligned_spikes = []
    aligned_waveforms = []
    for stimulus_time in stimulus_times:
        _t_start = stimulus_time + t_start
        _t_stop = stimulus_time + t_stop
        window = cluster.select((cluster.times >= _t_start) & (cluster.times < _t_stop))

        aligned_spikes.append(window.times - stimulus_time)
        aligned_waveforms.append(window.waveforms)

    return aligned_spikes, aligned_waveforms
