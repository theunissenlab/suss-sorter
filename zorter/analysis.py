"""
Stimuli are defined by arrival times and stimulus properties

This module provides convenience methods to look at responses
of DataNodes (i.e. putative clusters) to stimuli by aligning to
stimulus arrival times and filtering stimuli by their properties
"""

import numpy as np

from .core import DataNode


def align(node, stimulus_times, t_before, t_after):
    aligned_nodes = []
    for stimulus_time in stimulus_times:
        aligned_nodes.append(node.align(stimulus_time, before=t_before, after=t_after))

    return aligned_nodes
