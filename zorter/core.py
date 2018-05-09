import numpy as np


class DataNode(object):
    """Abstraction for hierarchical organization of spike data

    DataNode can reference children and appears as the
    median times and waveforms for each of its children.

    DataNode with children can be flattened with .flatten() which
    eliminates any number of intermediate layers
    """

    def __init__(self, times=None, waveforms=None, sample_rate=30000.0, children=None):
        # TODO: include more annotations such as labels, and idx values referencing master dataset
        # .... or.... only use indexes into a master node
        if children is not None and (times is not None or waveforms is not None):
            raise ValueError("Can only specify either child nodes or times and waveforms")

        if children is None and (times is None or waveforms is None):
            raise ValueError("Must specify either children or times and waveforms")

        if children is not None:
            sample_rates = [child.sample_rate for child in children]
            if len(np.unique(sample_rates)) > 1:
                raise ValueError("Cannot merge nodes with different sample rates: {}".format(set(sample_rates)))

            self.sample_rate = sample_rates[0]
            times = np.array([child.time for child in children])
            waveforms = np.array([child.waveform for child in children])
            self.times, self.waveforms, self.children = self._sort(np.array(times), np.array(waveforms), np.array(children))
        else:
            self.children = None
            self.sample_rate = sample_rate
            self.times, self.waveforms = self._sort(np.array(times), np.array(waveforms))

    def __len__(self):
        return len(self.times)

    def _sort(self, times, *args):
        _sorter = np.argsort(times)
        return (times[_sorter],) + tuple([arr[_sorter] for arr in args])

    def select(self, selector):
        if self.children is not None:
            return DataNode(children=self.children[selector])
        else:
            return DataNode(
                    times=self.times[selector],
                    waveforms=self.waveforms[selector],
                    sample_rate=self.sample_rate
            )

    def windows(self, dt=0.5 * 60.0):
        for t_start in np.arange(0.0, np.max(self.times), dt):
            t_stop = t_start + dt
            selector = np.where((self.times >= t_start) & (self.times < t_stop))[0]
            if self.children is not None:
                yield (
                    DataNode(children=self.children[selector]),
                    (t_start, t_stop)
                )
            else:
                yield (
                    DataNode(
                        self.times[selector],
                        self.waveforms[selector],
                        self.sample_rate
                    ),
                    (t_start, t_stop)
                )

    def flatten(self, depth=None):
        """Flatten the tree"""
        if self.children is None or (depth is not None and depth == 0):
            return self

        flattened_children = [child.flatten(depth=None if depth is None else depth - 1) for child in self.children]

        if np.unique([child.children is None for child in flattened_children]).size != 1:
            raise Exception("Children have different depths... something went wrong")

        if any([child.children is not None for child in flattened_children]):
            children = np.concatenate([child.children for child in flattened_children])
            return DataNode(children=children)
        else:
            times = np.concatenate([child.times for child in flattened_children])
            waveforms = np.vstack([child.waveforms for child in flattened_children])
            
            times, waveforms = self._sort(times, waveforms)
            return DataNode(times, waveforms, self.sample_rate)
    
    @property
    def time(self):
        return np.median(self.times)

    @property
    def waveform(self):
        return np.median(self.waveforms, axis=0)
