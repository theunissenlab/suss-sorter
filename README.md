# solid-garbanzo

Automated spike sorting for single channel electrophysiology

![Animated spikes](static/animated-2d.gif)

*Example of results after running automated clustering and manual cluster selection*

## Installation and Dependencies

This project requires Python3.5+ (I recommend using a virtualenv.)

#### From your python environment

```
git clone https://github.com/kevinyu/solid-garbanzo.git
cd solid-garbonzo
python setup.py build
python setup.py install
```
*This should install the package as well as its depenencies*

#### Other dependencies

* `ffmpeg` for creating HTML5 animated plots in notebooks

* `imagemagick` for generating animated plots in gif format

## Run

### Spike detection

TBD

### Spike sorting

Prepare your data into arrays of spike waveforms and spike arrival times. The sorting function takes three parameters

`times`: 1-dimensional array of length N\_spikes, each element is the arrival time of a spike in seconds

`waveforms`: 2-dimensional array of shape (N\_spikes, N\_samples), each row is a spike waveform

`sample_rate`: sampling rate used to generate `waveforms` in Hz

Usage example:

```(python)
from zorter.sort import sort
sorted_node = sort(times, waveforms, sample_rate=sample_rate)
```

## Description

Nearly fully automated spike sorting of extracellular signals on single channels for non-stationary units.

The functions will be divided into two parts: spike extraction from a raw extracellular signal, and spike sorting from waveform shapes and arrival times.

### Spike detection

TBD

### Spike sorting

TBD

# References

Magland JF, Barnett AH. Unimodal clustering using isotonic regression: ISO-SPLIT. arXiv, 2015

Dhawale AK, Poddar R, Wolff BE, Normand VA, Kopelowitz E, Olveczky BP. Automated long-term recording and analysis of neural activity in behaving animals. eLIFE, 2017

Hill DN, Mehta SB, Kleinfeld D. Quality Metrics to Accompany Spike Sorting of Extracellular Signals. JNeurosci, 2011
