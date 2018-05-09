# solid-garbanzo

Automated spike sorting for single channel electrophysiology

# Dependencies

This project requires Python3.5+. Recommended to use a virtual environment (setup with `virtualenv env -p python3`)

Install python dependencies by installing from the requirements.txt file: `pip install -r requirements.txt`

* `ffmpeg` for creating HTML5 animated plots in notebooks

* `imagemagick` for generating animated plots in gif format

# Description

Nearly fully automated spike sorting of extracellular signals on single channels for non-stationary units.

The functions are divided into two parts: spike extraction from a raw extracellular signal, and spike sorting from waveform shapes and arrival times.

## Spike detection

TBD

## Spike sorting

TBD

# References

Magland JF, Barnett AH. Unimodal clustering using isotonic regression: ISO-SPLIT. arXiv, 2015

Dhawale AK, Poddar R, Wolff BE, Normand VA, Kopelowitz E, Olveczky BP. Automated long-term recording and analysis of neural activity in behaving animals. eLIFE, 2017

Hill DN, Mehta SB, Kleinfeld D. Quality Metrics to Accompany Spike Sorting of Extracellular Signals. JNeurosci, 2011
