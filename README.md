# spindles-HMM

Code to replicate all results, tables, and figures of paper ["Robust autoregressive hidden semi–Markov models applied to EEG sleep spindles detection"](https://arxiv.org/abs/2010.08641) 

## Requirements
### MATLAB - Results, Tables and Figure 2
* MATLAB R2016b or newer (the implemented algorithms require broadcasting capabilities)
* Statistics and Machine Learning Toolbox
* Signal Processing Toolbox
* Parallel Computing Toolbox (optional, but extremely recommended for EM-based learning)
### Python - Figure 1
* python 3
* jupyter notebook
* matplotlib
* [daft](https://docs.daft-pgm.org/en/latest/) to plot the probabilistic graphical model
### Data
* DREAMS sleep spindles dataset

## Instructions
1. Add "Main_Code" folder to MATLAB path
2. Download **DREAMS sleep spindles** dataset. A reliable source is [here](https://zenodo.org/record/2650142) (make sure to download the *DatabaseSpindles.rar* file). Unpack .rar file.
3. Run `reformatDREAMS.m` script to reformat and downsample the EEG and expert labels
4. Algorithms/scripts in the "DREAMS" folder are now ready to be run. These are the scripts that replicate results, tables, and Figure 2
5. (Optional) Set up a python environment with matplotlib and daft to run `RARHSMM_GraphicalModel.ipynb` on a jupyter notebook

PS1: All main code is MATLAB, python is only used to render the probabilistic graphical model of Figure 1

PS2: The **DREAMS sleep spindles** dataset used to be hosted [here](http://www.tcts.fpms.ac.be/~devuyst/Databases/DatabaseSpindles/) but that
website has been down for quite some time now
