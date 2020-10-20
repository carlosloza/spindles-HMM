%Script to plot Figure 2 of paper
%Unsupervised learning scenario (i.e. learning only)
%DREAMS sleep spindles database
%
% Methodology:
%   - Estimate model parameters via EM algorithm according to robust 
%   autoregressive hidden semi-Markov model (RARHSMM) using all 8 subjects
%   in the dataset
% 
% IMPORTANT: To run this script, it is necessary to download the DREAMS
% sleep spindles dataset first AND reformat the data running the script 
% "reformatDREAMS.m". This will create the .mat file with EEG and labels to
% be used in the model
%
%Note: Requires Statistics and Machine Learning toolbox, Signal Processing
%toolbox and parallel computing toolbox 
%Author: Carlos Loza (carlos.loza@utexas.edu)
%https://github.com/carlosloza/spindles-HMM

%% Model and EM parameters
clearvars
close all
clc

Fs = 50;                                % Sampling frequency (Hz)
nSubTotal = 8;                          % Total number of subjects in dataset
K = 2;                                  % Number of regimes/modes
normflag = true;                        % Flag to zscore input sequence
parflag = true;                        % Flag to use parallel computing toolbox
% Maximum duration of regimes (seconds)
% KEY: to reproduce Figure 2, dmaxSec must be equal to 30. The resulting 
% algorithm will require a lot of memory and CPU so parallel computing is 
% strongly recommended, (i.e, parflag=true)
dmaxSec = 30;                           
p = 5;                                  % Autoregressive order
dmin = p + 1;                           % Minimum duration of regimes in samples (should be greater than p)
dmax = round(dmaxSec*Fs);               % Maximum duration of regimes in samples
robustIni = true;                       % Flag to use robust linear regression for initial estimate of AR coefficients

%% Load previously formatted data
% This can be replaced by a variable with the location of the .mat file
[file,path] = uigetfile('*.mat', 'Select "DREAMS_SleepSpindles.mat" file');
if isequal(file,0)
   disp('Canceled by user');
else
    load(fullfile(path,file));
end
% Convert to cells
ySeq = cell(1, nSubTotal);
labelsGT = cell(1, nSubTotal);
for i = 1:nSubTotal
    ySeq{i} = Y(i, :);
    labelsGT{i} = ExpertLabels(i).VisualScoresUnion;
end

%% Learning via EM algorithm
% Model hyperparameters (not learnable)
DurationParameters.dmax = dmax;
DurationParameters.dmin = dmin;
% Learning/estimating model parameters (All subjects for training)
HMModel = HMMLearning(ySeq, K,...
    'ARorder', p, 'normalize', normflag,...
    'DurationParameters', DurationParameters,...
    'robustIni', robustIni, 'Fs', Fs, 'parallel', parflag);

%% Figure 2 in paper
figure
% Magnitude of power spectral density
[H, F] = freqz(1, [1; HMModel.ObsParameters.meanParameters(1).Coefficients], 1024, Fs);
subplot(2,2,1:2)
plot(F, 20*log10(abs(H)), '--b')
hold on
[H, F] = freqz(1, [1; HMModel.ObsParameters.meanParameters(2).Coefficients], 1024, Fs);
plot(F, 20*log10(abs(H)), 'r')
ylabel('Magnitude (dB)')
xlabel('Frequency (Hz)')
legend('Non-spindle', 'Spindle')
% Additive observation noise (generalized t)
subplot(2,2,3)
x = -2:.1:2;
pd1 = makedist('tLocationScale', 0, HMModel.ObsParameters.sigma(1), HMModel.ObsParameters.nu(1));
pdf_t1 = pdf(pd1, x);
plot(x, pdf_t1, '--b')
hold on
pd2 = makedist('tLocationScale', 0, HMModel.ObsParameters.sigma(2), HMModel.ObsParameters.nu(2));
pdf_t2 = pdf(pd2, x);
plot(x, pdf_t2, 'r')
legend('Non-spindle', 'Spindle')
subplot(2,2,4)
bar(1/Fs:1/Fs:dmaxSec, HMModel.DurationParameters.PNonParametric(2,:))
xlim([0 2])
ylabel('Probability')
xlabel('Duration (s)')