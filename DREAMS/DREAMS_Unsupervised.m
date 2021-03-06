%Unsupervised learning scenario (i.e. learning and inference)
%DREAMS sleep spindles database
%
% Methodology:
%   - Partition data into 8 folds of trainig and test sets, e.g. 7 subjects
%   for training, one for testing
%   - Inside training set, estimate model parameters via EM algorithm 
%   according to robust autoregressive hidden semi-Markov model (RARHSMM)
%   - Predict labels on 1 remaining subject (test set) via Viterbi algorithm. 
%   Report performance measures F1 score, MCC, false positive proportion
%   - Loop over folds and report average performance on test set
% 
% IMPORTANT: To run this script, it is necessary to download the DREAMS
% sleep spindles dataset first AND reformat the data running the script 
% "reformatDREAMS.m". This will create the .mat file with EEG and labels to
% be used in the model
%
%Note: Requires Statistics and Machine Learning Toolbox
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
% KEY: make sure to choose a value larger than the maximum sleep spindle 
% duration, otherwise you will end up distorting the sleep spindle duration
% distribution
% dmaxSec = 30 requires a lot of memory and CPU so the results showed in
% the paper used high-performance computing (HPC) and the parallel computing
% toolbox, (i.e. parflag=true,dmaxSec=30)
dmaxSec = 30;                           
p = 5;                                  % Autoregressive order
dmin = p + 1;                           % Minimum duration of regimes in samples (should be greater than p)
dmax = round(dmaxSec*Fs);               % Maximum duration of regimes in samples
robustIni = true;                       % Flag to use robust linear regression for initial estimate of AR coefficients
% Performance measures
PerfTest = zeros(3, nSubTotal);
PerfTestAlt = zeros(3, nSubTotal);
NLLTest = zeros(1, nSubTotal);          % Predicitive negative log-likelihoods
HMModelUnsupervised = cell(1, nSubTotal); % Fitted models (one per fold)

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

%% Learning via EM algorithm and inference on test test
% Model hyperparameters (not learnable)
DurationParameters.dmax = dmax;
DurationParameters.dmin = dmin;
for subj_i = 1:nSubTotal
    testSet = subj_i;                           % Test set - 1 subject
    trainSet = setdiff(1:nSubTotal, testSet);   % Training set - 7 subjects
    fprintf('Test Subject %d, p = %d \n', testSet, p)
    % Learning - incomplete data
    yTrainSet = cell(1, numel(trainSet));
    % cell array for input sequences
    for k = 1:numel(trainSet)
        yTrainSet{k} = ySeq{trainSet(k)};
    end
    % Learning/estimating model parameters
    HMModel = HMMLearning(yTrainSet, K,...
        'ARorder', p, 'normalize', normflag,...
        'DurationParameters', DurationParameters,...
        'robustIni', robustIni, 'Fs', Fs, 'parallel', parflag);
    % Inference. Test set
    ytest = ySeq{testSet};
    [labelsPred, ~] = HMMInference(ytest, HMModel, 'normalize', normflag);
    % Performance measures
    [PerfTest(:, subj_i), PerfTestAlt(:, subj_i)] = PerfMeasures(labelsGT{testSet}(p+1:end), labelsPred(p+1:end), Fs);
    % Predictive negative log-likelihood
    loglike = HMMLikelihood(ytest, HMModel, 'normalize', normflag);
    NLLTest(subj_i) = -loglike;
    HMModelUnsupervised{subj_i} = HMModel;
    clear HHModel
end

%% Display results 
%Average MCC referenced in paper
fprintf('RESULTS: \n')
T = array2table([PerfTest(1:2,:) mean(PerfTest(1:2,:),2)],...
    'RowNames', {'F1', 'MCC'}, 'VariableNames',...
    {'S1','S2','S3','S4','S5','S6','S7','S8','Average'});
fprintf('F1 score and MCC between inference output and ground truth for supervised scheme \n \n')
disp(T)
fprintf('Average predictive negative log-likelihood: %d \n', round(mean(NLLTest)))      % Part of Table 2 in paper
fprintf('Event sensitivity: %.4f, event false positive rate: %.4f \n', mean(PerfTestAlt(1,:)), mean(PerfTestAlt(2,:)))