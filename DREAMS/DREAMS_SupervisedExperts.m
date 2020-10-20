%Expert-specific supervised learning scenario (i.e., inference only)
%DREAMS sleep spindles database
%
% Methodology:
%   - Partition data into 8 folds of trainig and test sets, e.g. 7 subjects
%   for training, one for testing
%   - Inside training set, estimate model parameters with complete data, (i.e.
%   labels are individual expert visual scores) according to robust
%   autoregressive hidden semi-Markov model (RARHSMM)
%   - Predict labels on 1 remaining subject (test set) via Viterbi algorithm 
%   with respect to i) same expert used for training, ii) the other expert
%   Report performance measures F1 score, MCC, false positive proportion.
%   There will be a total of 4 results in a confusion matrix-esque type of way
%   We also compute the performance measures for an aggregated model where
%   predicted labels are the result of logical AND between Viterbi outputs
%   from the two learned models (expert 1 and 2). Ground truth in this case
%   is the logical AND of expert labels also (as is common in the literature)
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

%% Model and algorithm parameters
clearvars
close all
clc

Fs = 50;                                % Sampling frequency (Hz)
nSubTotal = 8;                          % Total number of subjects in dataset
K = 2;                                  % Number of regimes/modes
normflag = true;                        % Flag to zscore input sequence
% Maximum duration of regimes (seconds)
% KEY: make sure to choose a value larger than the maximum sleep spindle 
% duration, otherwise you will end up distorting the sleep spindle duration
% distribution
dmaxSec = 30;                           
p = 5;                                  % Autoregressive order
dmin = p + 1;                           % Minimum duration of regimes in samples (should be greater than p)
dmax = round(dmaxSec*Fs);               % Maximum duration of regimes in samples
robustIni = true;                       % Flag to use robust linear regression for initial estimate of AR coefficients
% Performance measures
PerfTest = zeros(3, nSubTotal);
NLLTest = zeros(2, nSubTotal);          % Predicitive negative log-likelihoods
HMModelSupervised = cell(1, nSubTotal); % Fitted models (one per fold)
PerfTestExpert = cell(2, 2);
PerfTestExpert(:) = {nan(3, nSubTotal)};

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
labelsGTExpert1 = cell(1, nSubTotal);
labelsGTExpert2 = cell(1, nSubTotal);
for i = 1:nSubTotal
    ySeq{i} = Y(i, :);
    labelsGTExpert1{i} = ExpertLabels(i).Expert(1).VisualScores;
    if i <= 6           % subjects 7 and 8 don't have labels from expert 2
        labelsGTExpert2{i} = ExpertLabels(i).Expert(2).VisualScores;
    end    
    labelsGT{i} = ExpertLabels(i).VisualScoresUnion;
end

%% Learning with complete data (i.e., labels) and inference on test test
% Model hyperparameters (not learnable)
HMModelIni.Fs = Fs;
HMModelIni.StateParameters.K = K;
HMModelIni.normalize = normflag;
HMModelIni.robustIni = robustIni;
HMModelIni.ARorder = p;
HMModelIni.DurationParameters.dmax = dmax;
HMModelIni.DurationParameters.dmin = dmin;
% Loop over folds
for subj_i = 1:nSubTotal
    testSet = subj_i;                           % Test set - 1 subject
    trainSet = setdiff(1:nSubTotal, testSet);   % Training set - 7 subjects
    fprintf('Test Subject %d, AR order = %d \n', testSet, p)    
    % Expert 1 - Learning - complete data
    HMModel = HMModelIni;
    TrainingStructure(numel(trainSet)) = struct();
    for k = 1:numel(trainSet)
        TrainingStructure(k).y = ySeq{trainSet(k)};
        TrainingStructure(k).z = labelsGTExpert1{trainSet(k)};
    end
    % Learning/estimating model parameters
    HMModelExperts{1} = HMMLearningCompleteDataSleepSpindles(TrainingStructure, HMModel);     
    clear TrainingStructure   
    % Expert 2 - Learning - complete data
    HMModel = HMModelIni;
    trainSet = setdiff(trainSet, [7 8]);            % subjects 7 and 8 do not have labels from expert 2
    TrainingStructure(numel(trainSet)) = struct();
    for k = 1:numel(trainSet)
        TrainingStructure(k).y = ySeq{trainSet(k)};
        TrainingStructure(k).z = labelsGTExpert2{trainSet(k)};
    end   
    % Learning/estimating model parameters
    HMModelExperts{2} = HMMLearningCompleteDataSleepSpindles(TrainingStructure, HMModel);     
    clear TrainingStructure 
    ytest = ySeq{testSet};
    % 4 Cases
    labelsPredExpSep = zeros(2, numel(labelsGT{testSet}(p+1:end)));
    for i = 1:2
        % Inference. Test set
        labelsPred = HMMInference(ytest, HMModelExperts{i}, 'normalize', normflag);       
        loglike = HMMLikelihood(ytest, HMModelExperts{i}, 'normalize', normflag);
        NLLTest(i, subj_i) = -loglike;          % Predictive negative log-likelihood
        % Predicted labels
        labelsPred = labelsPred(p+1:end);
        labelsPredExpSep(i, :) = labelsPred;
        % Performance measures
        for j = 1:2
            if j == 1
                PerfTestExpert{i, j}(:, subj_i) = PerfMeasures(labelsGTExpert1{testSet}(p+1:end), labelsPred, Fs);
            elseif j == 2
                if testSet <= 6
                    PerfTestExpert{i, j}(:, subj_i) = PerfMeasures(labelsGTExpert2{testSet}(p+1:end), labelsPred, Fs);
                end
            end
        end     
    end
    % Aggregated case - logical AND of predicted labels
    clear labelsPred
    labelsPred = sum(labelsPredExpSep, 1)/2;
    labelsPred(labelsPred > 1) = 2;
    PerfTest(:, subj_i) = PerfMeasures(labelsGT{testSet}(p+1:end), labelsPred, Fs);
    HMModelSupervised{subj_i} = HMModelExperts;
end

%% Display results - Results in diagonal are referenced in paper
meanMCC = cell2mat(cellfun(@(x) mean(x(2,:), 'omitnan'), PerfTestExpert, 'UniformOutput', false));
T = array2table(meanMCC,...
    'RowNames', {'Expert1_Model', 'Expert2_Model'},...
    'VariableNames', {'Expert1_GroundTruth', 'Expert2_GroundTruth'});
fprintf('Average MCC between inference output of models trained with expert labels \n and ground truth (supervised scheme) \n \n')
disp(T)
% Results referenced in paper
fprintf('Average predictive measures of aggregated model. F1 score: %.4f, MCC: %.4f \n',...
    mean(PerfTest(1,:)), mean(PerfTest(2,:)))
fprintf('Average predictive negative log-likelihood. Expert1: %d, Expert2 %d \n',...
    round(mean(NLLTest(1,:))), round(mean(NLLTest(2,:))))   % Part of Table 2 in paper