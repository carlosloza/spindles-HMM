%% Unsupervised setting, i.e. Learning
% Use logsumexp implementation because of artifacts!
% Methodology:
% 7 out of 8 subjects will be the training set
% Inside training set, learn parameters with a Guassian likelihood on the
% observations (sensible choice because initial conditions are based on a 
% Gaussian AR Kalman filter)
% After the Gaussian model is learned, infer most likely sequences (via Viterbi)
% on THE SAME training set.
% Use this initial segmentation as pseudo-ground truth to learn a model (complete data)
% with a different observation likelihood (in theory it could still be Gaussian)
% Use this model as the initial conditions to the EM algorithm where the
% observation likelihood is the same as the one chosen on the previous step
% Predict labels on 1 remaining subject (test set) using final model
% 
% All recording will be downsampled or upsampled to have Fs=50Hz
% This function takes so long that it needs to be ran in a
% subject-by-subject basis

clearvars
close all
clc

Fs = 50;
nSubTotal = 8;
K = 2;
normflag = 1;

dmaxSec = 2;
p = 5;
dmin = p + 1;
dmax = round(dmaxSec*Fs);
robustIni = true;            

PerfTest = zeros(3, nSubTotal);
PerfTestAlt = zeros(3, nSubTotal);
NLLTest = zeros(1, nSubTotal);
HMModelSupervised = cell(1, nSubTotal);
%% Data structure
[file,path] = uigetfile('*.mat', 'Select "DREAMS_SleepSpindles.mat" file');
if isequal(file,0)
   disp('Canceled by user');
else
    load(fullfile(path,file));
end


% Datapath = uigetdir(pwd, 'Select folder of "DREAMS_SleepSpindles.mat" file');
% load([Datapath '/DREAMS_SleepSpindles.mat'])
ySeq = cell(1, nSubTotal);
labelsGT = cell(1, nSubTotal);
for i = 1:nSubTotal
    ySeq{i} = Y(i, :);
    labelsGT{i} = ExpertLabels(i).VisualScoresUnion;
end
% ySeq = cell(1, nSubTotal);
% labelsGT = cell(1, nSubTotal);
% for i = 1:nSubTotal
%     load(['Data/Subject' num2str(i) '_Fs' num2str(100) '.mat'])
%     ySeq{i} = y;
%     labelsGT{i} = labels;    
%     ySeq{i} = downsample(ySeq{i}, 2);
%     labelsGT{i} = downsample(labelsGT{i}, 2);
% end
%%
ObsParameters.meanModel = 'Linear';
DurationParameters.dmax = dmax;
DurationParameters.dmin = dmin;
DurationParameters.model = 'NonParametric';

for subj_i = 1:nSubTotal
    testSet = subj_i;
    trainSet = setdiff(1:nSubTotal, testSet);
    fprintf('Test Subject %d, p = %d \n', testSet, p)
    
    yTrainSet = cell(1, numel(trainSet));
    for k = 1:numel(trainSet)
        yTrainSet{k} = ySeq{trainSet(k)};
    end
    HMModel = HMMLearning(yTrainSet, K,...
        'ARorder', p, 'normalize', normflag,...
        'ObsParameters', ObsParameters,...
        'DurationParameters', DurationParameters,...
        'robustIni', true, 'Fs', Fs, 'parallel', false);
    
    % Test
    ytest = ySeq{testSet};
    [labelsPred, ~] = HMMInference(ytest, HMModel, 'normalize', normflag);
    [PerfTest(:, subj_i), PerfTestAlt(:, subj_i)] = PerfMeasures(labelsGT{testSet}(p+1:end), labelsPred(p+1:end), Fs);
    
    % Predictive log-likelihood
    loglike = HMMLikelihood(ytest, HMModel, 'normalize', normflag);
    NLLTest(subj_i) = -loglike;
    HMModelSupervised{subj_i} = HMModel;
end

%% Display results
fprintf('RESULTS: \n')
T = array2table([PerfTest(1:2,:) mean(PerfTest(1:2,:),2)],...
    'RowNames', {'F1', 'MCC'}, 'VariableNames',...
    {'S1','S2','S3','S4','S5','S6','S7','S8','Average'});
fprintf('F1 score and MCC between inference output and ground truth for supervised scheme \n \n')
disp(T)
fprintf('Average predictive negative log-likelihood: %d \n', round(mean(NLLTest)))
fprintf('Event sensitivity: %.4f, event false positive rate: %.4f \n', mean(PerfTestAlt(1,:)), mean(PerfTestAlt(2,:)))