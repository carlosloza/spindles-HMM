clearvars
close all
clc
%% Inference only
% Use logsumexp implementation because of artifacts!
% Methodology:
% 7 out of 8 subjects will be the training set
% Inside training set, estimate model parameters with complete data, i.e. labels
% Predict labels on 1 remaining subject (test set)
% 
% All recording will be downsampled or upsampled to have Fs=50Hz
% This function takes so long that it needs to be ran in a
% subject-by-subject basis

Fs = 50;
nSubTotal = 8;
K = 2;
normflag = 1;
%Estep = 'logsumexp';

ObsModel = 'Generalizedt';
dmaxSec = 30;
p = 5;
dmin = p + 1;
dmax = round(dmaxSec*Fs);
robustIni = true;

PerfTest = zeros(3, nSubTotal);
PerfTestAlt = zeros(3, nSubTotal);
NLLTest = zeros(1, nSubTotal);
HMModelSupervised = cell(1, nSubTotal);
%% Data structure
ySeq = cell(1, nSubTotal);
labelsGT = cell(1, nSubTotal);
for i = 1:nSubTotal
    load(['Data/Subject' num2str(i) '_Fs' num2str(100) '.mat'])
    ySeq{i} = y;
    labelsGT{i} = labels;
    
    ySeq{i} = downsample(ySeq{i}, 2);
    labelsGT{i} = downsample(labelsGT{i}, 2);
end
%%
%HMModelIni.type = 'ARHSMMED';
HMModelIni.Fs = Fs;
HMModelIni.StateParameters.K = K;
HMModelIni.normalize = normflag;
HMModelIni.robustIni = robustIni;
HMModelIni.ObsParameters.model = ObsModel;
HMModelIni.ObsParameters.meanModel = 'Linear';
HMModelIni.ARorder = p;
HMModelIni.DurationParameters.dmax = dmax;
HMModelIni.DurationParameters.dmin = dmin;
HMModelIni.DurationParameters.model = 'NonParametric';

for subj_i = 1:nSubTotal
    testSet = subj_i;
    trainSet = setdiff(1:nSubTotal, testSet);
    fprintf('Test Subject %d, p = %d \n', testSet, p)
    HMModel = HMModelIni;
    % Learning - complete data
    TrainingStructure(numel(trainSet)) = struct();
    for k = 1:numel(trainSet)
        TrainingStructure(k).y = ySeq{trainSet(k)};
        TrainingStructure(k).z = labelsGT{trainSet(k)};
    end
    HMModel = HMMLearningCompleteDataSleepSpindles(TrainingStructure, HMModel);      
    ytest = ySeq{testSet};
    labelsPred = HMMInference(ytest, HMModel, 'normalize', normflag);
    [PerfTest(:, subj_i), PerfTestAlt(:, subj_i)] = PerfMeasures(labelsGT{testSet}(p+1:end), labelsPred(p+1:end), Fs);
    % Predictive log-likelihood
    loglike = HMMLikelihood(ytest, HMModel, 'normalize', normflag);
    NLLTest(subj_i) = -loglike;
    HMModelSupervised{subj_i} = HMModel;
    clear TrainingStructure y
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