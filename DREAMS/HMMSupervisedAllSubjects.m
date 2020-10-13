%HMMSupervisedInference(ObsModel, p, robustMstepflag, subTest)
% dmax in seconds
% dmax in seconds
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
Estep = 'logsumexp';

ObsModel = 'Generalizedt';
dmaxSec = 3;
p = 5;
dmin = p + 1;
dmax = round(dmaxSec*Fs);
robustMstepflag = false;

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
HMModelIni.type = 'ARHSMMED';
HMModelIni.Fs = Fs;
HMModelIni.StateParameters.K = K;
HMModelIni.normalize = normflag;
HMModelIni.robustMstep = robustMstepflag;
HMModelIni.ObsParameters.model = ObsModel;
HMModelIni.ObsParameters.meanModel = 'Linear';
HMModelIni.ARorder = p;
HMModelIni.DurationParameters.dmax = dmax;
HMModelIni.DurationParameters.dmin = dmin;
HMModelIni.DurationParameters.model = 'NonParametric';


trainSet = 1:nSubTotal;
HMModel = HMModelIni;
% Learning - complete data
for k = 1:numel(trainSet)
    TrainingStructure(k).y = ySeq{trainSet(k)};
    TrainingStructure(k).z = labelsGT{trainSet(k)};
end
HMModel = HMMLearningCompleteDataSleepSpindles(TrainingStructure, HMModel);

%% Save results
if robustMstepflag == true
    save(['ICASSP results/Supervised/Inference/' ObsModel '/AR order ' num2str(p)...
        '/Robust_dmax' num2str(dmaxSec) 'sec_ALLSUBJECTS.mat'], 'HMModel')
else
    save(['ICASSP results/Supervised/Inference/' ObsModel '/AR order ' num2str(p)...
        '/NoRobust_dmax' num2str(dmaxSec) 'sec_ALLSUBJECTS.mat'], 'HMModel')
end