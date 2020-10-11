%% Inference only - For figures only
% Use logsumexp implementation because of artifacts!
% Methodology:
% 7 out of 8 subjects will be the training set
% Inside training set, estimate model parameters with complete data, i.e.
% labels from each expert separately, i.e. two models will be learned
% Predict labels on 1 remaining subject (test set) in the following way:
% - Model trained with labels from expert 1 wrt to labels from expert 1
% - Model trained with labels from expert 1 wrt to labels from expert 2
% - Model trained with labels from expert 2 wrt to labels from expert 1
% - Model trained with labels from expert 2 wrt to labels from expert 2
% This last four cases is a somewhat confussion matrix for experts
% Last case: Predicted labels from model trained with labels from expert 1 
% are eggregated (logical AND) with predicted labels from model trained
% with labels from expert 2. This aggregated labels are then compared to
% a similar aggregated ground truth (this last aggregated ground truth is 
% common in the literature)
% All recording will be downsampled or upsampled to have Fs=50Hz

clearvars
close all
clc

ObsModel = 'Generalizedt';
p = 5;
robustMstepflag = true;
dmaxSec = 30;

Fs = 50;
nSubTotal = 8;
K = 2;
dmax = round(dmaxSec*Fs);
dmin = p + 1;
normflag = 1;
Estep = 'logsumexp';

%% Data structure
ySeq = cell(1, nSubTotal);
labelsGT = cell(1, nSubTotal);
labelsGTExpert1 = cell(1, nSubTotal);
labelsGTExpert2 = cell(1, nSubTotal);
for i = 1:nSubTotal
    load(['Data/Subject' num2str(i) '_Fs' num2str(100) '.mat'])
    ySeq{i} = y;
    labelsGT{i} = labels;
    labelsGTExpert1{i} = labels1;
    if i <= 6
        labelsGTExpert2{i} = labels2;
    end
    
    % Downsampling
    ySeq{i} = downsample(ySeq{i}, 2);
    labelsGT{i} = downsample(labelsGT{i}, 2);
    labelsGTExpert1{i} = downsample(labelsGTExpert1{i}, 2);
    if i <= 6
        labelsGTExpert2{i} = downsample(labelsGTExpert2{i}, 2);
    end
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
HMModelIni.DurationParameters.dmin = dmin;
HMModelIni.DurationParameters.model = 'NonParametric';
if dmax > 0
    HMModelIni.DurationParameters.dmax = dmax;
end

DurationParameters.dmax = dmax;
DurationParameters.dmin = dmin;
DurationParameters.model = 'NonParametric';

trainSet = 1:nSubTotal;
% Expert 1
HMModelExpert1 = HMModelIni;
% Learning - complete data
for k = 1:numel(trainSet)
    TrainingStructure(k).y = ySeq{trainSet(k)};
    TrainingStructure(k).z = labelsGTExpert1{trainSet(k)};
end
HMModelExpert1 = HMMLearningCompleteDataSleepSpindles(TrainingStructure, HMModelExpert1);
clear TrainingStructure yLearnSet y

% Expert 2
HMModelExpert2 = HMModelIni;
% Learning - complete data
trainSet = setdiff(trainSet, [7 8]);            % these subjects do not have labels from expert 2
for k = 1:numel(trainSet)
    TrainingStructure(k).y = ySeq{trainSet(k)};
    TrainingStructure(k).z = labelsGTExpert2{trainSet(k)};
end
HMModelExpert2 = HMMLearningCompleteDataSleepSpindles(TrainingStructure, HMModelExpert2);
clear TrainingStructure yLearnSet y

%% Save results
if robustMstepflag == true
    save(['ICASSP results/Supervised/Experts/' ObsModel '/AR order ' num2str(p)...
        '/Robust_dmax' num2str(dmaxSec) 'sec_ALLSUBJECTS.mat'], 'HMModelExpert1', 'HMModelExpert2')
else
    save(['ICASSP results/Supervised/Experts/' ObsModel '/AR order ' num2str(p)...
        '/NoRobust_dmax' num2str(dmaxSec) 'sec_ALLSUBJECTS.mat'], 'HMModelExpert1', 'HMModelExpert2')
end