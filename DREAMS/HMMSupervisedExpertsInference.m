%% Inference only
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

ObsModel = 'Gaussian';
p = 5;
robustMstepflag = true;
dmaxSec = 3;

Fs = 50;
nSubTotal = 8;
K = 2;
%dmax = round(300*Fs);
%dmin = round(0.5*Fs);
dmax = round(dmaxSec*Fs);
dmin = p + 1;
normflag = 1;
Estep = 'logsumexp';


PerfTest = zeros(3, nSubTotal);
%PerfTestAlt = zeros(3, nSubTotal);
NLLTest = zeros(2, nSubTotal);
HMModelSupervised = cell(1, nSubTotal);

PerfTestExpert = cell(2,2);
PerfTestExpert(:) = {zeros(3, nSubTotal)};

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
for subj_i = 1:nSubTotal
    testSet = subj_i;
    trainSet = setdiff(1:nSubTotal, testSet);
    fprintf('Test Subject %d, p = %d \n', testSet, p)    
    % Expert 1
    HMModelExpert1 = HMModelIni;
    % Learning - complete data
    for k = 1:numel(trainSet)
        TrainingStructure(k).y = ySeq{trainSet(k)};
        TrainingStructure(k).z = labelsGTExpert1{trainSet(k)};
    end   
    HMModelExpert1 = HMMLearningCompleteDataSleepSpindles(TrainingStructure, HMModelExpert1);    
%     % Keep training
%     for k = 1:numel(trainSet)
%         yLearnSet{k} = ySeq{trainSet(k)};
%     end
%     % Smooth out duration distribution for duration of sleep spindles
%     DurationParameters.Ini = HMModelExpert1.DurationParameters.Ini;
%     HMModelExpert1 = HMMLearning(yLearnSet, K, 'type', HMModelExpert1.type,...
%         'ARorder', p, 'Estep', Estep, 'normalize', normflag,...
%         'ObsParameters', HMModelExpert1.ObsParameters,...
%         'DurationParameters', DurationParameters,...
%         'SleepSpindles', true,...
%         'Fs', Fs, 'robustMstep', false);   
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
%     % Keep training
%     for k = 1:numel(trainSet)
%         yLearnSet{k} = ySeq{trainSet(k)};
%     end
%     % Smooth out duration distribution for duration of sleep spindles
%     DurationParameters.Ini = HMModelExpert2.DurationParameters.Ini;
%     HMModelExpert2 = HMMLearning(yLearnSet, K, 'type', HMModelExpert2.type,...
%         'ARorder', p, 'Estep', Estep, 'normalize', normflag,...
%         'ObsParameters', HMModelExpert2.ObsParameters,...
%         'DurationParameters', DurationParameters,...
%         'SleepSpindles', true,...
%         'Fs', Fs, 'robustMstep', false);   
    clear TrainingStructure yLearnSet y

     HMModelExperts{1} = HMModelExpert1;
     HMModelExperts{2} = HMModelExpert2;
     clear HMModelExpert1 HMModelExpert2
    
    % Test   
    ytest = ySeq{testSet};
    % 4 Cases
    labelsPredExpSep = zeros(2, numel(labelsGT{testSet}(p+1:end)));
    for i = 1:2
        [labelsPred, ~] = HMMInference(ytest, HMModelExperts{i}, 'normalize', normflag);
        
        if dmax > 0
            % Predictive log-likelihood. Do not run this when dmax i set to
            % MAX because it will take forever and crash and die
            [loglike, ~, ~] = HMMLikelihood(ytest, HMModelExperts{i}, 'method', Estep, 'normalize', normflag);
            NLLTest(i, subj_i) = -loglike;
        end
        
        labelsPred = labelsPred(p+1:end);
        labelsPredExpSep(i, :) = labelsPred;
        for j = 1:2
            if j == 1
                [CM, ~] = ConfusionMatrixSpindles(labelsGTExpert1{testSet}(p+1:end), labelsPred, Fs);
            elseif j == 2
                if testSet <= 6
                    [CM, ~] = ConfusionMatrixSpindles(labelsGTExpert2{testSet}(p+1:end), labelsPred, Fs);
                end
            end
            if ~exist('CM', 'var')
                F1 = nan;
                MCC = nan;
                FPProp = nan;
            else
                TPR = CM(1,1)/sum(CM(1,:));     % recall
                RE = TPR;
                FPR = CM(2,1)/sum(CM(2,:));
                PR = CM(1,1)/sum(CM(:,1));
                FPProp = CM(2,1)/sum(CM(1,:));
                F1 = 2*(RE*PR)/(RE + PR);
                MCC = (CM(1,1)*CM(2,2)-CM(2,1)*CM(1,2))/...
                    sqrt((CM(1,1)+CM(1,2))*(CM(1,1)+CM(2,1))*(CM(2,2)+CM(2,1))*(CM(2,2)+CM(1,2)));
            end
            PerfTestExpert{i, j}(:, subj_i) = [F1 MCC FPProp]';
            clear CM
        end     
    end
    % Aggregated case
    clear labelsPred
    labelsPred = sum(labelsPredExpSep, 1)/2;
    labelsPred(labelsPred > 1) = 2;
    [CM, ~] = ConfusionMatrixSpindles(labelsGT{testSet}(p+1:end), labelsPred, Fs);
    TPR = CM(1,1)/sum(CM(1,:));     % recall
    RE = TPR;
    FPR = CM(2,1)/sum(CM(2,:));
    PR = CM(1,1)/sum(CM(:,1));
    FPProp = CM(2,1)/sum(CM(1,:));
    F1 = 2*(RE*PR)/(RE + PR);
    MCC = (CM(1,1)*CM(2,2)-CM(2,1)*CM(1,2))/...
        sqrt((CM(1,1)+CM(1,2))*(CM(1,1)+CM(2,1))*(CM(2,2)+CM(2,1))*(CM(2,2)+CM(1,2)));
    
%     TPRalt = CMevent(1,1)/sum(CMevent(1,:));
%     FPRalt = CMevent(2,1)/sum(CMevent(2,:));
%     FPPropalt = CMevent(2,1)/sum(CMevent(1,:));
    PerfTest(:, subj_i) = [F1 MCC FPProp]';
%    PerfTestAlt(:, subj_i) = [TPRalt FPRalt FPPropalt]';
%    NLLTest(subj_i) = -loglike;
    HMModelSupervised{subj_i} = HMModelExperts;
end
%% Save results - TODO
if robustMstepflag == true
    save(['ICASSP results/Supervised/Experts/' ObsModel '/AR order ' num2str(p)...
        '/Robust_dmax' num2str(dmaxSec) 'sec.mat'], 'PerfTestExpert', 'NLLTest', 'HMModelSupervised', 'PerfTest')
else
    save(['ICASSP results/Supervised/Experts/' ObsModel '/AR order ' num2str(p)...
        '/NoRobust_dmax' num2str(dmaxSec) 'sec.mat'], 'PerfTestExpert', 'NLLTest', 'HMModelSupervised', 'PerfTest')
end