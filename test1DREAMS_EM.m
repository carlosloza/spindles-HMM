%% 
% Take a subject as the initial conditions for EM, then learn model using 
% other (unseen) subjects. Lastly, test model (inference) on another unseen
% subject

clearvars 
close all
clc

FsAll = 100;

p = 10;
K = 2;
dmax = round(2*FsAll);
normflag = 1;
Estep = 'logsumexp';

subjIni = 6;
subjTrain = 6;
subjTest = 6;

% % For visual scorer 1 only! - no more!
% subj_tlim(1,:) = [50000, 100000];
% subj_tlim(2,:) = [25000, 70000];
% subj_tlim(3,:) = [88000, 100000];
% subj_tlim(4,:) = [1, 25000];
% subj_tlim(5,:) = [45000, 100000];
% subj_tlim(6,:) = [1, 100000];
% subj_tlim(7,:) = [65000, 99000];
% subj_tlim(8,:) = [25000, 100000];

%%  Initial conditions
% Training structure with labels and same sampling rate

load(['DREAMS/Subject' num2str(subjIni) '.mat']);
y = X;
if Fs > FsAll
    ndown = Fs/FsAll;
    y = downsample(y, ndown);
elseif Fs < FsAll
    y = resample(y, FsAll, Fs);
end
aux = zeros(2, numel(y));       % 2 scorers
for i_sc = 1:2
    if i_sc == 1
        v_sc = v_sc1;
    elseif i_sc == 2
        v_sc = v_sc2;
    end
    for i = 1:size(v_sc, 1)
        aux(i_sc, round(FsAll*v_sc(i, 1)):round(FsAll*v_sc(i, 1)) + round(FsAll*v_sc(i, 2))) = 1;
    end
end
labels = 1 + sum(aux, 1);
labels(labels > 1) = 2;
CompleteDataStructure.y = y;
CompleteDataStructure.z = labels;

%% ARHMM
HMModelComplete.type = 'ARHMM';
HMModelComplete.StateParameters.K = K;
HMModelComplete.ARorder = p;
HMModelComplete.normalize = normflag;
HMModelComplete.ObsParameters.model = 'Gaussian';

%HMModelComplete.DurationParameters.model = 'NonParametric';
%HMModelComplete.DurationParameters.dmax = dmax;

HMModelComplete = HMMLearningCompleteData(CompleteDataStructure, HMModelComplete);

% Learning
ySeq = cell(1, numel(subjTrain));
for s_i = 1:numel(subjTrain)
    load(['DREAMS/Subject' num2str(subjTrain(s_i)) '.mat']);
    y = X;
    if Fs > FsAll
        ndown = Fs/FsAll;
        y = downsample(y, ndown);
    elseif Fs < FsAll
        y = resample(y, FsAll, Fs);
    end
    ySeq{s_i} = y;
end

HMModel = HMMLearning(ySeq, K, 'type', 'ARHMM', 'Estep', Estep, 'normalize', normflag,...
    'ARorder', p, 'StateParameters', HMModelComplete.StateParameters, 'ObsParameters', HMModelComplete.ObsParameters);

clear y
% Inference
load(['DREAMS/Subject' num2str(subjTest) '.mat']);
ytest = X;
if Fs > FsAll
    ndown = Fs/FsAll;
    ytest = downsample(ytest, ndown);
elseif Fs < FsAll
    ytest = resample(ytest, FsAll, Fs);
end
aux = zeros(2, numel(ytest));       % 2 scorers
for i_sc = 1:2
    if i_sc == 1
        v_sc = v_sc1;
    elseif i_sc == 2
        v_sc = v_sc2;
    end
    for i = 1:size(v_sc, 1)
        aux(i_sc, round(FsAll*v_sc(i, 1)):round(FsAll*v_sc(i, 1)) + round(FsAll*v_sc(i, 2))) = 1;
    end
end
zGT = 1 + sum(aux, 1);
zGT(zGT > 1) = 2;
loglikeARHMM = HMMLikelihood(ytest, HMModelComplete, 'normalize', normflag, 'method', 'logsumexp');
[zHMM, loglike1] = HMMInference(ytest, HMModelComplete, 'normalize', normflag);

% %% Test subject
% load(['DREAMS/Subject' num2str(subjTest) '.mat']);
% ytest = X;
% if Fs > FsAll
%     ndown = Fs/FsAll;
%     ytest = downsample(ytest, ndown);
% elseif Fs < FsAll
%     ytest = resample(ytest, FsAll, Fs);
% end
% if scorer == 1
%     v_sc = v_sc1;
% elseif scorer == 2
%     v_sc = v_sc2;
% end
% % labels
% labels = ones(size(ytest));
% for i = 1:size(v_sc, 1)
%     labels(round(FsAll*v_sc(i, 1)):round(FsAll*v_sc(i, 1)) + round(FsAll*v_sc(i, 2))) = 2;
% end
% idx = subj_tlim(subjTest, :);
% ytest = ytest(idx(1) : idx(2));
% labels = labels(idx(1) : idx(2));
% zGT = labels;
% 
% 
% %% ARHMM
% HMModel.type = 'ARHMM';
% HMModel.StateParameters.K = 2;
% HMModel.ARorder = p;
% HMModel.normalize = normflag;
% HMModel.ObsParameters.model = 'Gaussian';
% HMModel = HMMLearningCompleteData(TrainingStructure, HMModel);
% loglikeARHMM = HMMLikelihood(ytest, HMModel, 'normalize', normflag, 'method', 'scaling');
% [zHMM, loglike1] = HMMInference(ytest, HMModel, 'normalize', normflag);
% 
% %% ARHSMMED
% HSMModel.type = 'ARHSMMED';
% HSMModel.StateParameters.K = 2;
% HSMModel.ARorder = p;
% HSMModel.normalize = normflag;
% HSMModel.ObsParameters.model = 'Gaussian';
% HSMModel.DurationParameters.model = 'NonParametric';
% HSMModel.DurationParameters.dmax = dmax;
% HSMModel = HMMLearningCompleteData(TrainingStructure, HSMModel);
% loglikeARHSMMED = HSMMLikelihood(ytest, HSMModel, 'normalize', normflag, 'method', 'scaling');
% [zHSMM, ds, loglike2] = HSMMInference(ytest, HSMModel, 'normalize', normflag);
% 
%% Compute accuracy, confusion matrix, TPR and so on
CMHMM = zeros(2, 2);
for i = p+1:numel(zGT)
    if zGT(i) == 1 && zHMM(i) == 1
        CMHMM(1, 1) = CMHMM(1, 1) + 1;
    elseif zGT(i) == 1 && zHMM(i) == 2
        CMHMM(1, 2) = CMHMM(1, 2) + 1;
    elseif zGT(i) == 2 && zHMM(i) == 1
        CMHMM(2, 1) = CMHMM(2, 1) + 1;
    elseif zGT(i) == 2 && zHMM(i) == 2
        CMHMM(2, 2) = CMHMM(2, 2) + 1;
    end
end
fprintf('HMM - accuracy: %.2f, TPR non-spindles: %.2f, TPR spindles: %.2f, FPR spindles: %.2f, \n', ...
    sum(diag(CMHMM))/sum(CMHMM(:)), CMHMM(1,1)/sum(CMHMM(1,:)),...
    CMHMM(2,2)/sum(CMHMM(2,:)), CMHMM(1,2)/(sum(CMHMM(:,2))))
% %%
% CMHSMM = zeros(2, 2);
% for i = p+1:numel(zGT)
%     if zGT(i) == 1 && zHSMM(i) == 1
%         CMHSMM(1, 1) = CMHSMM(1, 1) + 1;
%     elseif zGT(i) == 1 && zHSMM(i) == 2
%         CMHSMM(1, 2) = CMHSMM(1, 2) + 1;
%     elseif zGT(i) == 2 && zHSMM(i) == 1
%         CMHSMM(2, 1) = CMHSMM(2, 1) + 1;
%     elseif zGT(i) == 2 && zHSMM(i) == 2
%         CMHSMM(2, 2) = CMHSMM(2, 2) + 1;
%     end
% end
% fprintf('HSMM - accuracy: %.2f, TPR non-spindles: %.2f, TPR spindles: %.2f, FPR spindles: %.2f, \n', ...
%     sum(diag(CMHSMM))/sum(CMHSMM(:)), CMHSMM(1,1)/sum(CMHSMM(1,:)),...
%     CMHSMM(2,2)/sum(CMHSMM(2,:)), CMHSMM(1,2)/(sum(CMHSMM(:,2))))