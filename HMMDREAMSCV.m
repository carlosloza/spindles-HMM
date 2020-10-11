%% First attempt - semi-supervised approach
% Use logsumexp implementation because of artifacts!
% Methodology:
% 7 out of 8 subjects will be the training set
% Inside training set, choose one subject and labels for initial conditions
% of algorithm using complete data estimations and then train model with 6
% subjects
% Run 7-fold cross-validation, then choose best hyperparameter configuration
% Predict labels on 2 remaining subjects (test set)
% Repeat 
% 
% All recording will be downsampled or upsampled to have Fs=100Hz
% No robust processing at all!

clearvars 
close all
clc

Fs = 100;
p_v = [5 10 15 20];
nSubTotal = 8;
K = 2;

normflag = 1;
Estep = 'logsumexp';

for p_i = 1:length(p_v)
    p = p_v(p_i);
    HMModel.type = 'ARHMM';
    HMModel.StateParameters.K = K;
    HMModel.normalize = normflag;
    HMModel.ObsParameters.model = 'Gaussian';
    HMModel.ARorder = p;
    for subj_i = 1:nSubTotal
        testSet = subj_i;
        trainSet = setdiff(1:8, testSet);
        for i = 1:numel(trainSet)
            valSet = trainSet(i);
            truetrainSet = setdiff(trainSet, valSet);
            for j = 1:numel(truetrainSet)
                % Initial conditions
                CondIniSet = truetrainSet(j);
                NoCondIniSet = setdiff(truetrainSet, CondIniSet);
                % Create complete data structure
                for k = 1:numel(CondIniSet)
                    load(['DREAMS/Subject' num2str(CondIniSet(k)) '_Fs' num2str(Fs) '.mat'])  
                    TrainingStructure(k).y = y;
                    TrainingStructure(k).z = labels;
                end
                HMModel = HMMLearningCompleteData(TrainingStructure, HMModel);
                clear TrainingStructure
                % Create training batch
                ySeq = cell(1, numel(NoCondIniSet));
                for k = 1:numel(NoCondIniSet)
                    load(['DREAMS/Subject' num2str(NoCondIniSet(k)) '_Fs' num2str(Fs) '.mat'])  
                    ySeq{k} = y;
                end
                HMModel = HMMLearning(ySeq, K, 'type', 'ARHMM', 'ARorder', p, 'Estep', Estep, 'normalize', normflag,...
                    'StateParameters', HMModel.StateParameters, 'ObsParameters', HMModel.ObsParameters);
                % Validation set
                for k = 1:numel(valSet)
                    load(['DREAMS/Subject' num2str(valSet(k)) '_Fs' num2str(Fs) '.mat'])  
                    ytest = y;
                    labelsGT = labels;
                    [labelsPred, loglike1] = HMMInference(ytest, HMModel, 'normalize', normflag);
                    [CM, CMevent] = ConfusionMatrixSpindles(labelsGT(p+1:end), labelsPred(p+1:end), Fs);
                end
            end
        end
    end
end



dmax = round(2*Fs);
scorer = 1;
normflag = 1;

subjTrain = 6;
subjTest = 6;


% For visual scorer 1 only!
subj_tlim(1,:) = [50000, 100000];
subj_tlim(2,:) = [25000, 70000];
subj_tlim(3,:) = [88000, 100000];
subj_tlim(4,:) = [1, 25000];
subj_tlim(5,:) = [45000, 100000];
subj_tlim(6,:) = [1, 100000];
subj_tlim(7,:) = [65000, 99000];
subj_tlim(8,:) = [25000, 100000];

% Training structure with labels and same sampling rate
for s_i = 1:numel(subjTrain)
    load(['DREAMS/Subject' num2str(subjTrain(s_i)) '.mat']);
    y = X;
    if Fs > Fs
        ndown = Fs/Fs;
        y = downsample(y, ndown);
    elseif Fs < Fs
        y = resample(y, Fs, Fs);
    end
    if scorer == 1
        v_sc = v_sc1;
    elseif scorer == 2
        v_sc = v_sc2;
    end
    % labels
    labels = ones(size(y));
    for i = 1:size(v_sc, 1)
        labels(round(Fs*v_sc(i, 1)):round(Fs*v_sc(i, 1)) + round(Fs*v_sc(i, 2))) = 2;
    end
    idx = subj_tlim(subjTrain(s_i), :);
    y = y(idx(1) : idx(2));
    labels = labels(idx(1) : idx(2));
    TrainingStructure(s_i).y = y;
    TrainingStructure(s_i).z = labels;
end

%% Test subject
load(['DREAMS/Subject' num2str(subjTest) '.mat']);
ytest = X;
if Fs > Fs
    ndown = Fs/Fs;
    ytest = downsample(ytest, ndown);
elseif Fs < Fs
    ytest = resample(ytest, Fs, Fs);
end
if scorer == 1
    v_sc = v_sc1;
elseif scorer == 2
    v_sc = v_sc2;
end
% labels
labels = ones(size(ytest));
for i = 1:size(v_sc, 1)
    labels(round(Fs*v_sc(i, 1)):round(Fs*v_sc(i, 1)) + round(Fs*v_sc(i, 2))) = 2;
end
idx = subj_tlim(subjTest, :);
ytest = ytest(idx(1) : idx(2));
labels = labels(idx(1) : idx(2));
zGT = labels;


%% ARHMM
HMModel.type = 'ARHMM';
HMModel.StateParameters.K = 2;
HMModel.ARorder = p;
HMModel.normalize = normflag;
HMModel.ObsParameters.model = 'Gaussian';
HMModel = HMMLearningCompleteData(TrainingStructure, HMModel);
loglikeARHMM = HMMLikelihood(ytest, HMModel, 'normalize', normflag, 'method', 'scaling');
[zHMM, loglike1] = HMMInference(ytest, HMModel, 'normalize', normflag);

%% ARHSMMED
HSMModel.type = 'ARHSMMED';
HSMModel.StateParameters.K = 2;
HSMModel.ARorder = p;
HSMModel.normalize = normflag;
HSMModel.ObsParameters.model = 'Gaussian';
HSMModel.DurationParameters.model = 'NonParametric';
HSMModel.DurationParameters.dmax = dmax;
HSMModel = HMMLearningCompleteData(TrainingStructure, HSMModel);
loglikeARHSMMED = HSMMLikelihood(ytest, HSMModel, 'normalize', normflag, 'method', 'scaling');
[zHSMM, ds, loglike2] = HSMMInference(ytest, HSMModel, 'normalize', normflag);

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
%%
CMHSMM = zeros(2, 2);
for i = p+1:numel(zGT)
    if zGT(i) == 1 && zHSMM(i) == 1
        CMHSMM(1, 1) = CMHSMM(1, 1) + 1;
    elseif zGT(i) == 1 && zHSMM(i) == 2
        CMHSMM(1, 2) = CMHSMM(1, 2) + 1;
    elseif zGT(i) == 2 && zHSMM(i) == 1
        CMHSMM(2, 1) = CMHSMM(2, 1) + 1;
    elseif zGT(i) == 2 && zHSMM(i) == 2
        CMHSMM(2, 2) = CMHSMM(2, 2) + 1;
    end
end
fprintf('HSMM - accuracy: %.2f, TPR non-spindles: %.2f, TPR spindles: %.2f, FPR spindles: %.2f, \n', ...
    sum(diag(CMHSMM))/sum(CMHSMM(:)), CMHSMM(1,1)/sum(CMHSMM(1,:)),...
    CMHSMM(2,2)/sum(CMHSMM(2,:)), CMHSMM(1,2)/(sum(CMHSMM(:,2))))