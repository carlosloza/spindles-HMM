%% 
% Extract Spindles epochs (marked by experts), inspect regressors (i.e. past 
% p samples) and estimate AR coefficients by OLS
clearvars 
close all
clc

p = 10;
subj = 1;
load(['DREAMS/Subject' num2str(subj) '.mat']);
y = X;

scorer = 1;
if scorer == 1
    v_sc = v_sc1;
elseif scorer == 2
     v_sc = v_sc2;
end


% ndown = 1;
% y = downsample(y, ndown);
% Fs = Fs/ndown;

load(['DREAMS/Filters/Spindles_' num2str(Fs) 'Hz'])
%y = filtfilt(h_b, h_a, y);

%y = zscore(y);

labels = ones(size(y));
for i = 1:size(v_sc, 1)
    labels(round(Fs*v_sc(i, 1)):round(Fs*v_sc(i, 1)) + round(Fs*v_sc(i, 2))) = 2;
end

%%
y = y(50000:100000);
labels = labels(50000:100000);
zGT = labels;

dmax = round(1.7*Fs);
%dmax = round(1.1*Fs);

normflag = 1;
%HMModel.DurationParameters.model = 'NonParametric';

%% ARHMM
HMModel.type = 'ARHMM';
HMModel.ARorder = p;
HMModel.normalize = normflag;
HMModel.ObsParameters.model = 'Gaussian';
HMModel = HMMLearningCompleteData(y, labels, HMModel);
loglikeARHMM = HMMLikelihood(y, HMModel, 'normalize', normflag, 'method', 'scaling');
[zHMM, loglike1] = HMMInference(y, HMModel, 'normalize', normflag);

%% ARHSMMED
HSMModel.type = 'ARHSMMED';
HSMModel.ARorder = p;
HSMModel.normalize = normflag;
HSMModel.ObsParameters.model = 'Gaussian';
HSMModel.DurationParameters.model = 'NonParametric';
HSMModel.DurationParameters.dmax = dmax;
HSMModel = HMMLearningCompleteData(y, labels, HSMModel);
loglikeARHSMMED = HSMMLikelihood(y, HSMModel, 'normalize', normflag, 'method', 'scaling');
[zHSMM, ds, loglike2] = HSMMInference(y, HSMModel, 'normalize', normflag);

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