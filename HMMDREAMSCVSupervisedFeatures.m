%% First attempt - Inference only
% Observations are features from EEG
% Use logsumexp implementation because of artifacts!
% Methodology:
% 7 out of 8 subjects will be the training set
% Inside training set, choose 6 subjects and labels for initial conditions
% of algorithm using complete data estimations
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
p_v = [2 5 10 15 20];
%p_v = 0;
nSubTotal = 8;
K = 2;
%dmax = 2*Fs;
 

%winLen = round(0.2*Fs);
%ovp = round(winLen*0.9);
winLen = round(1*Fs);
ovp = winLen - 1;
%bEdges = [0.1 4 7 14 30 50];
bEdges = [10.5 16];

dmax = round(1.1*(2*Fs/(winLen - ovp)));

normflag = 0;
Estep = 'logsumexp';
robustMstepflag = 0;

p_opt = zeros(1, nSubTotal);
PerfVal = zeros(3, nSubTotal);
PerfValAlt = zeros(3, nSubTotal);
PerfTest = zeros(3, nSubTotal);
PerfTestAlt = zeros(3, nSubTotal);
NLLTest = zeros(1, nSubTotal);

%%
for i = 1:8
    load(['DREAMS/Subject' num2str(i) '_Fs' num2str(Fs) '.mat'])
    [yFeat, zUp] = EEGFeatures(y, labels, Fs, winLen, ovp, bEdges);
    FeatStruct(i).y = yFeat;
    FeatStruct(i).z = zUp;
end
%%
tic
HMModelIni.type = 'ARHSMMED';
HMModelIni.StateParameters.K = K;
HMModelIni.normalize = normflag;
HMModelIni.robustMstep = robustMstepflag;
HMModelIni.ObsParameters.model = 'Gaussian';
HMModelIni.ARorder = 0;
HMModelIni.DurationParameters.dmax = dmax;
HMModelIni.DurationParameters.model = 'NonParametric';
for subj_i = 1:nSubTotal
    testSet = subj_i;
    trainSet = setdiff(1:8, testSet);
    PerfValaux = zeros(3, numel(trainSet), numel(p_v));
    PerfValAltaux = zeros(3, numel(trainSet), numel(p_v));
    for p_i = 1:length(p_v)
        fprintf('Test Subject %d, p = %d \n', testSet, p_v(p_i))
        p = p_v(p_i);
        for i = 1:numel(trainSet)
            HMModel = HMModelIni;
            HMModel.ARorder = p;
            valSet = trainSet(i);
            truetrainSet = setdiff(trainSet, valSet);
            % Learning - complete data
            for k = 1:numel(truetrainSet)
                %load(['DREAMS/Subject' num2str(truetrainSet(k)) '_Fs' num2str(Fs) '.mat'])
                %[yFeat, zUp] = EEGFeatures(y, labels, Fs, winLen, ovp, bEdges);
                %TrainingStructure(k).y = yFeat;
                %TrainingStructure(k).z = zUp;
                TrainingStructure(k).y = FeatStruct(truetrainSet(k)).y;
                TrainingStructure(k).z = FeatStruct(truetrainSet(k)).z;
            end
            HMModel = HMMLearningCompleteData(TrainingStructure, HMModel);
            clear TrainingStructure
            clear y
            Perfaux = zeros(6, numel(valSet));
            for k = 1:numel(valSet)
                load(['DREAMS/Subject' num2str(valSet(k)) '_Fs' num2str(Fs) '.mat'])
                %[yFeat, zUp] = EEGFeatures(y, labels, Fs, winLen, ovp, bEdges);
                %ytest = yFeat;
                labelsGT = labels;
                ytest = FeatStruct(valSet(k)).y;
                if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'ARHMM')
                    labelsPred = HMMInference(ytest, HMModel, 'normalize', normflag);
                else
                    labelsPred = HSMMInference(ytest, HMModel, 'normalize', normflag);
                end
                labelsPred = resampleLabels(labelsPred, y, winLen, ovp);
                [CM, CMevent] = ConfusionMatrixSpindles(labelsGT(p+1:end), labelsPred(p+1:end), Fs);            % Careful here with autoregressive models!!!
                TPR = CM(1,1)/sum(CM(1,:));
                FPR = CM(2,1)/sum(CM(2,:));
                FPProp = CM(2,1)/sum(CM(1,:));
                TPRalt = CMevent(1,1)/sum(CMevent(1,:));
                FPRalt = CMevent(2,1)/sum(CMevent(2,:));
                FPPropalt = CMevent(2,1)/sum(CMevent(1,:));
                Perfaux(:, k) = [TPR FPR FPProp TPRalt FPRalt FPPropalt]';
            end
            PerfValaux(:, i, p_i) = mean(Perfaux(1:3,:), 2);
            PerfValAltaux(:, i, p_i) = mean(Perfaux(4:6,:), 2);
            clear HMModel
        end
        asd = 1;
    end
    aux1 = squeeze(mean(PerfValaux, 2));
    aux2 = squeeze(mean(PerfValAltaux, 2));
    [~, idx] = min(aux1(3,:));                  % based on FPproportion
    p_opt(subj_i) = p_v(idx);
    PerfVal(:, subj_i) = aux1(:, idx);
    PerfValAlt(:, subj_i) = aux2(:, idx);
    % Test
    HMModel = HMModelIni;
    p = p_opt(subj_i);
    HMModel.ARorder = p;
    for k = 1:numel(trainSet)
        %load(['DREAMS/Subject' num2str(trainSet(k)) '_Fs' num2str(Fs) '.mat'])
        %[yFeat, zUp] = EEGFeatures(y, labels, Fs, winLen, ovp, bEdges);
        %TrainingStructure(k).y = yFeat;
        %TrainingStructure(k).z = zUp;
        
        TrainingStructure(k).y = FeatStruct(trainSet(k)).y;
        TrainingStructure(k).z = FeatStruct(trainSet(k)).z;
    end
    HMModel = HMMLearningCompleteData(TrainingStructure, HMModel);
    clear TrainingStructure
    clear y
    Perfaux = zeros(6, numel(testSet));
    NLLaux = zeros(1, numel(testSet));
    for k = 1:numel(testSet)
        load(['DREAMS/Subject' num2str(testSet(k)) '_Fs' num2str(Fs) '.mat'])
        %[yFeat, zUp] = EEGFeatures(y, labels, Fs, winLen, ovp, bEdges);
        %ytest = yFeat;
        labelsGT = labels;
        ytest = FeatStruct(testSet(k)).y;
        
        if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'ARHMM')
            [labelsPred, NLLaux(k)] = HMMInference(ytest, HMModel, 'normalize', normflag);
        else
            [labelsPred, ~, NLLaux(k)] = HSMMInference(ytest, HMModel, 'normalize', normflag);
        end
        labelsPred = resampleLabels(labelsPred, y, winLen, ovp);
        [CM, CMevent] = ConfusionMatrixSpindles(labelsGT(p+1:end), labelsPred(p+1:end), Fs);
        TPR = CM(1,1)/sum(CM(1,:));
        FPR = CM(2,1)/sum(CM(2,:));
        FPProp = CM(2,1)/sum(CM(1,:));
        TPRalt = CMevent(1,1)/sum(CMevent(1,:));
        FPRalt = CMevent(2,1)/sum(CMevent(2,:));
        FPPropalt = CMevent(2,1)/sum(CMevent(1,:));
        Perfaux(:, k) = [TPR FPR FPProp TPRalt FPRalt FPPropalt]';
    end
    PerfTest(:, subj_i) = mean(Perfaux(1:3,:), 2);
    PerfTestAlt(:, subj_i) = mean(Perfaux(4:6,:), 2);   
    NLLTest(subj_i) = -mean(NLLaux);
end
toc
%%
if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'HSMMED')
    rstring = '';
else
    if robustMstepflag == 0
        rstring = '';
    else
        rstring = '_Robust';
    end
end
% save(['DREAMS/Results/MFCC Multivariate/' HMModel.type '_Supervised' rstring '.mat'],...
%     'p_v', 'p_opt', 'PerfVal', 'PerfValAlt', 'PerfTest', 'PerfTestAlt', 'NLLTest')