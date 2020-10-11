%% First attempt - Inference only
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
p_v = [5 10 15 20];
%p_v = 0;
nSubTotal = 8;
K = 2;
dmax = 3*Fs;
dmin = round(0.5*Fs);

normflag = 1;
%Estep = 'scaling';

robustMstepflag = 0;

p_opt = zeros(1, nSubTotal);
PerfVal = zeros(3, nSubTotal);
PerfValAlt = zeros(3, nSubTotal);
PerfTest = zeros(3, nSubTotal);
PerfTestAlt = zeros(3, nSubTotal);
NLLTest = zeros(1, nSubTotal);

HMModelIni.type = 'ARHSMMED';
HMModelIni.StateParameters.K = K;
HMModelIni.normalize = normflag;
HMModelIni.robustMstep = robustMstepflag;
HMModelIni.ObsParameters.model = 'Generalizedt';
%HMModelIni.ObsParameters.model = 'Gaussian';
%HMModelIni.ObsParameters.meanFcnType = 'Linear';
HMModelIni.ObsParameters.meanFcnType = 'SVM';
HMModelIni.ARorder = 0;
HMModelIni.DurationParameters.dmax = dmax;
HMModelIni.DurationParameters.dmin = dmin;
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
                load(['DREAMS/Data/Subject' num2str(truetrainSet(k)) '_Fs' num2str(Fs) '.mat'])
                TrainingStructure(k).y = y;
                %TrainingStructure(k).y = [0 diff(y)];
                %TrainingStructure(k).y = emd(y, 'MaxNumIMF', 1, 'Display', 0)';
                TrainingStructure(k).z = labels;
                %TrainingStructure(k).y = y(subj_tlim(truetrainSet(k),1):subj_tlim(truetrainSet(k),2));
                %TrainingStructure(k).z = labels1(subj_tlim(truetrainSet(k),1):subj_tlim(truetrainSet(k),2));
            end
            HMModel = HMMLearningCompleteData(TrainingStructure, HMModel);
            clear TrainingStructure
            clear y
            Perfaux = zeros(6, numel(valSet));
            for k = 1:numel(valSet)
                load(['DREAMS/Data/Subject' num2str(valSet(k)) '_Fs' num2str(Fs) '.mat'])
                ytest = y;
                %ytest = [0 diff(y)];
                %ytest = emd(y, 'MaxNumIMF', 1, 'Display', 0)';
                labelsGT = labels;
                if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'ARHMM')
                    [labelsPred, llike] = HMMInference(ytest, HMModel, 'normalize', normflag);
                else
                    labelsPred = HSMMInference(ytest, HMModel, 'normalize', normflag);
                end
                [CM, CMevent] = ConfusionMatrixSpindles(labelsGT(p+1:end), labelsPred(p+1:end), Fs);
                TPR = CM(1,1)/sum(CM(1,:));     % recall
                RE = TPR;
                FPR = CM(2,1)/sum(CM(2,:));
                PR = CM(1,1)/sum(CM(:,1));
                FPProp = CM(2,1)/sum(CM(1,:));
                F1 = 2*(RE*PR)/(RE + PR);
                MCC = (CM(1,1)*CM(2,2)-CM(2,1)*CM(1,2))/...
                    sqrt((CM(1,1)+CM(1,2))*(CM(1,1)+CM(2,1))*(CM(2,2)+CM(2,1))*(CM(2,2)+CM(1,2)));
                
                TPRalt = CMevent(1,1)/sum(CMevent(1,:));
                FPRalt = CMevent(2,1)/sum(CMevent(2,:));
                FPPropalt = CMevent(2,1)/sum(CMevent(1,:));
                %Perfaux(:, k) = [TPR FPR FPProp TPRalt FPRalt FPPropalt]';
                Perfaux(:, k) = [F1 MCC FPProp TPRalt FPRalt FPPropalt]';
            end
            PerfValaux(:, i, p_i) = mean(Perfaux(1:3,:), 2);
            PerfValAltaux(:, i, p_i) = mean(Perfaux(4:6,:), 2);
            clear HMModel
        end
        asd = 1;
    end
    aux1 = squeeze(mean(PerfValaux, 2));
    aux2 = squeeze(mean(PerfValAltaux, 2));
    [~, idx] = max(aux1(2,:));                  % based on MCC
    p_opt(subj_i) = p_v(idx);
    PerfVal(:, subj_i) = aux1(:, idx);
    PerfValAlt(:, subj_i) = aux2(:, idx);
    % Test
    HMModel = HMModelIni;
    p = p_opt(subj_i);
    HMModel.ARorder = p;
    for k = 1:numel(trainSet)
        load(['DREAMS/Data/Subject' num2str(trainSet(k)) '_Fs' num2str(Fs) '.mat'])
        TrainingStructure(k).y = y;
        %TrainingStructure(k).y = [0 diff(y)];
        %TrainingStructure(k).y = emd(y, 'MaxNumIMF', 1, 'Display', 0)';
        TrainingStructure(k).z = labels;      
    end
    HMModel = HMMLearningCompleteData(TrainingStructure, HMModel);
    clear TrainingStructure
    clear y
    Perfaux = zeros(6, numel(testSet));
    NLLaux = zeros(1, numel(testSet));
    for k = 1:numel(testSet)
        load(['DREAMS/Data/Subject' num2str(testSet(k)) '_Fs' num2str(Fs) '.mat'])
        ytest = y;
        %ytest = [0 diff(y)];
        %ytest = emd(y, 'MaxNumIMF', 1, 'Display', 0)';
        labelsGT = labels;
        if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'ARHMM')
            [labelsPred, NLLaux(k)] = HMMInference(ytest, HMModel, 'normalize', normflag);
        else
            [labelsPred, ~, NLLaux(k)] = HSMMInference(ytest, HMModel, 'normalize', normflag);
        end
        [CM, CMevent] = ConfusionMatrixSpindles(labelsGT(p+1:end), labelsPred(p+1:end), Fs);
        TPR = CM(1,1)/sum(CM(1,:));     % recall
        RE = TPR;
        FPR = CM(2,1)/sum(CM(2,:));
        PR = CM(1,1)/sum(CM(:,1));
        FPProp = CM(2,1)/sum(CM(1,:));
        F1 = 2*(RE*PR)/(RE + PR);
        MCC = (CM(1,1)*CM(2,2)-CM(2,1)*CM(1,2))/...
            sqrt((CM(1,1)+CM(1,2))*(CM(1,1)+CM(2,1))*(CM(2,2)+CM(2,1))*(CM(2,2)+CM(1,2)));
        
        TPRalt = CMevent(1,1)/sum(CMevent(1,:));
        FPRalt = CMevent(2,1)/sum(CMevent(2,:));
        FPPropalt = CMevent(2,1)/sum(CMevent(1,:));
        %Perfaux(:, k) = [TPR FPR FPProp TPRalt FPRalt FPPropalt]';
        Perfaux(:, k) = [F1 MCC FPProp TPRalt FPRalt FPPropalt]';
        
        
%         TPR = CM(1,1)/sum(CM(1,:));
%         FPR = CM(2,1)/sum(CM(2,:));
%         FPProp = CM(2,1)/sum(CM(1,:));
%         TPRalt = CMevent(1,1)/sum(CMevent(1,:));
%         FPRalt = CMevent(2,1)/sum(CMevent(2,:));
%         FPPropalt = CMevent(2,1)/sum(CMevent(1,:));
%         Perfaux(:, k) = [TPR FPR FPProp TPRalt FPRalt FPPropalt]';
    end
    PerfTest(:, subj_i) = mean(Perfaux(1:3,:), 2);
    PerfTestAlt(:, subj_i) = mean(Perfaux(4:6,:), 2);   
    NLLTest(subj_i) = -mean(NLLaux);
end
%%
if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'HSMMED')
    rstring = '';
else
    if robustMstepflag == 0
        rstring = '';
    else
        rstring = '_Robust';
    end
    save(['DREAMS/Results/Raw Univariate/' HMModel.type '_Supervised' rstring...
        '_Mean_' HMModel.ObsParameters.meanFcnType HMModel.ObsParameters.model '.mat'],...
        'p_v', 'p_opt', 'PerfVal', 'PerfValAlt', 'PerfTest', 'PerfTestAlt', 'NLLTest')
end
% save(['DREAMS/Results/Raw Univariate/' HMModel.type '_Supervised' rstring '.mat'],...
%     'p_v', 'p_opt', 'PerfVal', 'PerfValAlt', 'PerfTest', 'PerfTestAlt', 'NLLTest')