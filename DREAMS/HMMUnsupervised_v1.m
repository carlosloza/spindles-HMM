%% Learning on unlabeled training data and test (inference) on held out data
% It supports either raw or pre-processed EEG (dual Basis Pursuit Denoising)
% Use logsumexp implementation because of artifacts! - maybe (we'll see...)
% Methodology:
% 7 out of 8 subjects will be the training set
% Inside training set, cross-validate AR order (leave one out) - 
% choose optimal based on maximum predictive log likelihood
% Train on entire training set with optimal AR order
% Run inference on test data (1 subject). Compare to ground truth(s):
% Expert 1, expert 2, and union of both. Report F1 scores and MCCs 
% Also report predictive log likelihood
% Cycle through entire dataset
% 
% All recording will be downsampled or upsampled to have Fs=100Hz

clearvars 
close all
clc

Fs = 100/2;
%p_v = 3:1:10;
p_v = 4;
nSubTotal = 8;
K = 2;

dmax = round(2.5*Fs);
%dmax = round(1000*Fs);
dmin = 1;

HMMtype = 'ARHSMMVT';
normflag = 1;
Estep = 'logsumexp';


robustMstepflag = false;
raw = 1;                        % raw = 0 goes to pre-processed data

OptPredLogLikeVal = zeros(1, nSubTotal);
PredLogLikeTest = zeros(1, nSubTotal);
PerfTest = zeros(2, nSubTotal);         % F1 and MCC
ARpOpt = zeros(1, nSubTotal);
labelsPredSeq = cell(1, nSubTotal);

% PerfVal = zeros(3, nSubTotal);
% PerfValAlt = zeros(3, nSubTotal);
% PerfTest = zeros(3, nSubTotal);
% PerfTestAlt = zeros(3, nSubTotal);
% NLLTest = zeros(1, nSubTotal);

%% Data structure
ySeq = cell(1, nSubTotal);
labelsGT = cell(1, nSubTotal);
if raw == 1
    for i = 1:nSubTotal
        %load(['Data/Subject' num2str(i) '_Fs' num2str(Fs) '.mat'])
        load(['Data/Subject' num2str(i) '_Fs' num2str(100) '.mat'])
        ySeq{i} = downsample(y, 2);
        labelsGT{i} = downsample(labels, 2);
    end
else
    for i = 1:nSubTotal
        %load(['Data/Subject' num2str(i) '_Fs' num2str(Fs) '_Transformed.mat'])
        load(['Data/Subject' num2str(i) '_Fs' num2str(100) '_Transformed.mat'])
        ySeq{i} = downsample(y, 2);
        labelsGT{i} = downsample(labels, 2);
    end
end

%%
ObsParameters.model = 'Generalizedt';
%ObsParameters.model = 'Gaussian';
ObsParameters.meanModel = 'Linear';
%HMModelIni.ARorder = 0;
DurationParameters.dmax = dmax;
DurationParameters.dmin = dmin;
DurationParameters.model = 'NonParametric';
%% HSMMVT parameters
DurationParameters.DurationIntervals = [2 50 100 150 200 250]/2;
StateParameters.K = 2;
StateParameters.pi = [1 0]';
StateParameters.A(:,:,1) = [0.50 0.50; 0 0];
StateParameters.A(:,:,2) = [0.50 0.50; 1 0];
StateParameters.A(:,:,3) = [0.50 0.50; 1 0];
StateParameters.A(:,:,4) = [0.50 0.50; 1 0];
StateParameters.A(:,:,5) = [0.50 0.50; 1 0];
%%
for subj_i = 1:nSubTotal
    testSet = subj_i;
    trainSet = setdiff(1:nSubTotal, testSet);
%     PerfValaux = zeros(3, numel(trainSet), numel(p_v));
%     PerfValAltaux = zeros(3, numel(trainSet), numel(p_v));   
    PredLogLikeVal = zeros(length(p_v), numel(trainSet));
    PredRelSigmaPowVal = zeros(length(p_v), numel(trainSet));
    for p_i = 1:length(p_v)
        fprintf('Test Subject %d, p = %d \n', testSet, p_v(p_i))
        p = p_v(p_i);
        for i = 1:numel(trainSet)
            valSet = trainSet(i);
            truetrainSet = setdiff(trainSet, valSet);
            % Learning - complete data
            yLearnSet = cell(1, numel(truetrainSet));
            for k = 1:numel(truetrainSet)
                yLearnSet{k} = ySeq{truetrainSet(k)};
            end
            
            HMModel = HMMLearning(yLearnSet, K, 'type', HMMtype,...
                'ARorder', p, 'Estep', Estep, 'normalize', normflag,...
                'ObsParameters', ObsParameters, 'DurationParameters', DurationParameters,...
                'StateParameters', StateParameters, 'SleepSpindles', true,...
                'Fs', Fs, 'robustMstep', robustMstepflag);
            
%             HMModel = HMMLearning(yLearnSet, K, 'type', HMMtype,...
%                 'ARorder', p, 'Estep', Estep, 'normalize', normflag,...
%                 'ObsParameters', ObsParameters, 'DurationParameters', DurationParameters,...
%                 'SleepSpindles', true,...
%                 'Fs', Fs, 'robustMstep', robustMstepflag);
            
            clear y
            PredLogLikeVal(p_i, i) = HMMLikelihood(ySeq{valSet}, HMModel, 'method', Estep, 'normalize', normflag);          % valSet is always one time series so this is OK
            labelsPredVal = HMMInference(ySeq{valSet}, HMModel, 'normalize', normflag);
            PredRelSigmaPowVal(p_i, i) = RelativeSigmaPower(ySeq{valSet}(p+1:end), labelsPredVal(p+1:end), Fs);
            clear HMModel
        end
    end
    [OptPredLogLikeVal(subj_i), idx] = max(mean(PredLogLikeVal, 2));
    ARpOpt(subj_i) = p_v(idx);
    clear HHModel
    % Test
    % Retrain model with all training samples
    yLearnSet = cell(1, numel(trainSet));
    for i = 1:numel(trainSet)
        yLearnSet{i} = ySeq{trainSet(i)};
    end
    HMModel = HMMLearning(yLearnSet, K, 'type', HMMtype,...
                'ARorder', ARpOpt(subj_i), 'Estep', Estep, 'normalize', normflag,...
                'ObsParameters', ObsParameters, 'SleepSpindles', true, 'Fs', Fs,...
                'robustMstep', robustMstepflag);
    % Test set
    PredLogLikeTest(subj_i) = HMMLikelihood(ySeq{testSet}, HMModel, 'method', Estep, 'normalize', normflag); 
    labelsPred = HMMInference(ySeq{testSet}, HMModel, 'normalize', normflag);
    labelsPredSeq{subj_i} = labelsPred;
    [CM, CMevent] = ConfusionMatrixSpindles(labelsGT{testSet}(ARpOpt(subj_i) + 1:end),...
        labelsPred(ARpOpt(subj_i) + 1:end), Fs);
    TPR = CM(1,1)/sum(CM(1,:));     % recall
    RE = TPR;
    FPR = CM(2,1)/sum(CM(2,:));
    PR = CM(1,1)/sum(CM(:,1));
    FPProp = CM(2,1)/sum(CM(1,:));
    F1 = 2*(RE*PR)/(RE + PR);
    if isnan(F1)
        F1 = 0;
    end
    MCC = (CM(1,1)*CM(2,2)-CM(2,1)*CM(1,2))/...
        sqrt((CM(1,1)+CM(1,2))*(CM(1,1)+CM(2,1))*(CM(2,2)+CM(2,1))*(CM(2,2)+CM(1,2)));
    PerfTest(1, subj_i) = F1;
    PerfTest(2, subj_i) = MCC;
    
%     asd = 1;
%     aux1 = squeeze(mean(PerfValaux, 2));
%     aux2 = squeeze(mean(PerfValAltaux, 2));
%     [~, idx] = max(aux1(2,:));                  % based on MCC
%     p_opt(subj_i) = p_v(idx);
%     PerfVal(:, subj_i) = aux1(:, idx);
%     PerfValAlt(:, subj_i) = aux2(:, idx);
%     % Test
%     HMModel = HMModelIni;
%     p = p_opt(subj_i);
%     HMModel.ARorder = p;
%     for k = 1:numel(trainSet)
%         load(['DREAMS/Subject' num2str(trainSet(k)) '_Fs' num2str(Fs) '_Transformed.mat'])
%         TrainingStructure(k).y = y_osc;
%         %TrainingStructure(k).y = [0 diff(y)];
%         %TrainingStructure(k).y = emd(y, 'MaxNumIMF', 1, 'Display', 0)';
%         TrainingStructure(k).z = labels;      
%     end
%     HMModel = HMMLearningCompleteData(TrainingStructure, HMModel);
%     clear TrainingStructure
%     clear y
%     Perfaux = zeros(6, numel(testSet));
%     NLLaux = zeros(1, numel(testSet));
%     for k = 1:numel(testSet)
%         load(['DREAMS/Subject' num2str(testSet(k)) '_Fs' num2str(Fs) '_Transformed.mat'])
%         ytest = y_osc;
%         %ytest = [0 diff(y)];
%         %ytest = emd(y, 'MaxNumIMF', 1, 'Display', 0)';
%         labelsGT = labels;
%         if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'ARHMM')
%             [labelsPred, NLLaux(k)] = HMMInference(ytest, HMModel, 'normalize', normflag);
%         else
%             [labelsPred, ~, NLLaux(k)] = HSMMInference(ytest, HMModel, 'normalize', normflag);
%         end
%         [CM, CMevent] = ConfusionMatrixSpindles(labelsGT(p+1:end), labelsPred(p+1:end), Fs);
%         TPR = CM(1,1)/sum(CM(1,:));     % recall
%         RE = TPR;
%         FPR = CM(2,1)/sum(CM(2,:));
%         PR = CM(1,1)/sum(CM(:,1));
%         FPProp = CM(2,1)/sum(CM(1,:));
%         F1 = 2*(RE*PR)/(RE + PR);
%         MCC = (CM(1,1)*CM(2,2)-CM(2,1)*CM(1,2))/...
%             sqrt((CM(1,1)+CM(1,2))*(CM(1,1)+CM(2,1))*(CM(2,2)+CM(2,1))*(CM(2,2)+CM(1,2)));
%         
%         TPRalt = CMevent(1,1)/sum(CMevent(1,:));
%         FPRalt = CMevent(2,1)/sum(CMevent(2,:));
%         FPPropalt = CMevent(2,1)/sum(CMevent(1,:));
%         %Perfaux(:, k) = [TPR FPR FPProp TPRalt FPRalt FPPropalt]';
%         Perfaux(:, k) = [F1 MCC FPProp TPRalt FPRalt FPPropalt]';
%     end
    
%     PerfTest(:, subj_i) = mean(Perfaux(1:3,:), 2);
%     PerfTestAlt(:, subj_i) = mean(Perfaux(4:6,:), 2);   
%     NLLTest(subj_i) = -mean(NLLaux);
end
%%
if robustMstepflag == 0
    rstring = '';
else
    rstring = '_Robust';
end
if raw == 1
    auxstring = 'Raw Univariate';
else
end
save(['Results/' auxstring '/Unsupervised/' HMModel.type rstring...
     '_' HMModel.ObsParameters.model '.mat'],...
    'p_v', 'Fs', 'OptPredLogLikeVal', 'PredLogLikeTest', 'PerfTest',...
    'ARpOpt', 'labelsPredSeq')