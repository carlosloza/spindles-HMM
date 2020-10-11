%% Learning on unlabeled training data with initial EM condition from labeled data 
% and test (inference) on held out data
% It supports either raw or pre-processed EEG (dual Basis Pursuit Denoising)
% Use logsumexp implementation because of artifacts! - maybe (we'll see...)
% Methodology:
% 7 out of 8 subjects will be the training set
% Inside training set, choose a set of complete data recordings, i.e.
% labeled data to initialize the model 
% Train with remaining unlabeled recordings
% Run inference on test data (1 subject). Compare to ground truth(s):
% Expert 1, expert 2, and union of both. Report F1 scores and MCCs 
% Also report predictive log likelihood
% Cycle through entire dataset
% 
% All recording will be downsampled or upsampled to have Fs = 100Hz
% AR order fixed (p = 5 for Fs = 100 Hz), i.e. no cross-validation of AR
% order

clearvars 
close all
clc

%Fs = 100/2;
Fs = 50;
p = 5;
nSubTotal = 8;
K = 2;

%dmax = round(3.0*Fs);
dmax = round(370*Fs);
dmax = round(126*Fs);            % 60 seems to work, 30 maybe, 126 or 100 ideally
dmax = round(3*Fs);
%dmin = round(0.5*Fs);
dmin = p+1;

HMMtype = 'ARHSMMED';
normflag = 1;
Estep = 'logsumexp';

robustMstepflag = false;
raw = 0;                        % raw = 0 goes to pre-processed data

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
for i = 1:nSubTotal
    load(['Data/Subject' num2str(i) '_Fs' num2str(Fs) '.mat'])
    ySeq{i} = y;
    labelsGT{i} = labels;
end

%%
HMModelIni.type = HMMtype;
HMModelIni.Fs = Fs;
HMModelIni.StateParameters.K = K;
HMModelIni.normalize = normflag;
HMModelIni.robustMstep = robustMstepflag;
%HMModelIni.ObsParameters.model = 'Generalizedt';
HMModelIni.ObsParameters.model = 'Gaussian';
HMModelIni.ObsParameters.meanModel = 'Linear';
HMModelIni.ARorder = p;
HMModelIni.DurationParameters.dmax = dmax;
HMModelIni.DurationParameters.dmin = dmin;
HMModelIni.DurationParameters.model = 'NonParametric';

DurationParametersT.dmax = dmax;
DurationParametersT.dmin = dmin;
DurationParametersT.model = 'NonParametric';

figure
for subj_i = 1:nSubTotal
    testSet = subj_i;
    trainSet = setdiff(1:nSubTotal, testSet);
    PredLogLikeVal = zeros(length(p_v), numel(trainSet));
    PredRelSigmaPowVal = zeros(length(p_v), numel(trainSet));
    %for p_i = 1:length(p_v)
        fprintf('Test Subject %d, p = %d \n', testSet, p)
        for i = 1:numel(trainSet)
            
            HMModel = HMModelIni;
        
            
            valSet = 1;
            
            IniSet = trainSet(i);
            truetrainSet = setdiff(trainSet, IniSet);
            % Learning - complete data
            for k = 1:numel(IniSet)               
                TrainingStructure(k).y = ySeq{IniSet(k)};
                TrainingStructure(k).z = labelsGT{IniSet(k)};
            end
            
%             for k = 1:8               
%                 TrainingStructure(k).y = ySeq{k};
%                 TrainingStructure(k).z = labelsGT{k};
%             end
            
            HMModel = HMMLearningCompleteDataSleepSpindles(TrainingStructure, HMModel);
            
            HMModel_true = HMModel;
 %%           
%             loglike1 = HMMLikelihood(ySeq{IniSet(k)}, HMModel, 'method', 'logsumexp', 'normalize', normflag)
%             
%             HMModel = HMMLearning(ySeq{IniSet(k)}, K, 'type', HMMtype,...
%                 'ARorder', p, 'Estep', Estep, 'normalize', normflag,...
%                 'ObsParameters', HMModel.ObsParameters,...
%                 'DurationParameters', DurationParametersT,...
%                 'robustMstep', robustMstepflag, 'Fs', Fs, 'SleepSpindles', true);
            
 %%          
            % Keep training
            yLearnSet = cell(1, numel(truetrainSet));
            for k = 1:numel(truetrainSet)
                yLearnSet{k} = ySeq{truetrainSet(k)};
            end
            % First change the variance/scale of previous model
            sigmaIniaux = 10;
            nSeq = numel(truetrainSet);
            Ypini = zeros(sigmaIniaux*p*nSeq, p);
            Yini = zeros(sigmaIniaux*p*nSeq, 1);
            %idxini = 1;
            for iSeq = 1:nSeq
                y = yLearnSet{iSeq};
                Ypaux = HMModel.DelayARMatrix(iSeq).Seq;
                %N = HMModel.N(iSeq);
                %idxend = idxini + N - p - 1;
                %Yp(idxini:idxend, :) = Ypaux;
                %idxini = idxend + 1;
                Ypini((iSeq - 1)*sigmaIniaux*p + 1: iSeq*sigmaIniaux*p, :) = ...
                    Ypaux(1:sigmaIniaux*p, :);
                Yini((iSeq - 1)*sigmaIniaux*p + 1: iSeq*sigmaIniaux*p) = ...
                    y(p+1:(sigmaIniaux + 1)*p)';
            end
            
            HMModel = HMMLearning(yLearnSet, K, 'type', HMMtype,...
                'ARorder', p, 'Estep', Estep, 'normalize', normflag,...
                'ObsParameters', HMModel.ObsParameters,...
                'DurationParameters', DurationParametersT,...
                'robustMstep', robustMstepflag, 'Fs', Fs, 'SleepSpindles', true);
            clear y
            PredLogLikeVal(p_i, i) = HMMLikelihood(ySeq{valSet}, HMModel, 'method', Estep, 'normalize', normflag);          % valSet is always one time series so this is OK
            labelsPredVal = HMMInference(ySeq{valSet}, HMModel, 'normalize', normflag);
            PredRelSigmaPowVal(p_i, i) = RelativeSigmaPower(ySeq{valSet}(p+1:end), labelsPredVal(p+1:end), Fs);
            clear HMModel
        end
    %end
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