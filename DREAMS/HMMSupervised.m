function HMMSupervised(ObsModel, p, robustMstepflag, dmaxSec)
% dmax in seconds

%% Inference and learning
% Use logsumexp implementation because of artifacts!
% Methodology:
% 7 out of 8 subjects will be the training set
% Inside training set, estimate model parameters with complete data, i.e. labels
% Run EM with estimated parameters as initial conditions (smooth out the 
% duration distribution for spindles)
% Predict labels on 1 remaining subject (test set)
% 
% All recording will be downsampled or upsampled to have Fs=50Hz

Fs = 50;
nSubTotal = 8;
K = 2;
%dmax = round(300*Fs);
%dmin = round(0.5*Fs);
dmax = round(dmaxSec*Fs);
dmin = p + 1;
normflag = 1;
Estep = 'logsumexp';

%robustMstepflag = false;
%raw = 0;

%p_opt = zeros(1, nSubTotal);
%PerfVal = zeros(3, nSubTotal);
%PerfValAlt = zeros(3, nSubTotal);

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
%HMModelIni.ObsParameters.model = 'Generalizedt';
%HMModelIni.ObsParameters.model = 'Gaussian';
HMModelIni.ObsParameters.meanModel = 'Linear';
HMModelIni.ARorder = p;
%HMModelIni.DurationParameters.dmax = dmax;
HMModelIni.DurationParameters.dmin = dmin;
HMModelIni.DurationParameters.model = 'NonParametric';
DurationParameters.dmax = dmax;
DurationParameters.dmin = dmin;
DurationParameters.model = 'NonParametric';
for subj_i = 1:nSubTotal
    testSet = subj_i;
    trainSet = setdiff(1:nSubTotal, testSet);
    fprintf('Test Subject %d, p = %d \n', testSet, p)    
    HMModel = HMModelIni;
    % Learning - complete data
    for k = 1:numel(trainSet)
        TrainingStructure(k).y = ySeq{trainSet(k)};
        TrainingStructure(k).z = labelsGT{trainSet(k)};
    end
    HMModel = HMMLearningCompleteDataSleepSpindles(TrainingStructure, HMModel);
    
    %         [HH, FF] = freqz(1, [1; HMModel.ObsParameters.meanParameters(1).Coefficients], 1024, Fs);
    %         plot(FF, 20*log10(abs(HH)))
    %         hold on
    %         [HH, FF] = freqz(1, [1; HMModel.ObsParameters.meanParameters(2).Coefficients], 1024, Fs);
    %         plot(FF, 20*log10(abs(HH)), 'r')
    %         pause(0.1)
    
    % Keep training
    for k = 1:numel(trainSet)
        yLearnSet{k} = ySeq{trainSet(k)};
    end
    % Smooth out duration distribution for duration of sleep spindles
    DurationParameters.Ini = HMModel.DurationParameters.Ini;
    HMModel = HMMLearning(yLearnSet, K, 'type', HMModel.type,...
        'ARorder', p, 'Estep', Estep, 'normalize', normflag,...
        'ObsParameters', HMModel.ObsParameters,...
        'DurationParameters', DurationParameters,...
        'SleepSpindles', true,...
        'Fs', Fs, 'robustMstep', false);
    
    clear TrainingStructure yLearnSet y
    
    ytest = ySeq{testSet};
    [labelsPred, ~] = HMMInference(ytest, HMModel, 'normalize', normflag);   
    [CM, CMevent] = ConfusionMatrixSpindles(labelsGT{testSet}(p+1:end), labelsPred(p+1:end), Fs);
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
    PerfTest(:, subj_i) = [F1 MCC FPProp]';
    PerfTestAlt(:, subj_i) = [TPRalt FPRalt FPPropalt]';
    % Predictive log-likelihood
    [loglike, ~, ~] = HMMLikelihood(ytest, HMModel, 'method', Estep, 'normalize', normflag);
    NLLTest(subj_i) = -loglike;
    HMModelSupervised{subj_i} = HMModel;
end
%% Save results
if robustMstepflag == true
    save(['ICASSP results/Supervised/EM/' ObsModel '/AR order ' num2str(p)...
        '/Robust_dmax' num2str(dmaxSec) 'sec.mat'], 'PerfTest', 'PerfTestAlt', 'NLLTest', 'HMModelSupervised')
else
    save(['ICASSP results/Supervised/EM/' ObsModel '/AR order ' num2str(p)...
        '/NoRobust_dmax' num2str(dmaxSec) 'sec.mat'], 'PerfTest', 'PerfTestAlt', 'NLLTest', 'HMModelSupervised')
end

end