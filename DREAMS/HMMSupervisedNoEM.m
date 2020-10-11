function HMMSupervisedNoEM(ObsModel, p, robustMstepflag, subTest)
% dmax in seconds
% dmax in seconds

%% Inference only
% Use logsumexp implementation because of artifacts!
% Methodology:
% 7 out of 8 subjects will be the training set
% Inside training set, estimate model parameters with complete data, i.e. labels
% Predict labels on 1 remaining subject (test set)
% 
% All recording will be downsampled or upsampled to have Fs=50Hz
% This function takes so long that it needs to be ran in a
% subject-by-subject basis

Fs = 50;
nSubTotal = 8;
K = 2;
dmin = p + 1;
normflag = 1;
Estep = 'logsumexp';

PerfTest = zeros(3, 1);
PerfTestAlt = zeros(3, 1);
NLLTest = 0;
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
HMModelIni.ObsParameters.meanModel = 'Linear';
HMModelIni.ARorder = p;
%HMModelIni.DurationParameters.dmax = dmax;
HMModelIni.DurationParameters.dmin = dmin;
HMModelIni.DurationParameters.model = 'NonParametric';

testSet = subTest;
trainSet = setdiff(1:nSubTotal, testSet);
%fprintf('Test Subject %d, p = %d \n', testSet, p)
HMModel = HMModelIni;
% Learning - complete data
for k = 1:numel(trainSet)
    TrainingStructure(k).y = ySeq{trainSet(k)};
    TrainingStructure(k).z = labelsGT{trainSet(k)};
end
HMModel = HMMLearningCompleteDataSleepSpindles(TrainingStructure, HMModel);

%     % Keep training
%     for k = 1:numel(trainSet)
%         yLearnSet{k} = ySeq{trainSet(k)};
%     end
%     % Smooth out duration distribution for duration of sleep spindles
%     HMModel = HMMLearning(yLearnSet, K, 'type', HMModel.type,...
%         'ARorder', p, 'Estep', Estep, 'normalize', normflag,...
%         'ObsParameters', HMModel.ObsParameters,...
%         'DurationParameters', DurationParameters,...
%         'SleepSpindles', true,...
%         'Fs', Fs, 'robustMstep', robustMstepflag);

clear TrainingStructure y

ytest = ySeq{testSet};
[labelsPred, loglike] = HMMInference(ytest, HMModel, 'normalize', normflag);
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
PerfTest(:, 1) = [F1 MCC FPProp]';
PerfTestAlt(:, 1) = [TPRalt FPRalt FPPropalt]';
NLLTest = -loglike;                 % This is not OK, loglike cannot come from Viterbi algorithm
HMModelSupervised = HMModel;  
    
%% Save results
if robustMstepflag == true
    save(['ICASSP results/Supervised/NoEM/' ObsModel '/AR order ' num2str(p)...
        '/Robust_Subj' num2str(subTest) '.mat'], 'PerfTest', 'PerfTestAlt', 'NLLTest', 'HMModelSupervised')
else
    save(['ICASSP results/Supervised/NoEM/' ObsModel '/AR order ' num2str(p)...
        '/NoRobust_Subj' num2str(subTest) '.mat'], 'PerfTest', 'PerfTestAlt', 'NLLTest', 'HMModelSupervised')
end

end