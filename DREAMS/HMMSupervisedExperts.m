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

ObsModel = 'Generalizedt';
p = 5;
robustIni = true;
dmaxSec = 30;

Fs = 50;
nSubTotal = 8;
K = 2;
dmax = round(dmaxSec*Fs);
dmin = p + 1;
normflag = 1;
Estep = 'logsumexp';

PerfTest = zeros(3, nSubTotal);
NLLTest = zeros(2, nSubTotal);
HMModelSupervised = cell(1, nSubTotal);
PerfTestExpert = cell(2,2);
PerfTestExpert(:) = {nan(3, nSubTotal)};

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
HMModelIni.Fs = Fs;
HMModelIni.StateParameters.K = K;
HMModelIni.normalize = normflag;
HMModelIni.robustIni = robustIni;
HMModelIni.ObsParameters.model = ObsModel;
HMModelIni.ObsParameters.meanModel = 'Linear';
HMModelIni.ARorder = p;
HMModelIni.DurationParameters.dmin = dmin;
HMModelIni.DurationParameters.dmax = dmax;
HMModelIni.DurationParameters.model = 'NonParametric';
for subj_i = 1:nSubTotal
    testSet = subj_i;
    trainSet = setdiff(1:nSubTotal, testSet);
    fprintf('Test Subject %d, p = %d \n', testSet, p)    
    % Expert 1
    HMModel = HMModelIni;
    % Learning - complete data
    TrainingStructure(numel(trainSet)) = struct();
    for k = 1:numel(trainSet)
        TrainingStructure(k).y = ySeq{trainSet(k)};
        TrainingStructure(k).z = labelsGTExpert1{trainSet(k)};
    end   
    HMModelExperts{1} = HMMLearningCompleteDataSleepSpindles(TrainingStructure, HMModel);     
    clear TrainingStructure   
    % Expert 2
    HMModel = HMModelIni;
    % Learning - complete data
    trainSet = setdiff(trainSet, [7 8]);            % these subjects do not have labels from expert 2
    TrainingStructure(numel(trainSet)) = struct();
    for k = 1:numel(trainSet)
        TrainingStructure(k).y = ySeq{trainSet(k)};
        TrainingStructure(k).z = labelsGTExpert2{trainSet(k)};
    end   
    HMModelExperts{2} = HMMLearningCompleteDataSleepSpindles(TrainingStructure, HMModel);     
    clear TrainingStructure
    % Test   
    ytest = ySeq{testSet};
    % 4 Cases
    labelsPredExpSep = zeros(2, numel(labelsGT{testSet}(p+1:end)));
    for i = 1:2
        labelsPred = HMMInference(ytest, HMModelExperts{i}, 'normalize', normflag);       
        % Predictive log-likelihood.
        loglike = HMMLikelihood(ytest, HMModelExperts{i}, 'normalize', normflag);
        NLLTest(i, subj_i) = -loglike;
        % Predicted labels
        labelsPred = labelsPred(p+1:end);
        labelsPredExpSep(i, :) = labelsPred;
        for j = 1:2
            if j == 1
                PerfTestExpert{i, j}(:, subj_i) = PerfMeasures(labelsGTExpert1{testSet}(p+1:end), labelsPred, Fs);
            elseif j == 2
                if testSet <= 6
                    PerfTestExpert{i, j}(:, subj_i) = PerfMeasures(labelsGTExpert2{testSet}(p+1:end), labelsPred, Fs);
                end
            end
        end     
    end
    % Aggregated case
    clear labelsPred
    labelsPred = sum(labelsPredExpSep, 1)/2;
    labelsPred(labelsPred > 1) = 2;
    PerfTest(:, subj_i) = PerfMeasures(labelsGT{testSet}(p+1:end), labelsPred, Fs);
    HMModelSupervised{subj_i} = HMModelExperts;
end
%% Display results
meanMCC = cell2mat(cellfun(@(x) mean(x(2,:), 'omitnan'), PerfTestExpert, 'UniformOutput', false));
T = array2table(meanMCC,...
    'RowNames', {'Expert1_Model', 'Expert2_Model'},...
    'VariableNames', {'Expert1_GroundTruth', 'Expert2_GroundTruth'});
fprintf('Average MCC between inference output of models trained with expert labels \n and ground truth (supervised scheme) \n \n')
disp(T)
fprintf('Average predictive measures of aggregated model. F1 score: %.4f, MCC: %.4f \n',...
    mean(PerfTest(1,:)), mean(PerfTest(2,:)))
fprintf('Average predictive negative log-likelihood. Expert1: %d, Expert2 %d \n',...
    round(mean(NLLTest(1,:))), round(mean(NLLTest(2,:))))