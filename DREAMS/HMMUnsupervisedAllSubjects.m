function HMMUnsupervisedAllSubjects(ObsModel, p, robustMstepflag, dmaxSec)
% dmax in seconds

%% Learning only - All subjects - For manuscript figures
% Use logsumexp implementation because of artifacts!
% Methodology:
% All subjects will be the training set
% Inside training set, run EM with estimated parameters as initial conditions 
% (smooth out the duration distribution for spindles)
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

% PerfTest = zeros(3, nSubTotal);
% PerfTestAlt = zeros(3, nSubTotal);
% NLLTest = zeros(1, nSubTotal);
% HMModelSupervised = cell(1, nSubTotal);
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
HMModelIni.DurationParameters.dmax = dmax;
HMModelIni.DurationParameters.dmin = dmin;
HMModelIni.DurationParameters.model = 'NonParametric';
% DurationParameters.dmax = dmax;
% DurationParameters.dmin = dmin;
% DurationParameters.model = 'NonParametric';

trainSet = 1:nSubTotal;
HMModel = HMModelIni;
% Keep training
for k = 1:numel(trainSet)
    yTrainSet{k} = ySeq{trainSet(k)};
end
% Smooth out duration distribution for duration of sleep spindles
%DurationParameters.Ini = HMModel.DurationParameters.Ini;
HMModel = HMMLearning(yTrainSet, K, 'type', HMModelIni.type,...
        'ARorder', p, 'Estep', Estep, 'normalize', normflag,...
        'ObsParameters', HMModelIni.ObsParameters,...
        'DurationParameters', HMModelIni.DurationParameters,...
        'robustMstep', false, 'Fs', Fs, 'SleepSpindles', true);
%% Save results
if robustMstepflag == true
    save(['ICASSP results/Unsupervised/' ObsModel '/AR order ' num2str(p)...
        '/Robust_dmax' num2str(dmaxSec) 'sec_ALLSUBJECTS.mat'], 'HMModel')
else
    save(['ICASSP results/Unsupervised/' ObsModel '/AR order ' num2str(p)...
        '/NoRobust_dmax' num2str(dmaxSec) 'sec_ALLSUBJECTS.mat'], 'HMModel')
end

end