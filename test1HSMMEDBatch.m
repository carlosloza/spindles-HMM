%% Poisson model for durations
clearvars
close all
clc

nSeq = 10;

%N = 58000000;
%N = 500;
%N = 6000;
%N = 100000;
%N = 360000;
%N = 180000;
K = 3;
pik = [0.3 0.2 0.5]';
% No self-transitions
A = [0.0 0.35 0.65;...
    0.75 0.0 0.25;...
    0.40 0.60 0.0];
StateParameters.pi = pik;
StateParameters.A = A;
ObsParameters.model = 'Gaussian';
ObsParameters.mu = [0 1 -2];
ObsParameters.sigma = [0.1 0.2 0.05];
DurationParametersGen.model = 'Poisson';
DurationParametersGen.lambda = [2 6 10];

rng(34)

ySeq = cell(1, nSeq);
NSeq = randi([400 600], 1, nSeq);
for iSeq = 1:nSeq
    if iSeq == 5
        dfg = 1;
    end
    [HMMSeq, HSMModelGen] = HSMMGenerate(NSeq(iSeq), K, 'type', 'HSMMED', 'StateParameters', StateParameters, 'ObsParameters', ObsParameters, 'DurationParameters', DurationParametersGen);
    ySeq{iSeq} = HMMSeq.y;
end
disp('Generation done')
%%
% figure
% subplot(2, 1, 1)
% plot(HMMSeq.z)
% subplot(2, 1, 2)
% plot(HMMSeq.y)
% 
% %%
% figure
% subplot(2,1,1)
% hist((HMMSeq.y), 50)
% subplot(2,1,2)
% hist(zscore(HMMSeq.y), 50)

%%
DurationParameters.model = 'Poisson';
DurationParameters.dmax = 50;
Estep = 'scaling';
normflag = 0;
tic
%profile on
HSMModel = HSMMLearning(ySeq, K, 'type', 'HSMMED', 'DurationParameters', DurationParameters, 'Estep', Estep, 'normalize', normflag);
%profile viewer
toc
HSMModel.loglike
figure, plot(HSMModel.loglike)