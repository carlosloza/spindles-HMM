%% Test/example of HSMMED with univariate observations and Poisson distributed durations. 
% Batch implementation
% TODO: Nice plots and comparison to ground truth
% Gaussian observations - scaling - PASS
% Gaussian observations - logsumexp - PASS
% Generalizedt observations - scaling - PASS
% Generalizedt observations - logsumexp - PASS

clearvars
close all
%clc

% For reproducibility
rng(34)

%%
nSeq = 100;
Nmin = 400;
Nmax = 600;

K = 3;
pik = [0.3 0.2 0.5]';
% No self-transitions
A = [0.0 0.35 0.65;...
    0.75 0.0 0.25;...
    0.40 0.60 0.0];
StateParametersGen.pi = pik;
StateParametersGen.A = A;

DurationParametersGen.model = 'Poisson';
DurationParametersGen.lambda = [2 6 10];

% ObsParametersGen.model = 'Gaussian';
% ObsParametersGen.mu = [0 1 -2];
% ObsParametersGen.sigma = [0.1 0.2 0.05];

ObsParametersGen.model = 'Generalizedt';
ObsParametersGen.mu = [0 1 -2];
ObsParametersGen.sigma = [0.1 0.2 0.15];
ObsParametersGen.nu = [4 7 10];

ySeq = cell(1, nSeq);
NSeq = randi([Nmin Nmax], 1, nSeq);
for iSeq = 1:nSeq
    [HMMSeq, HSMModelGen] = HSMMGenerate(NSeq(iSeq), K, 'type', 'HSMMED', 'StateParameters', StateParametersGen, 'ObsParameters', ObsParametersGen, 'DurationParameters', DurationParametersGen);
    ySeq{iSeq} = HMMSeq.y;
end
disp('Generation done')

%%
DurationParameters.model = 'Poisson';
DurationParameters.dmax = 50;

%ObsParameters.model = 'Gaussian';
ObsParameters.model = 'Generalizedt';

Estep = 'logsumexp';
normflag = 0;
tic
HMModel = HMMLearning(ySeq, K, 'type', 'HSMMED', 'DurationParameters', DurationParameters, 'Estep', Estep, 'normalize', normflag, 'ObsParameters', ObsParameters);
toc
HMModel.loglike
figure, plot(HMModel.loglike)