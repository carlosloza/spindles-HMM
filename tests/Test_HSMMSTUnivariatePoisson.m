%% Test/example of HSMMEDST with univariate observations and Poisson distributed durations
% TODO: Nice plots and comparison to ground truth
% Gaussian observations - scaling - PASS
% Gaussian observations - logsumexp - PASS
% Generalizedt observations - scaling - PASS
% Generalizedt observations - logsumexp - PASS

clearvars
close all
clc

% For reproducibility
rng(34)

%%
N = 50000;              % CAUTION: gave a non-convergence warning on the initial t-distribution fit when N = 1000
K = 3;
dmax = 20;
distInt = [1 5 dmax];
pik = [0.35 0.15 0.5]';
% Yes self-transitions
A(:, :, 1) = [0.10 0.35 0.55;...
    0.55 0.20 0.25;...
    0.30 0.60 0.10];
A(:, :, 2) = [0.30 0.40 0.30;...
    0.20 0.10 0.70;...
    0.40 0.35 0.25];
StateParametersGen.pi = pik;
StateParametersGen.A = A;
DurationParametersGen.model = 'Poisson';
DurationParametersGen.lambda = [2 4 8];
DurationParametersGen.DurationIntervals = distInt;
DurationParametersGen.dmax = dmax;

ObsParametersGen.model = 'Gaussian';
ObsParametersGen.mu = [0 1 -2];
ObsParametersGen.sigma = [0.1 0.2 0.05];

% ObsParametersGen.model = 'Generalizedt';
% ObsParametersGen.mu = [0 1 -2];
% ObsParametersGen.sigma = [0.1 0.2 0.15];
% ObsParametersGen.nu = [4 7 10];

[HMMSeq, HSMModelGen] = HSMMVTGenerate(N, K, 'type', 'HSMMVT', 'StateParameters', StateParametersGen,...
    'ObsParameters', ObsParametersGen, 'DurationParameters', DurationParametersGen);
disp('Generation done')

%%
tic
%profile on
[loglikeGen, alphaEM, auxEM] = HMMLikelihood(HMMSeq.y, HSMModelGen, 'method', 'logsumexp', 'normalize', 0, 'returnAlpha', 1);
%profile viewer
toc
loglikeGen


%%
DurationParameters.model = 'Poisson';
DurationParameters.dmax = dmax;
DurationParameters.DurationIntervals = distInt;
ObsParameters.model = 'Gaussian';

%ObsParameters.model = 'Generalizedt';
Estep = 'logsumexp';
normflag = 0;
tic
HMModel = HMMLearning(HMMSeq.y, K, 'type', 'HSMMVT', 'DurationParameters', DurationParameters, 'Estep', Estep, 'normalize', normflag, 'ObsParameters', ObsParameters);
toc
HMModel.loglike
figure, plot(HMModel.loglike)

%% Inference
normflag = 0;
[z, loglike1] = HMMInference(HMMSeq.y, HSMModelGen, 'normalize', normflag);
% 
% %%
% [z, loglike1] = HMMInference(HMMSeq.y, HMModel, 'normalize', normflag);
% loglike1
% figure
% subplot(2, 1, 1)
% plot(HMMSeq.z)
% subplot(2, 1, 2)
% plot(z)
% 
% loglike = HMMLikelihood(HMMSeq.y, HMModel, 'method', 'scaling', 'normalize', normflag)