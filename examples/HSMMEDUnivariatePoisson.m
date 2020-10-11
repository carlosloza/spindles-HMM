%% Test/example of HSMMED with univariate observations and Poisson distributed durations
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
N = 5000;              % CAUTION: gave a non-convergence warning on the initial t-distribution fit when N = 1000
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

ObsParametersGen.model = 'Gaussian';
ObsParametersGen.mu = [0 1 -2];
ObsParametersGen.sigma = [0.1 0.2 0.05];

% ObsParametersGen.model = 'Generalizedt';
% ObsParametersGen.mu = [0 1 -2];
% ObsParametersGen.sigma = [0.1 0.2 0.15];
% ObsParametersGen.nu = [4 7 10];

[HMMSeq, HSMModelGen] = HSMMGenerate(N, K, 'type', 'HSMMED', 'StateParameters', StateParametersGen, 'ObsParameters', ObsParametersGen, 'DurationParameters', DurationParametersGen);
disp('Generation done')
%%
figure
subplot(2, 1, 1)
plot(HMMSeq.z)
subplot(2, 1, 2)
plot(HMMSeq.y)

%%
figure
subplot(2,1,1)
hist((HMMSeq.y), 50)
subplot(2,1,2)
hist(zscore(HMMSeq.y), 50)

%%
DurationParameters.model = 'Poisson';
DurationParameters.dmax = 50;
ObsParameters.model = 'Gaussian';
%ObsParameters.model = 'Generalizedt';
Estep = 'logsumexp';
normflag = 0;
tic
HMModel = HMMLearning(HMMSeq.y, K, 'type', 'HSMMED', 'DurationParameters', DurationParameters, 'Estep', Estep, 'normalize', normflag, 'ObsParameters', ObsParameters);
toc
HMModel.loglike
figure, plot(HMModel.loglike)

%%
tic
[z, loglike1] = HMMInference(HMMSeq.y, HMModel, 'normalize', normflag);
toc
loglike1
figure
subplot(2, 1, 1)
plot(HMMSeq.z)
subplot(2, 1, 2)
plot(z)

loglike = HMMLikelihood(HMMSeq.y, HMModel, 'method', 'scaling', 'normalize', normflag)