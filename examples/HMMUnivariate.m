%% Test/example of HMM with univariate observations
% TODO: Nice plots and comparison to ground truth
% Gaussian - scaling - PASS
% Gaussian - logsumexp - PASS
% Generalizedt - scaling - PASS
% Generalized - logsumexp - PASS

clearvars
close all
clc

% For reproducibility
rng(34)

N = 10000;
K = 3;
typ = 'HMM';
pik = [0.3 0.2 0.5]';
A = [0.75 0.15 0.10;...
    0.15 0.8 0.05;...
    0.2 0.1 0.7];
StateParameters.pi = pik;
StateParameters.A = A;

% ObsParameters.model = 'Gaussian';
% ObsParameters.mu = [0 1 -2];
% ObsParameters.sigma = [0.1 0.2 0.05];

ObsParameters.model = 'Generalizedt';
ObsParameters.mu = [0 1 -2];
ObsParameters.sigma = [0.1 0.2 0.15];
%ObsParameters.nu = [4 7 10];
ObsParameters.nu = [1000 1000 1000];

[HMMSeq, HMModelSeq] = HMMGenerate(N, K, 'type', 'HMM', 'StateParameters', StateParameters, 'ObsParameters', ObsParameters);
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
[loglikeee, alphaEM, auxEM] = HMMLikelihood(HMMSeq.y, HMModelSeq, 'method', 'logsumexp', 'normalize', 0, 'returnAlpha', 1);
loglikeee

%%
Estep = 'logsumexp';
normflag = 0;
tic
%ObsParametersLearn.model = 'Generalizedt';
ObsParametersLearn.model = 'Gaussian';
HMModel = HMMLearning(HMMSeq.y, K, 'type', 'HMM', 'Estep', Estep, 'normalize', normflag, 'ObsParameters', ObsParametersLearn);
toc
HMModel.loglike
figure, plot(HMModel.loglike)

%%
[z, loglike1] = HMMInference(HMMSeq.y, HMModel, 'normalize', normflag);
loglike1
figure
subplot(2, 1, 1)
plot(HMMSeq.z)
subplot(2, 1, 2)
plot(z)

loglike = HMMLikelihood(HMMSeq.y, HMModel, 'method', 'scaling', 'normalize', normflag)