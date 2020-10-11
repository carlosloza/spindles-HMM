%%
clearvars
close all
clc

%N = 58000000;
N = 1000;
K = 3;
d = 2;
typ = 'HMM';
pik = [0.3 0.2 0.5]';
A = [0.75 0.15 0.10;...
    0.15 0.8 0.05;...
    0.2 0.1 0.7];
StateParameters.pi = pik;
StateParameters.A = A;
ObsParameters.model = 'MultivariateGaussian';
ObsParameters.mu = [10 15 -5; -5 5 0];

% ObsParameters.sigma(:,:,1) = [2 0.5; 0.5 1];
% ObsParameters.sigma(:,:,2) = [1.5 0.1; 0.1 2.5];
% ObsParameters.sigma(:,:,3) = [1.25 0.2; 0.2 1.25];

ObsParameters.sigma(:,:,1) = [5 1; 1 3];
ObsParameters.sigma(:,:,2) = [2.5 -1.5; -1.5 4];
ObsParameters.sigma(:,:,3) = [5 2; 2 5];

rng(34)

HMMSeq = HMMGenerate(N, K, 'type', 'HMM', 'StateParameters', StateParameters, 'ObsParameters', ObsParameters);
disp('Generation done')
%%
figure, scatter(HMMSeq.y(1,:), HMMSeq.y(2,:))

normflag = 0;

%%
HMMGT.type = 'HMM';
HMMGT.StateParameters.K = K;
HMMGT.StateParameters.pi = StateParameters.pi;
HMMGT.StateParameters.A = A;
HMMGT.ObsParameters = ObsParameters;
loglike = HMMLikelihood(HMMSeq.y, HMMGT, 'method', 'logsumexp', 'normalize', normflag);

%%
Estep = 'logsumexp';

tic
HMModel = HMMLearning(HMMSeq.y, K, 'type', 'HMM', 'Estep', Estep, 'normalize', normflag);
toc
HMModel.loglike
figure, plot(HMModel.loglike)

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