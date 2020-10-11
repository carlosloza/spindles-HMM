%% Test/example of HMM with univariate observations. Batch implementation
% TODO: Nice plots and comparison to ground truth
% Gaussian - scaling - PASS
% Gaussian - logsumexp - PASS
% Generalizedt - scaling - PASS
% Generalizedt - logsumexp - PASS
clearvars
close all
clc

% For reproducibility
rng(34)
% Batch size
nSeq = 100;

Nmin = 800;
Nmax = 1200;
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
ObsParameters.nu = [4 7 10];

ySeq = cell(1, nSeq);
NSeq = randi([Nmin Nmax], 1, nSeq);
for iSeq = 1:nSeq
    HMMSeq = HMMGenerate(NSeq(iSeq), K, 'type', 'HMM', 'StateParameters', StateParameters, 'ObsParameters', ObsParameters);
    ySeq{iSeq} = HMMSeq.y;
end
disp('Generation done')

%%
Estep = 'scaling';
normflag = 0;
tic
%ObsParametersLearn.model = 'Gaussian';
ObsParametersLearn.model = 'Generalizedt';
HMModel = HMMLearning(ySeq, K, 'type', 'HMM', 'Estep', Estep, 'normalize', normflag, 'ObsParameters', ObsParametersLearn);
toc
HMModel.loglike
figure, plot(HMModel.loglike)