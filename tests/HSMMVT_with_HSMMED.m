%% 

clearvars
close all
clc

% For reproducibility
rng(34)

%%
N = 10000;              % CAUTION: gave a non-convergence warning on the initial t-distribution fit when N = 1000
K = 3;
pik = [0.35 0.15 0.5]';
% No self-transitions
A = [0.0 0.35 0.65;...
    0.75 0.0 0.25;...
    0.40 0.60 0.0];
StateParametersGen.pi = pik;
StateParametersGen.A = A;
DurationParametersGen.model = 'Poisson';
DurationParametersGen.lambda = [2 4 8];

ObsParametersGen.model = 'Gaussian';
ObsParametersGen.mu = [0 1 -2];
ObsParametersGen.sigma = [0.1 0.2 0.05];

% ObsParametersGen.model = 'Generalizedt';
% ObsParametersGen.mu = [0 1 -2];
% ObsParametersGen.sigma = [0.1 0.2 0.15];
% ObsParametersGen.nu = [4 7 10];

[HMMSeq, HSMModelGen] = HSMMGenerate(N, K, 'type', 'HSMMED', 'StateParameters', StateParametersGen, 'ObsParameters', ObsParametersGen, 'DurationParameters', DurationParametersGen);
disp('Generation done')
%% Likelihood only
disp('HSMMED')
HSMModelGen.DurationParameters.dmax = 20;
loglike1 = HMMLikelihood(HMMSeq.y, HSMModelGen, 'method', 'logsumexp', 'normalize', 0);
round(loglike1)

disp('HSMMVT')
HSMModel = HSMModelGen;
HSMModel.type = 'HSMMVT';
HSMModel.StateParameters.A = repmat(A, 1, 1, HSMModel.DurationParameters.dmax);
HSMModel.DurationParameters.DurationIntervals = 1:HSMModel.DurationParameters.dmax;
[loglike2, alphaEM, auxEM] = HMMLikelihood(HMMSeq.y, HSMModel, 'method', 'logsumexp', 'normalize', 0, 'returnAlpha', 1);
round(loglike2)

%% Learning
Estep = 'logsumexp';
normflag = 0;
ObsParameters1.model = 'Gaussian';

disp('HSMMED')
DurationParameters1.model = 'NonParametric';
%DurationParameters1.lambda = [2 4 8];
DurationParameters1.dmax = 10;
HMModel_1 = HMMLearning(HMMSeq.y, K, 'type', 'HSMMED', 'DurationParameters', DurationParameters1,...
    'Estep', Estep, 'normalize', normflag, 'ObsParameters', ObsParameters1);
HMModel_1.loglike

disp('HSMMVT')
DurationParameters2.model = 'NonParametric';
%DurationParameters2.lambda = [2 4 8];
DurationParameters2.dmax = 10;
%StateParameters.A = repmat(A, 1, 1, HSMModel.DurationParameters.dmax);
DurationParameters2.DurationIntervals = 1:DurationParameters2.dmax;

HMModel_2 = HMMLearning(HMMSeq.y, K, 'type', 'HSMMVT', 'DurationParameters', DurationParameters2,...
    'Estep', Estep, 'normalize', normflag, 'ObsParameters', ObsParameters1);
HMModel_2.loglike