%% Test/example of ARHSMMED with univariate observations - sinusoids
% v2 - not autoregressive

clearvars
close all
clc

rng(34)

K = 2;
pik = [0.4 0.6]';

N = 10000;                 % number of sinusoid segments
%nsin = 100;
% No self-transitions
A = [0 1;...
    1 0];

StateParametersGen.pi = pik;
StateParametersGen.A = A;
DurationParametersGen.model = 'Poisson';
DurationParametersGen.lambda = [10 100];

ObsParametersGen.model = 'Gaussian';
ObsParametersGen.mu = [0 -2];
ObsParametersGen.sigma = [0.1 0.2];

[HMMSeq, HSMModelGen] = HSMMGenerate(N, K, 'type', 'HSMMED', 'StateParameters', StateParametersGen, 'ObsParameters', ObsParametersGen, 'DurationParameters', DurationParametersGen);
disp('Generation done')

%% Learning right dmax
Estep = 'logsumexp';
normflag = 0;

truedmax = 200;

DurationParameters.model = 'NonParametric';
DurationParameters.dmin = 1;
%DurationParameters.dmin = 1;
DurationParameters.dmax = truedmax;
ObsParameters.model = 'Gaussian';

tic
HMModel_1 = HMMLearning(HMMSeq.y, K, 'type', 'HSMMED', 'normalize', normflag, 'Estep', Estep, 'DurationParameters', DurationParameters, 'ObsParameters', ObsParameters);
toc

HMModel_1.loglike
%figure, plot(HMModel_1.loglike)

%%  Learning wrong dmax
wdmax = 20;

DurationParameters.model = 'NonParametric';
DurationParameters.dmin = 1;
DurationParameters.dmax = wdmax;
ObsParameters.model = 'Gaussian';

tic
HMModel_2 = HMMLearning(HMMSeq.y, K, 'type', 'HSMMED', 'normalize', normflag, 'Estep', Estep, 'DurationParameters', DurationParameters, 'ObsParameters', ObsParameters);
toc

HMModel_2.loglike
%figure, plot(HMModel_2.loglike)

%%
figure, subplot(2,2,1), stem(HMModel_1.DurationParameters.PNonParametric(1,:)), xlim([0 truedmax]), title([num2str(truedmax)])
subplot(2,2,2), stem(HMModel_1.DurationParameters.PNonParametric(2,:)), xlim([0 wdmax])
subplot(2,2,3), stem(HMModel_2.DurationParameters.PNonParametric(1,:)), xlim([0 truedmax]), title([num2str(wdmax)])
subplot(2,2,4), stem(HMModel_2.DurationParameters.PNonParametric(2,:)), xlim([0 wdmax])