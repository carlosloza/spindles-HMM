%% Test/example of ARHSMMED with univariate observations - sinusoids
% Poisson distributed durations
% Gaussian observations - scaling - PASS
% Gaussian observations - logsumexp - PASS
% Generalizedt observations - scaling - PASS
% Generalizedt observations - logsumexp - PASS

clearvars
close all
clc

rng(34)

K = 3;
pik = [0.3 0.2 0.5]';
nsin = 100;                 % number of sinusoid segments
%nsin = 100;
% No self-transitions
A = [0.0 0.35 0.65;...
    0.75 0.0 0.25;...
    0.40 0.60 0.0];

lambdak = [50 80 100];
Fs = 100;               % Sampling frequency

x = [];
zaux = find(mnrnd(1, pik));
dState = poissrnd(lambdak(zaux));
dStateall = dState;
t = 1/Fs:1/Fs:dState/Fs;
xaux = sin(2*pi*10*zaux*t);
x = [x xaux];
for i = 1:nsin
    zaux = find(mnrnd(1, A(zaux, :)));
    %zaux = find(mnrnd(1, pik));
    dState = poissrnd(lambdak(zaux));
    dStateall = [dStateall dState];
    t = 1/Fs:1/Fs:dState/Fs;
    xaux = sin(2*pi*10*zaux*t);
    x = [x xaux];
end
% Add gaussian zero-mean noise (0.1 standard deviation)

x = x + random('Normal',0, 0.1, size(x));
%x = x + random('tLocationScale',0, 0.1, 5, size(x));

%% Learning
Estep = 'logsumexp';
normflag = 0;

DurationParameters.model = 'Poisson';
DurationParameters.dmin = 20;
DurationParameters.dmax = 200;

ObsParameters.model = 'Gaussian';
%ObsParameters.model = 'Generalizedt';

tic
HMModel = HMMLearning(x, K, 'type', 'ARHSMMED', 'ARorder', 8, 'normalize', normflag, 'Estep', Estep, 'DurationParameters', DurationParameters, 'ObsParameters', ObsParameters);
toc

HMModel.loglike
figure, plot(HMModel.loglike)

%%
[z, loglike1, drem] = HMMInference(x, HMModel, 'normalize', normflag);
loglike1

loglike = HMMLikelihood(x, HMModel, 'method', 'scaling', 'normalize', normflag)

% %% Likelihood
% tic
% loglike = HSMMLikelihood(x, HMModel, 'method', 'scaling', 'normalize', normflag, 'returnAlpha', 0);
% toc
% loglike
% 
% %% inference - Viterbi
% tic
% [z, ds, loglike1] = HSMMInference(x, HSMModel, 'normalize', normflag);
% loglike1
% toc
% 
% %% compare durations
% dsgt = [];
% for i = 1:length(dStateall)
%     dsgt = [dsgt dStateall(i):-1:1];
% end

