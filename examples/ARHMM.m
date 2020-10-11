%% Test/example of ARHMM with univariate observations - sinusoids
% TODO: Nice plots and comparison to ground truth
% PSD plots
% Gaussian - scaling - PASS
% Gaussian - logsumexp - PASS
% Generalizedt - scaling - PASS
% Generalizedt - logsumexp - PASS
clearvars
close all
clc

% For reproducibility
rng(34)

%% Generate noisy synthetic data
K = 3;
Fs = 100;               % Sampling frequency
x = zeros(1, 3*Fs);
t = 1/Fs:1/Fs:3;
for i = 1:K
    t_aux = t(Fs*(i - 1) + 1:Fs*i);
    x(Fs*(i - 1) + 1:Fs*i) = sin(2*pi*10*i*t_aux);
end
% Add gaussian zero-mean noise (0.1 standard deviation)

x = x + random('Normal',0, 0.1, size(x));
%x = x + random('tLocationScale',0, 0.1, 5, size(x));

disp('Generation done')

%%
Estep = 'logsumexp';
normflag = 0;

ObsParametersLearn.model = 'Gaussian';

tic
HMModel = HMMLearning(x, K, 'type', 'ARHMM', 'ARorder', 8, 'normalize', normflag, 'Estep', Estep, 'ObsParameters', ObsParametersLearn);
toc
HMModel.loglike
figure, plot(HMModel.loglike)

%%
[z, loglike1] = HMMInference(x, HMModel, 'normalize', normflag);
loglike1

loglike = HMMLikelihood(x, HMModel, 'method', 'scaling', 'normalize', normflag)