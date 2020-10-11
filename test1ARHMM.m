%%
clearvars
close all
clc

% N = 1000;
% K = 3;
% typ = 'HMM';
% pik = [0.3 0.2 0.5]';
% A = [0.75 0.15 0.10;...
%     0.15 0.8 0.05;...
%     0.2 0.1 0.7];
% StateParameters.pi = pik;
% StateParameters.A = A;
% ObsParameters.model = 'Gaussian';
% ObsParameters.mu = [0 1 -2];
% ObsParameters.sigma = [0.1 0.2 0.05];

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
% Add zero-mean noise (0.1 standard deviation)
x = x + 0.1*randn(size(x));

disp('Generation done')

%%
Estep = 'scaling';
normflag = 0;
tic
HMModel = HMMLearning(x, K, 'type', 'ARHMM', 'ARorder', 8, 'normalize', normflag, 'Estep', Estep);
toc
HMModel.loglike
figure, plot(HMModel.loglike)

%%
[z, loglike1] = HMMInference(x, HMModel, 'normalize', normflag);
loglike1

loglike = HMMLikelihood(x, HMModel, 'method', 'scaling', 'normalize', normflag)