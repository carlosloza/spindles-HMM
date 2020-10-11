%%
clearvars
close all
clc

rng(34)

K = 3;
pik = [0.3 0.2 0.5]';
%nsin = 2400;                 % number of sinusoid segments
nsin = 100;
% No self-transitions
A = [0.0 0.35 0.65;...
    0.75 0.0 0.25;...
    0.40 0.60 0.0];

lambdak = [50 80 100];
%sigk = [2 5 8];
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
    if dState <= 0
        disp('Hey!')
    end
    t = 1/Fs:1/Fs:dState/Fs;
    xaux = sin(2*pi*10*zaux*t);
    x = [x xaux];
end
% Add zero-mean noise (0.1 standard deviation)
x = x + 0.1*randn(size(x));
%% Learning
Estep = 'scaling';
DurationParameters.model = 'Poisson';
normflag = 1;
DurationParameters.dmin = 20;
DurationParameters.dmax = 200;
tic
[HSMModel, muy] = HSMMLearning(x, K, 'type', 'ARHSMMED', 'ARorder', 8, 'DurationParameters', DurationParameters, 'Estep', Estep, 'normalize', normflag);
toc
HSMModel.loglike
figure, plot(HSMModel.loglike)

%% Likelihood
tic
loglike = HSMMLikelihood(x, HSMModel, 'method', 'scaling', 'normalize', normflag, 'returnAlpha', 0);
toc
loglike

%% inference - Viterbi
tic
[z, ds, loglike1] = HSMMInference(x, HSMModel, 'normalize', normflag);
loglike1
toc

%% compare durations
dsgt = [];
for i = 1:length(dStateall)
    dsgt = [dsgt dStateall(i):-1:1];
end

