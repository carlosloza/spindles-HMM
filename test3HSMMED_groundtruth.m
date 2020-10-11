%% Gaussian model for durations
% Parameter learning, likelihood and Viterbi inference
% ground truth parameters are provided as initializations
clearvars
close all
clc

%N = 58000000;
N = 500;
N = 6000;
%N = 100000;
%N = 360000;
%N = 180000;
K = 3;
pik = [0.3 0.2 0.5]';
% No self-transitions
A = [0.0 0.35 0.65;...
    0.75 0.0 0.25;...
    0.40 0.60 0.0];
StateParameters.pi = pik;
StateParameters.A = A;
ObsParameters.model = 'Gaussian';
ObsParameters.mu = [0 1 -2];
ObsParameters.sigma = [0.1 0.2 0.05];
DurationParametersGen.model = 'Gaussian';
DurationParametersGen.mu = [10 30 50];
DurationParametersGen.sigma = [2 5 8];

rng(34)

[HMMSeq, HSMModelGen] = HSMMGenerate(N, K, 'type', 'HSMMED', 'StateParameters', StateParameters, 'ObsParameters', ObsParameters, 'DurationParameters', DurationParametersGen);
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
edges = 0.5:1:max(HMMSeq.dState)+0.5;
%[N, edges] = histcounts(HMMSeq.dState, edges);
DurationParametersGen.dmax = max(HMMSeq.dState);
aux = cell(1, K);
for i = 1:numel(HMMSeq.dState)
    z = HMMSeq.z(min([sum(HMMSeq.dState(1:i)) numel(HMMSeq.y)]));
    aux{z} = [aux{z} HMMSeq.dState(i)];    
end
dcount = zeros(K, max(HMMSeq.dState));
for k = 1:K
    [N, ~] = histcounts(aux{k}, edges);
    dcount(k, :) = N/sum(N);
end
DurationParametersGen.dmax = max(HMMSeq.dState);
DurationParametersGen.model = 'NonParametric';
DurationParametersGen.PNonParametric = dcount;

%DurationParametersGen.dmax = 100;

StateParameters.A(:, :, 2) = eye(K);

normflag = 0;
Estep = 'scaling';
tic
%profile on
[HSMModel, muy] = HSMMLearning(HMMSeq.y, K, 'type', 'HSMMED', 'ObsParameters', ObsParameters,...
    'StateParameters', StateParameters, 'DurationParameters', DurationParametersGen,...
    'Estep', Estep, 'normalize', normflag);
%profile viewer
toc
HSMModel.loglike
figure, plot(HSMModel.loglike)

%% likelihood
tic
loglike = HSMMLikelihood(HMMSeq.y, HSMModel, 'method', 'scaling', 'normalize', normflag, 'returnAlpha', 0);
toc
loglike

%% inference - Viterbi
tic
[z, ds, loglike1] = HSMMInference(HMMSeq.y, HSMModel, 'normalize', normflag);
loglike1
toc

%% compare durations
dsgt = [];
for i = 1:length(HMMSeq.dState)
    dsgt = [dsgt HMMSeq.dState(i):-1:1];
end
dsgt = dsgt(1:N);