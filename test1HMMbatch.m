%%
clearvars
close all
clc

nSeq = 100;

%N = 58000000;
N = 1000;
K = 3;
typ = 'HMM';
pik = [0.3 0.2 0.5]';
A = [0.75 0.15 0.10;...
    0.15 0.8 0.05;...
    0.2 0.1 0.7];
StateParameters.pi = pik;
StateParameters.A = A;
ObsParameters.model = 'Gaussian';
ObsParameters.mu = [0 1 -2];
ObsParameters.sigma = [0.1 0.2 0.05];

rng(34)

ySeq = cell(1, nSeq);
zIni = zeros(1, nSeq);
NSeq = randi([80 120], 1, nSeq);
for iSeq = 1:nSeq
    HMMSeq = HMMGenerate(NSeq(iSeq), K, 'type', 'HMM', 'StateParameters', StateParameters, 'ObsParameters', ObsParameters);
    zIni(iSeq) = HMMSeq.z(1);
    ySeq{iSeq} = HMMSeq.y;
end
disp('Generation done')
%%
% figure
% subplot(2, 1, 1)
% plot(HMMSeq.z)
% subplot(2, 1, 2)
% plot(HMMSeq.y)
% 
% %%
% figure
% subplot(2,1,1)
% hist((HMMSeq.y), 50)
% subplot(2,1,2)
% hist(zscore(HMMSeq.y), 50)

%%
Estep = 'scaling';
normflag = 1;
tic
HMModel = HMMLearning(ySeq, K, 'type', 'ARHMM', 'Estep', Estep, 'normalize', normflag, 'ARorder', 8);
%HMModel = HMMLearning(ySeq, K, 'type', 'HMM', 'Estep', Estep, 'normalize', normflag);
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