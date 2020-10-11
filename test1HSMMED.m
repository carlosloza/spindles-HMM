%% DO NOT USE THIS!!!!!!!!
clearvars
close all
clc

%N = 58000000;
N = 500;
K = 3;
typ = 'HMM';
pik = [0.3 0.2 0.5]';
A = [0.75 0.15 0.10;...
    0.15 0.8 0.05;...
    0.2 0.1 0.7];
muk = [0 1 -2]';
sigk = [0.1 0.2 0.05]';

rng(34)

HMMSeq = HMMGenerate(N, K, 'type', 'HMM', 'pi', pik, 'A', A, 'mu', muk, 'sigma', sigk);
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
dmax = 100;
tic
[HSMModel, muy] = HSMMLearning(HMMSeq.y, K, 'type', 'HSMMED', 'dmax', dmax, 'durationType', 'NonParametric');
toc
HSMModel.loglike
figure, plot(HSMModel.loglike)