%% Test/example of ARHSMMED with univariate observations - sinusoids
% NEW RULE:
% MAKE SURE DMIN IS AT LEAST EQUAL TO AR ORDER, P
% CONCLUSIONS:
% SETTING SHORTER DMAX MAINLY AFFECTS THE TRANSITION MATRIX (ROWS WHERE MAXIMUM
% DURATION IN THAT ROW/MODE IS LARGER THAN SET DMAX) AND DURATION
% DISTRIBUTIONS WITH MAXIMUM POSSIBLE DURATION SMALLER THAN SET DMAX
% AR COEFFICIENTS AND RESIDUAL VARIANCE ARE NOT AFFECTED
% Poisson distributed durations

clearvars
close all
clc

rng(34)

K = 2;
pik = [0.4 0.6]';

nsin = 100;                 % number of sinusoid segments
%nsin = 100;
% No self-transitions
A = [0 1;...
    1 0];

%lambdak = [20 60];
lambdak = [20 100];
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

%% Learning right dmax
Estep = 'logsumexp';
normflag = 0;
p = 8;

truedmax = 200;

DurationParameters.model = 'NonParametric';
DurationParameters.dmin = p + 1;
%DurationParameters.dmin = 1;
DurationParameters.dmax = truedmax;
ObsParameters.model = 'Gaussian';

tic
HMModel_1 = HMMLearning(x, K, 'type', 'ARHSMMED', 'ARorder', p, 'normalize', normflag, 'Estep', Estep, 'DurationParameters', DurationParameters, 'ObsParameters', ObsParameters);
toc

HMModel_1.loglike
%figure, plot(HMModel_1.loglike)

%%  Learning wrong dmax

wdmax = 30;

DurationParameters.model = 'NonParametric';
DurationParameters.dmin = p + 1;
DurationParameters.dmax = wdmax;
ObsParameters.model = 'Gaussian';

tic
HMModel_2 = HMMLearning(x, K, 'type', 'ARHSMMED', 'ARorder', 8, 'normalize', normflag, 'Estep', Estep, 'DurationParameters', DurationParameters, 'ObsParameters', ObsParameters);
toc

HMModel_2.loglike
%figure, plot(HMModel_2.loglike)

%%
figure, subplot(2,2,1), stem(HMModel_1.DurationParameters.PNonParametric(1,:)), xlim([0 truedmax]), title([num2str(truedmax)])
subplot(2,2,2), stem(HMModel_1.DurationParameters.PNonParametric(2,:)), xlim([0 truedmax])
subplot(2,2,3), stem(HMModel_2.DurationParameters.PNonParametric(1,:)), xlim([0 truedmax]), title([num2str(wdmax)])
subplot(2,2,4), stem(HMModel_2.DurationParameters.PNonParametric(2,:)), xlim([0 truedmax])

%%
figure
subplot(2,2,1)
[HH, FF] = freqz(1, [1; HMModel_1.ObsParameters.meanParameters(1).Coefficients], 1024, 100);
plot(FF, 20*log(abs(HH)))
subplot(2,2,2)
[HH, FF] = freqz(1, [1; HMModel_1.ObsParameters.meanParameters(2).Coefficients], 1024, 100);
plot(FF, 20*log(abs(HH)))
subplot(2,2,3)
[HH, FF] = freqz(1, [1; HMModel_2.ObsParameters.meanParameters(1).Coefficients], 1024, 100);
plot(FF, 20*log(abs(HH)))
subplot(2,2,4)
[HH, FF] = freqz(1, [1; HMModel_2.ObsParameters.meanParameters(2).Coefficients], 1024, 100);
plot(FF, 20*log(abs(HH)))