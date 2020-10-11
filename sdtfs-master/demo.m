%% EEG pre-processor for sleep spindle detection
%
% This script demonstrates the use of dual Basis Pursuit Denoising for
% spindle detection. Please read the paper below for details. 
%
% Ankit Parekh.
% Last Edit: 4/15/17
% Contact: ankit.parekh@nyu.edu
% 
% Please cite as: 
% Sleep spindle detection using time-frequency sparsity. 
% A. Parekh, I. W. Selesnick, D. M. Rapoport, I. Ayappa. 
% IEEE Signal Processing in Medicine and Biology Symposium, Dec. 2014.

%% Load the data and initializze
clear; close all; clc;
I = sqrt(-1);
dB = @(x) 20 * log10(abs(x));
printme = @(x) print(x,'-dpdf');
%% 
%load TestData
subj = 1;
Fs = 100;
load(['../DREAMS/Subject' num2str(subj) '_Fs' num2str(Fs) '.mat'])
%load(['../DREAMS/Subject' num2str(subj) '.mat']);
y = y;
fs = Fs;
N = length(y);
n = 0:N-1;
%% Plot STFT for small and large windows 

% Small window
R = 64/2;         % R : frame length
Nfft = 64/2;      % Nfft : FFT length in STFT (Nfft >= R)
[A1, A1H, normA1] = MakeTransforms('STFT', N, [R 16 4 Nfft]);

% Large window
R2 = 256/2;
Nfft2 = 256/2;
[A2, A2H, normA2] = MakeTransforms('STFT', N, [R2 16 4 Nfft2]);

% Plot 
figure(1), clf
    
subplot(3, 1, 1)
plot(n/fs, y)
title('Data [y]')
AHy = A1H(y);
A2Hy = A2H(y);
tt = R/16 * ( (0:size(AHy, 2)) - 1 );    % tt : time axis for STFT

subplot(3, 1, 2)
imagesc(tt, [0 5], abs(AHy),[0 20])
axis xy
xlim([0 N])
ylim([0 0.5])
title('STFT of Input data (Small window)')
ylabel('Frequency')
xlabel('Time')
colorbar

subplot(3, 1, 3)
tt2 = R2/16 * ( (0:size(A2Hy, 2)) - 1 );    % tt : time axis for STFT
imagesc(tt2, [0 5], abs(A2Hy),[0 20])
axis xy
xlim([0 N])
ylim([0 0.5])
title('STFT of Input data (Large Window)')
ylabel('Frequency')
xlabel('Time')
colorbar

set(gcf,'Paperposition',[0 0 5 5]);
printme('Input.pdf')
%% Solve the dualBPD problem to separate transients and oscillations
%lam1 = 0.87;
%lam2 = 0.87;
lam1 = 0.06*rms(y);
lam2 = 0.06*rms(y);

[x1,x2,c1,c2,cost] = dualBPD(y,A1,A1H,A2,A2H,lam1,lam2,0.1,100);

x1 = x1';
x2 = x2';
% Display cost function history to observe convergence of algorithm.
figure(2), clf
plot(cost)
grid on
title('Cost function history')
xlabel('Iteration')

%% Display Separation of Transients and Oscillations

figure(3), clf

subplot(3, 2, 1.5)
plot(n/fs, y);
title('Input EEG')


subplot(3,2,3)
plot(n/fs, real(x1));
title('Transient component')

residual = y-real(x1+x2)';
subplot(3,2,5)
in = real(x2)+residual';
plot(n/fs,in);
title('Oscillatory component')

subplot(3, 2, 4)
imagesc(tt2, [0 5], dB(A1H(A1(c1))), [0 20])
axis xy
xlim([0 N])
ylim([0 0.5])
title(sprintf('STFT coefficients'));
ylabel('Frequency')
xlabel('Time')
colorbar


subplot(3, 2, 6)
imagesc(tt2, [0 5], abs(A2H(A2(c2))),[0 20])
axis xy
xlim([0 N])
ylim([0 0.5])
title(sprintf('STFT coefficients'));
ylabel('Frequency')
xlabel('Time')
colorbar

set(gcf,'Paperposition',[0 0 8 5]);
printme('Demo_Result.pdf')