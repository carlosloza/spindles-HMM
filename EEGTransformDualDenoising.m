%%
% Transform DREAMS slepindles dataset using dual Basis Pursuit Denoising
clearvars
close all
clc

Fs = 100;
% Small window
R = 64/2;         % R : frame length
Nfft = 64/2;      % Nfft : FFT length in STFT (Nfft >= R)
% Large window
R2 = 256/2;
Nfft2 = 256/2;

for i = 1:8
    load(['DREAMS/Subject' num2str(i) '_Fs' num2str(Fs) '.mat'])
    N = length(y);
    %TransStruct.y = y;
    %TransStruct.z = labels;
    [A1, A1H, normA1] = MakeTransforms('STFT', N, [R 16 4 Nfft]);
    [A2, A2H, normA2] = MakeTransforms('STFT', N, [R2 16 4 Nfft2]);
%     TransStruct(i).A1 = A1;
%     TransStruct(i).A1H = A1H;
%     TransStruct(i).A2 = A2;
%     TransStruct(i).A2H = A2H;
    lam1 = 0.06*rms(y);
    lam2 = 0.06*rms(y);
    [x1,x2,c1,c2,cost] = dualBPD(y,A1,A1H,A2,A2H,lam1,lam2,0.1,100);
    residual = y - real(x1+x2);
    y_osc = real(x2) + residual;
    %TransStruct.y_osc = y_osc;
    save(['DREAMS/Subject' num2str(i) '_Fs' num2str(Fs) '_Transformed.mat'],...
        'y', 'labels', 'y_osc')
end