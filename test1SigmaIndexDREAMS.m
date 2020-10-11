clearvars 
close all
clc

subj = 1;
load(['DREAMS/Subject' num2str(subj) '.mat']);
y = X;

%%
SigIndex = SigmaIndex(y, Fs)

scorer = 1;
if scorer == 1
    v_sc = v_sc1;
elseif scorer == 2
     v_sc = v_sc2;
end


% ndown = 1;
% y = downsample(y, ndown);
% Fs = Fs/ndown;

load(['DREAMS/Filters/Spindles_' num2str(Fs) 'Hz'])
%y = filtfilt(h_b, h_a, y);

%y = zscore(y);