%%
clearvars
close all
clc

subj = 1;
Fs = 100;
load(['DREAMS/Subject' num2str(subj) '_Fs' num2str(Fs) '.mat'])

[CM, CMevent] = ConfusionMatrixSpindles(y, Fs, labels2, labels1)
CM(1,1)/sum(CM(1,:))
CM(2,1)/sum(CM(2,:))
CM(2,1)/sum(CM(1,:))
CMevent(1,1)/sum(CMevent(1,:))
CMevent(2,1)/sum(CMevent(2,:))
CMevent(2,1)/sum(CMevent(1,:))

[CM, CMevent] = ConfusionMatrixSpindles(y, Fs, labels1, labels2)
CM(1,1)/sum(CM(1,:))
CM(2,1)/sum(CM(2,:))
CM(2,1)/sum(CM(1,:))
CMevent(1,1)/sum(CMevent(1,:))
CMevent(2,1)/sum(CMevent(2,:))
CMevent(2,1)/sum(CMevent(1,:))