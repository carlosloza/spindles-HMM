%%
clearvars
close all
clc

subjTest = 1;
robustMstepflag = true;

HMMSupervisedNoEM('Generalizedt', 5, robustMstepflag, subjTest)
% for i = 1:8
%     HMMSupervisedNoEM('Generalizedt', 5, true, 1)
% end