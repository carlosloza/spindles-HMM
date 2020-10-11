clearvars
close all
clc

% addpath('../../HMM')
% poolObj = gcp('nocreate');
% if isempty(poolObj)
%     parpool(12)
% end

%ObsModel_v{1} = 'Gaussian';
ObsModel = 'Generalizedt';
p = 5;
robustMstepflag = true;
dmax_v = [5 10 30];
    
for i = 1:numel(dmax_v)
    dmax = dmax_v(i);
    fprintf('Model %s, p = %d, dmax = %d \n', ObsModel, p, dmax)
    HMMSupervised(ObsModel, p, robustMstepflag, dmax);
end