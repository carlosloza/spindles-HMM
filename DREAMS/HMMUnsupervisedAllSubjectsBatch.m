clearvars
close all
clc

% poolObj = gcp('nocreate');
% if isempty(poolObj)
%     parpool(12)
% end

ObsModel = 'Generalizedt';
p = 5;
robustMstepflag = true;
dmax_v = [2 5 10 30 60];
    
for i = 1:numel(dmax_v)
    dmax = dmax_v(i);
    fprintf('Model %s, p = %d, dmax = %d \n', ObsModel, p, dmax)
    HMMUnsupervisedAllSubjects(ObsModel, p, robustMstepflag, dmax)
end