function [PerfM, PerfMAlt] = PerfMeasures(labelsGT, labelsPred, Fs)
%PERFMEASURES Performance measures for sleep spindles detection
%
%   Parameters
%   ----------
%   labelsGT :          row vector, size (1, N)
%                       Univariate time series (sequence) with binary
%                       labels from experts. 1: non-spindle, 2: spindle
%   labelsPred :        row vector, size (1, N)
%                       Univariate time series (sequence) with binary
%                       labels from automatic sleep spindles detector
%                       1: non-spindle, 2: spindle
%   Fs :                float, default 50
%                       Sampling frequency in Hz
%                       Example: 'Fs', 50
%   Returns
%   -------
%   PerfM :             column vector, size (3, 1)
%                       By-sample performance measures: F1 score, MCC, and
%                       False positive proportion as defined in [1]
%                       "Automatic Sleep Spindles Detection - Overview and
%                       Development of a Standard Proposal Assessment Method"
%   PerfMAlt :      `   column vector, size (3, 1)
%                       By-event performance measures: true positive rate, 
%                       false positive rate, and false positive proportion 
%                       as defined in [1]
%
%Example: PerfTest = PerfMeasures(labels1, labels2, 50)
%Author: Carlos Loza (carlos.loza@utexas.edu)
%https://github.com/carlosloza/spindles-HMM

%% Main calculations
[CM, CMevent] = ConfusionMatrixSpindles(labelsGT, labelsPred, Fs);  % confusion matrix
% sample-based performance measures
TPR = CM(1,1)/sum(CM(1,:));                 % true positive rate
RE = TPR;                                   % recall
FPR = CM(2,1)/sum(CM(2,:));                 % false positive rate
PR = CM(1,1)/sum(CM(:,1));
FPProp = CM(2,1)/sum(CM(1,:));
F1 = 2*(RE*PR)/(RE + PR);
MCC = (CM(1,1)*CM(2,2)-CM(2,1)*CM(1,2))/...
    sqrt((CM(1,1)+CM(1,2))*(CM(1,1)+CM(2,1))*(CM(2,2)+CM(2,1))*(CM(2,2)+CM(1,2)));
% event-based performance measures (assume average duration of sleep spindles is 1 sec as in [1])
TPRalt = CMevent(1,1)/sum(CMevent(1,:));
FPRalt = CMevent(2,1)/sum(CMevent(2,:));
FPPropalt = CMevent(2,1)/sum(CMevent(1,:));
PerfM = [F1 MCC FPProp]';
PerfMAlt = [TPRalt FPRalt FPPropalt]';
end
%% Confusion matrix - sample and event based performance measures
function [CM, CMevent] = ConfusionMatrixSpindles(labelsGT, labelsPred, Fs)
% Confussion matrix based on time samples
CM = zeros(2, 2);
for i = 1:numel(labelsGT)
    if labelsGT(i) == 1 && labelsPred(i) == 1
        CM(2, 2) = CM(2, 2) + 1;
    elseif labelsGT(i) == 1 && labelsPred(i) == 2
        CM(2, 1) = CM(2, 1) + 1;
    elseif labelsGT(i) == 2 && labelsPred(i) == 1
        CM(1, 2) = CM(1, 2) + 1;
    elseif labelsGT(i) == 2 && labelsPred(i) == 2
        CM(1, 1) = CM(1, 1) + 1;
    end
end
% Confussion matrix based on events
CMevent = zeros(2, 2);
% True positie and false negative counts
TPevent = 0;
FNevent = 0;
aux = find(diff(labelsGT == 2) ~= 0);
aux = reshape(aux, 2, length(aux)/2);
for i = 1:size(aux, 2)
    idx1 = aux(1, i) + 1;
    idx2 = aux(2, i);
    if numel(find(labelsPred(idx1:idx2) == 2)) > 0
        TPevent = TPevent + 1;
    else
        FNevent = FNevent + 1;
    end
end
CMevent(1,1) = TPevent;
CMevent(1,2) = FNevent;
% False positive and approximate true negative counts
FPevent = 0;
aux = find(diff(labelsPred == 2) ~= 0);
if mod(length(aux),2) == 0
    aux = reshape(aux, 2, length(aux)/2);
else
    aux = reshape(aux(1:end-1), 2, (length(aux)-1)/2);
end
for i = 1:size(aux, 2)
    idx1 = aux(1, i) + 1;
    idx2 = aux(2, i);
    if numel(find(labelsGT(idx1:idx2) == 2)) == 0
        FPevent = FPevent + 1;
    end
end
% Assume average duration of sleep spindles is 1 sec as in [1]
TNevent = length(labelsGT)/Fs - FPevent - TPevent - FNevent;
CMevent(2,1) = FPevent;
CMevent(2,2) = TNevent;
end