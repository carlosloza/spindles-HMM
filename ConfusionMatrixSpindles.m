function [CM, CMevent] = ConfusionMatrixSpindles(labelsGT, labelsPred, Fs)
% always assume labels = 1 means no spindle and labels = 2 means spindle

%% Confussion matrix based on time samples
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
%% Confussion matrix based on events
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
TNevent = length(labelsGT)/Fs - FPevent - TPevent - FNevent;
CMevent(2,1) = FPevent;
CMevent(2,2) = TNevent;
end