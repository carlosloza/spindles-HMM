function [yFeat, zUp] = EEGFeatures(y, z, Fs, winLen, ovp, bEdges)

% [coeffs, delta, deltaDelta, loc] = mfcc(y', Fs,...
%     'WindowLength', winLen, 'OverlapLength', ovp, 'BandEdges', bEdges,...
%     'NumCoeffs', numel(bEdges) - 2, 'LogEnergy', 'Ignore');
% yFeat = [coeffs delta deltaDelta]';

[s, f, t, ps] = spectrogram(y, hann(winLen, 'periodic'), ovp, [0:1:Fs/2], Fs);
yFeat = zeros(numel(bEdges) - 1, numel(t));
for i = 1:numel(bEdges) - 1
    yFeat(i, :) = sum(ps(f >= bEdges(i) & f <= bEdges(i+1), :), 1);
end
yFeat = log10(yFeat);

% sigma index
S = 2*abs(s)./sum(hann(winLen, 'periodic'));
maxSp = max(S(f >= 10.5 & f <= 16, :), [], 1);
meanSp = mean(S(f >= 10.5 & f <= 16, :), 1);
meanLow = mean(S(f >= 4 & f <= 10, :), 1);
meanHigh = mean(S(f >= 20 & f <= 40, :), 1);
maxAlpha = max(S(f >= 7.5 & f <= 10, :), [], 1);
meanAlpha = mean(S(f >= 7.5 & f <= 10, :), 1);

SigmaIdx = 2*maxSp./(meanLow + meanHigh);

%yFeat = [yFeat; log10(SigmaIdx); log10(maxAlpha./maxSp)];

%yFeat = [yFeat; log10(SigmaIdx)];

%yFeat = yFeat./std(yFeat, [], 2);

yFeat = log(SigmaIdx);

loc = winLen:winLen-ovp:length(y);
zUp = ones(1, numel(loc));
for i = 1:numel(loc)
    aux = z(loc(i) - winLen + 1:loc(i));
    if sum(aux == 2) > round(winLen)/2
        zUp(i) = 2;
    end
end

end