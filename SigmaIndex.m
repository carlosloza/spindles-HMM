function SigmaIdx = SigmaIndex(y, Fs)

idxSpindles = true(size(y));

% Segmentation in x-second snippets
LSp = 1*Fs;
tEp(:, 1) = 1:LSp:length(y);
tEp(:, 2) = tEp(:, 1) + LSp - 1;
tEp(end, 2) = length(y);
n = max([2^nextpow2(LSp), 512]);
w = hanning(LSp);
fsp = [10.5 16];
flow = [4 10];
fhigh = [20 40];
falpha = [7.5 10];
freq = 0:Fs/n:Fs/2;
SigmaIdx = zeros(1, size(tEp, 1));
for i = 1:size(tEp, 1)
    yaux = y(tEp(i, 1): tEp(i, 2));
    yaux = yaux - mean(yaux);
    fftc = fft(yaux.*w', n);
    ydft = fftc(1:n/2+1);
    
    Rsp = ydft(freq >= fsp(1) & freq <= fsp(2));
    Ssp = 2 * abs(Rsp)/sum(w);
    
    Rlow = ydft(freq >= flow(1) & freq <= flow(2));
    Slow = 2 * abs(Rlow)/sum(w);
    
    Rhigh = ydft(freq >= fhigh(1) & freq <= fhigh(2));
    Shigh = 2 * abs(Rhigh)/sum(w);
    
    %SigmaIdx(i) = 2*max(Ssp)/(mean(Slow) + mean(Shigh));
    SigmaIdx(i) = 2*max(Ssp)/(mean(Shigh));
    
    Ralpha = ydft(freq >= falpha(1) & freq <= falpha(2));
    Salpha = 2 * abs(Ralpha)/sum(w);
    
%     if max(Salpha) > max(Ssp)
%         SigmaIdx(i) = 0;
%     end    
end

a = 1;

end