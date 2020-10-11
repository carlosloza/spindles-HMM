function PSDARyz(y, z, p, Fs, opt)
%% PSD of input (unlabeled/unsegmented)
K = numel(unique(z(z~=0)));
f = 0:0.5:round(Fs/2);          % frequency resolution

if nargin == 5
    figure
    subplot(K+1, 1, 1)
    y = zscore(y);
    [pxx, f] = pcov(y, p, f, Fs);
    plot(f, 10*log10(pxx));
    aux = ylim;
    
    %% PSD of segments, i.e. labeled data
    for k = 1:K
        subplot(K+1, 1, k+1)
        ylabel = y(z == k);
        ylabel = zscore(ylabel);
        [pxx, f] = pcov(ylabel, p, f, Fs);
        plot(f, 10*log10(pxx));
        ylim(aux);
    end
else
    figure
    y = zscore(y);
    [pxx, f] = pcov(y, p, f, Fs);
    plot(f, 10*log10(pxx));
    hold on
    c_cell{1} = 'k';
    c_cell{2} = 'r';
    for k = 1:K
        ylabel = y(z == k);
        ylabel = zscore(ylabel);
        [pxx, f] = pcov(ylabel, p, f, Fs);
        plot(f, 10*log10(pxx), c_cell{k});
    end
end
    
end