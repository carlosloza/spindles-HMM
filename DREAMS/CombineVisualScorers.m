%%
clearvars
close all
clc

FsAll = 100;
subj = 2;

load(['DREAMS/Subject' num2str(subj) '.mat']);
y = X;
if Fs > FsAll
    ndown = Fs/FsAll;
    y = downsample(y, ndown);
elseif Fs < FsAll
    y = resample(y, FsAll, Fs);
end
if subj <= 6
    aux = zeros(2, numel(y));       % 2 scorers
    for i_sc = 1:2
        if i_sc == 1
            v_sc = v_sc1;
        elseif i_sc == 2
            v_sc = v_sc2;
        end
        for i = 1:size(v_sc, 1)
            aux(i_sc, round(FsAll*v_sc(i, 1)):round(FsAll*v_sc(i, 1)) + round(FsAll*v_sc(i, 2))) = 1;
        end
    end
    labels = 1 + sum(aux, 1);
    labels(labels > 1) = 2;
    labels1 = 1 + aux(1, :);
    labels2 = 1 + aux(2, :);
    Fs = FsAll;
    save(['DREAMS/Subject' num2str(subj) '_Fs' num2str(Fs) '.mat'], 'y', 'labels1', 'labels2', 'labels')
    figure
    subplot(3,1,1)
    plot(labels1)
    ylim([-0.5 2.5])
    subplot(3,1,2)
    plot(labels2)
    ylim([-0.5 2.5])
    subplot(3,1,3)
    plot(labels)
    ylim([-0.5 2.5])
else
    aux = zeros(1, numel(y));       % 1 scorer
    v_sc = v_sc1;
    for i = 1:size(v_sc, 1)
        aux(1, round(FsAll*v_sc(i, 1)):round(FsAll*v_sc(i, 1)) + round(FsAll*v_sc(i, 2))) = 1;
    end

    labels = 1 + aux;
    labels(labels > 1) = 2;
    labels1 = 1 + aux;
    Fs = FsAll;
    save(['DREAMS/Subject' num2str(subj) '_Fs' num2str(Fs) '.mat'], 'y', 'labels1', 'labels')
    figure
    subplot(2,1,1)
    plot(labels1)
    ylim([-0.5 2.5])
    subplot(2,1,2)
    plot(labels)
    ylim([-0.5 2.5])
end