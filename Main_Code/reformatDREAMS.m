% Script to reformat DREAMS sleep spindles dataset to .mat format with
% sampling frequency of 50 Hz
%
% IMPORTANT: To run this script, it is necessary to download the DREAMS
% sleep spindles dataset first. A reliable source is https://zenodo.org/record/2650142
% There are several .rar files so make sure to download the right one:
% DatabaseSpindles.rar
% After downloading, unpack the .rar file into a folder (most likely named
% "DatabaseSpindles").
% After unpacking, run this script.
% The resulting labels/scores will be coded as follows:
% 1: non-spindle regime
% 2: sleep spindle
%
% NOTE: The script extracts the EEG from the .txt files (not from the .EDF
% files) in order to avoid extra edfreader related functions
% Side note: the database used to be hosted at 
% http://www.tcts.fpms.ac.be/~devuyst/Databases/DatabaseSpindles/ but that
% website has been down for quite some time now

clearvars
close all
clc

newFs = 50;                             % new sampling frequency
LenRecordSec = 30*60;                   % 30-minute-long recordings
nSub = 8;                               % number of subjects in database
Y = zeros(nSub, LenRecordSec*newFs);
% GUI to point to the DatabaseSpindles spindles folder
% This can be replaced by a variable with the location of the folder
DREAMSpath = uigetdir(pwd, ['Select DREAMS sleep spindles dataset folder '...
    '(most likely "DatabaseSpindles" from https://zenodo.org/record/2650142)']);
if DREAMSpath == 0
    error('Canceled by user')
else
    try 
       aux = readtable([DREAMSpath '\excerpt1.txt']);
    catch ME
        if strcmpi(ME.identifier, 'MATLAB:readtable:OpenFailed')
            error('Required files not found in selected folder')
        end
    end
    fprintf('Required files found in selected folder! \n')
end
% Start reformating
ExpertLabels(nSub) = struct();
fprintf('Reformatting data... \n')
for sub_i = 1:nSub
    % EEG
    x = table2array(readtable([DREAMSpath '\excerpt' num2str(sub_i) '.txt']));
    y = x(~isnan(x))';
    if sub_i == 6
        y = y(1:360000);                    % special case
    end
    Fs = length(y)/LenRecordSec;
    % Resampling (actually downsampling)
    switch Fs
        case 50
            Y(sub_i, :) = y;
        case 100
            Y(sub_i, :) = downsample(y, 2);
        case 200
            Y(sub_i, :) = downsample(y, 4);
    end
    % Expert visual scores
    if sub_i <= 6
        % 2 visual scores
        aux = zeros(2, LenRecordSec*newFs);
        for i = 1:2           
            T = table2array(readtable([DREAMSpath '\Visual_scoring'...
                num2str(i) '_excerpt' num2str(sub_i) '.txt']));
            for j = 1:size(T, 1)
                aux(i, round(newFs*T(j, 1)):round(newFs*T(j, 1)) + round(newFs*T(j, 2))) = 1;
            end  
            ExpertLabels(sub_i).Expert(i).VisualScores = 1 + aux(i, :);        
        end
        labels = 1 + sum(aux, 1);
        labels(labels > 1) = 2;
        ExpertLabels(sub_i).VisualScoresUnion = labels;   
    else
        % 1 visual score
        aux = zeros(1, LenRecordSec*newFs);
        T = table2array(readtable([DREAMSpath '\Visual_scoring1'...
            '_excerpt' num2str(sub_i) '.txt']));
        for j = 1:size(T, 1)
            aux(1, round(newFs*T(j, 1)):round(newFs*T(j, 1)) + round(newFs*T(j, 2))) = 1;
        end
        labels = 1 + aux;
        ExpertLabels(sub_i).Expert(1).VisualScores = labels;   
        ExpertLabels(sub_i).VisualScoresUnion = labels; 
    end
end
fprintf('Done! \n')
Fs = newFs;
% Save as a .mat file in the same folder
save([DREAMSpath '\DREAMS_SleepSpindles.mat'], 'Y', 'ExpertLabels', 'Fs')