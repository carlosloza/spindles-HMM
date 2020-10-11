%% Consolidate results from Inference with max dmax
clearvars
close all
clc

ObsModel = 'Gaussian';
p = 5;
robustFlag = true;

nSubTotal = 8;

PerfTest_All = zeros(3, nSubTotal);
PerfTestAlt_All = zeros(3, nSubTotal);
NLLTest_All = zeros(1, nSubTotal);
HMModelSupervised_All = cell(1, nSubTotal);

for i = 1:nSubTotal
    if robustFlag == true
        load([ObsModel '\AR order ' num2str(p) '\Robust_Subj' num2str(i) '.mat'])
    else
        load([ObsModel '\AR order ' num2str(p) '\NoRobust_Subj' num2str(i) '.mat'])
    end
    PerfTest_All(:, i) = PerfTest;
    PerfTestAlt_All(:, i) = PerfTestAlt;
    NLLTest_All(i) = NLLTest;
    HMModelSupervised_All{i} = HMModelSupervised;
end
PerfTest = PerfTest_All;
PerfTestAlt = PerfTestAlt_All;
NLLTest = NLLTest_All;
HMModelSupervised = HMModelSupervised_All;

if robustFlag == true
    save([ObsModel '\AR order ' num2str(p) '\Robust_dmax_MAX.mat'], ...
        'PerfTest','PerfTestAlt', 'NLLTest', 'HMModelSupervised')
else
    save([ObsModel '\AR order ' num2str(p) '\NoRobust_dmax_MAX.mat'], ...
        'PerfTest','PerfTestAlt', 'NLLTest', 'HMModelSupervised')
end