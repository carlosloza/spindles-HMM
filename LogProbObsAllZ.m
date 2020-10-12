%% Probabilities of observations for all possible hidden state cases
function logpYZ = LogProbObsAllZ(y, K, HMModel)
iIni = HMModel.ARorder + 1;
logpYZ = -inf(K, numel(y));
for k = 1:K
    muk = (HMModel.DelayARMatrix * ...
        HMModel.ObsParameters.meanParameters(k).Coefficients)';
    sigk = HMModel.ObsParameters.sigma(k);
    nuk = HMModel.ObsParameters.nu(k);
    logpYZ(k, iIni:end) = log(pdf('tLocationScale', y(iIni:end) - muk, 0, sigk, nuk));
end
end