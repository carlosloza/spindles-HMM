function logpYZ = LogProbObsGivenZ(y, HMModel)
%LOGPROBOBSGIVENZ  Conditional log-probabilities of observations given z 
% for robust autoregressive hidden semi-Markov model (RARHSMM)
%   
%   Parameters
%   ----------
%   y :             row vector, size (1, N)
%                   Univariate time series (sequence)
%   HMModel :       structure
%                   Fully learned RARHSMM (either after supervised or unsupervised
%                   scheme) or partially learned RARHSMM (going through EM)
%
%   Returns
%   -------
%   logpYZ :        matrix, size (K, N)
%                   Conditional log-probabilities of observations given z
%
%Example: z = HMMInference(y, HMModel)
%Note: Requires Statistics and Machine Learning Toolbox 
%Author: Carlos Loza (carlos.loza@utexas.edu)
%https://github.com/carlosloza/spindles-HMM

%% 
K = HMModel.StateParameters.K;          % number of regimes/modes
iIni = HMModel.ARorder + 1;             % initial time sample for calculations
logpYZ = -inf(K, numel(y));
for k = 1:K
    muk = (HMModel.DelayARMatrix * ...
        HMModel.ObsParameters.meanParameters(k).Coefficients)';     % linear generalized t mean
    sigk = HMModel.ObsParameters.sigma(k);                          % scale parameters
    nuk = HMModel.ObsParameters.nu(k);                              % degrees of freedom
    logpYZ(k, iIni:end) = log(pdf('tLocationScale', y(iIni:end) - muk, 0, sigk, nuk));  % observation conditional likelihoods/probabilities
end
end