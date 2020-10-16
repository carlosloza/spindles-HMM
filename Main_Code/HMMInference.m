function [z, loglike, drem] = HMMInference(y, HMModel, varargin)
%HMMINFERENCE Estimation of most likely sequence of hidden labels (states/regimes/modes)
% under robust autoregressive hidden semi-Markov model (RARHSMM)
%
%   Parameters
%   ----------
%   y :             row vector, size (1, N)
%                   Univariate time series (sequence)
%   HMModel :       structure
%                   Fully learned RARHSMM (either after supervised or unsupervised
%                   scheme)
%   normalize :     logical value, default true
%                   Flag to zscore input sequence
%                   example: 'normalize', false
%
%   Returns
%   -------
%   z :             row vector, size (1, N)
%                   Most likely sequence of labels
%   loglike :       float
%                   Log joint probability of hidden labels and observations
%   drem :          row vector, size (1, N)
%                   Remaining duration (in samples) of hidden labels
%                   Mainly used for debugging
%
%Example: z = HMMInference(y, HMModel)
%Author: Carlos Loza (carlos.loza@utexas.edu)
%https://github.com/carlosloza/spindles-HMM

%% General parameters and settings
K = HMModel.StateParameters.K;          % number of regimes/modes
N = size(y, 2);                         % number of time samples in y 
iIni = HMModel.ARorder + 1;             % initial time sample for estimation
dmax = HMModel.DurationParameters.dmax; % maximum duration of regimes (D in paper)
pmin = 2e-300;                          % To avoid underflow
% For compatibility
if ~isfield(HMModel.DurationParameters, 'flag')
    HMModel.DurationParameters.flag = 0;
end
% Defaults
normflag = true;
% Check inputs
for i = 1:length(varargin)
    if strcmpi(varargin{i}, 'normalize')
        normflag = varargin{i + 1};
    end
end
if normflag
    y = zscore(y);                      % Normalize observations
end
p = HMModel.ARorder;                    % Autoregressive order
% Build auxiliary matrix of AR predictors (embedding matrix)
Yp = zeros(N - p, p);
for i = 1:N-p
    Yp(i, :) = -fliplr(y(i : i + p - 1));
end
HMModel.DelayARMatrix = Yp;
% Conditional and marginal log-likelihoods to be used later
logpYZ = LogProbObsGivenZ(y, HMModel);
logpDZ = log(HMModel.DurationParameters.PNonParametric);
% Work in log domain
pzIni = double(HMModel.StateParameters.pi);
pzIni(pzIni < pmin) = pmin;
logpzIni = log(pzIni);
Atrans = HMModel.StateParameters.A(:, :, 1);
Atrans(Atrans < pmin) = pmin;
logAtrans = log(Atrans);
%% Viterbi decoding (extended to handle bivariate hidden states)
psi_d = cell(1, N);
psi_z = cell(1, N);
% Forward max-product algorithm
% First iteration
deltprev = logpYZ(:, iIni) + logpDZ + logpzIni;
idx = zeros(K, dmax - 1);
idx(:, 1) = (K*dmax+1:K*dmax+K)';
% Useful for indexing inside main loop
for i = 2:dmax-1
    idx(:, i) = (dmax + 1)*K + idx(1, i - 1):...
        (dmax + 1)*K + idx(1, i - 1) + K - 1;
end
% Rest of iterations
auxidx_z_n1 = repmat((1:K)', 1, dmax-1);
aux1 = repmat(2:dmax, K, 1);
for i = iIni + 1:N
    % maximum over z_n1 (previous z)
    [mx1, auxidx] = max(deltprev(:, 1) + logAtrans, [], 1);  % K x 1
    max_z_n1 = [mx1' deltprev(:, 2:dmax)];
    psi_z{i} = uint16([auxidx' auxidx_z_n1]);       % cast to save memory
    % maximum over d_n1 (previous duration)
    [max_d_n1, idxtemp] = max(cat(3, logpDZ + max_z_n1(:, 1),...
        [max_z_n1(:, 2:dmax) -inf(K, 1)]), [], 3);
    idxdmax = idxtemp(:,dmax);
    idxtemp = idxtemp(:, 1:dmax-1);
    idxtemp(idxtemp == 1) = nan;
    idxtemp(idxtemp == 2) = 0;
    idxtemp = idxtemp + aux1;
    idxtemp(isnan(idxtemp)) = 1;
    psi_d{i} = uint16([idxtemp idxdmax]);           % cast to save memory
    deltprev = logpYZ(:, i) + max_d_n1;
end
[loglike, idxloglike1] = max(deltprev(:));
z = zeros(1, N);
drem = zeros(1, N);
[z(end), drem(end)] = ind2sub([K dmax], idxloglike1);
% Backtracking
for i = N-1:-1:iIni
    drem(i) = psi_d{i+1}(z(i+1), drem(i+1));
    z(i) = psi_z{i+1}(z(i+1), drem(i));
end
end