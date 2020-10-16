function [loglike, alphaEM, auxEM] = HMMLikelihood(y, HMModel, varargin)
%HMMLIKELIHOOD  Log-likelihood calculation (LLC) under 
% robust autoregressive hidden semi-Markov model (RARHSMM)
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
%                   Example: 'normalize', false
%   returnAlpha :   logical value, default false
%                   Flag to return E-step alpha estimators
%                   Example: 'returnAlpha', true
%   Returns
%   -------
%   loglike :       float
%                   Log-likelihood of y under RARHSMM
%   alphaEM :       cell, size (1, N), each element of cell is a matrix of
%                   size (K, dmax) where K is the number of modes/regimes
%                   and dmax is the maximum duration of such modes (D in paper)
%   auxEM :         structure
%                   Auxiliary conditional loglikelihoods under the RARHSMM
%                   Fields: 
%                   - logpYZ: conditional log-likelihoods of observations
%                   given state labels
%                   - logpDZ, marginal log-likelihoods of durations
%
%Example: loglike = HMMLikelihood(y, HMModel)
%Author: Carlos Loza (carlos.loza@utexas.edu)
%https://github.com/carlosloza/spindles-HMM

%% General parameters and settings
K = HMModel.StateParameters.K;          % number of regimes/modes
N = size(y, 2);                         % number of time samples in y  
iIni = HMModel.ARorder + 1;             % initial time sample for LLC
dmax = HMModel.DurationParameters.dmax; % maximum duration of regimes (D in paper)
% For compatibility
if ~isfield(HMModel.DurationParameters, 'flag')
    HMModel.DurationParameters.flag = 0;
end
% Defaults
normflag = true;
alphaflag = false;
% Check inputs
for i = 1:length(varargin)
    if strcmpi(varargin{i}, 'normalize')
        normflag = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'returnAlpha')
        alphaflag = varargin{i + 1};
    end
end
if normflag
    y = zscore(y);                      % Normalize observations
end
% For compatibility
if ~isfield(HMModel, 'DelayARMatrix')
    p = HMModel.ARorder;                % Autoregressive order
    % Build auxiliary matrix of AR predictors (embedding matrix)
    Yp = zeros(N - p, p);
    for i = 1:N-p
        Yp(i, :) = -fliplr(y(i : i + p - 1));
    end
    HMModel.DelayARMatrix = Yp;
end
% Conditional and marginal log-likelihoods to be used later
logpYZ = LogProbObsGivenZ(y, HMModel);
logpDZ = log(HMModel.DurationParameters.PNonParametric);
%% Forward sum-product algorithm - eq (7) in paper
% Forward pass of message passing (sum-product algorithm)
% Two implementations are needed: 
% - The first one (alphaflag = true) saves all alpha variables in a cell.
% These probabilities are then used in E-step of EM
% - The second one (alphaflag = false) does not save the alpha variables,
% it rather updates a local alpha that ends up becoming the joint
% probability of all the observations AND the final hidden states, z_N, d_N
nforw = iIni+1:N;
auxEM.logpYZ = logpYZ;
auxEM.logpDZ = logpDZ;
logpzIni = log(HMModel.StateParameters.pi);
Atrans = HMModel.StateParameters.A(:, :, 1);
logAtrans = log(Atrans);
if alphaflag
    alphaEM = cell(1, N);    % inside each cell element: states x durations
    alphaEM(:) = {-inf(K, dmax, 'double')};     % initialization
    % First iteration
    alphaEM{iIni} = logpYZ(:, iIni) + logpDZ + logpzIni;
    logalphaEMprev = alphaEM{iIni};
    % Remaining iterations
    for i = 1:numel(nforw)
        logalphaauxdmin = logsumexp(logalphaEMprev(:, 1) + logAtrans, 1)';
        logalphaEMpos = -inf(K, dmax);
        logalphaEMpos(:, dmax) = logpYZ(:, nforw(i)) + logpDZ(:, dmax) + logalphaauxdmin;
        logalphaEMpos(:, 1:dmax-1) = logpYZ(:, nforw(i)) + ...
            logsumexp(cat(3, logpDZ(:,1:dmax-1) + logalphaauxdmin, logalphaEMprev(:, 2:dmax)), 3);
        logalphaEMprev = logalphaEMpos;
        alphaEM{nforw(i)} = logalphaEMpos;      % allocate variable in cell
    end
else
    alphaEM = [];
    % First iteration
    logalphaEMprev = logpYZ(:, iIni) + logpDZ + logpzIni;
    % Remaining iterations
    for i = 1:numel(nforw)
        logalphaauxdmin = logsumexp(logalphaEMprev(:, 1) + logAtrans, 1)';
        logalphaEMpos = -inf(K, dmax);
        logalphaEMpos(:, dmax) = logpYZ(:, nforw(i)) + logpDZ(:, dmax) + logalphaauxdmin;
        logalphaEMpos(:, 1:dmax-1) = logpYZ(:, nforw(i)) + ...
            logsumexp(cat(3, logpDZ(:,1:dmax-1) + logalphaauxdmin, logalphaEMprev(:, 2:dmax)), 3);
        logalphaEMprev = logalphaEMpos;         % update variable (matrix)
    end
end
% log-likelihood: log(p(y|theta))
loglike = logsumexp(logsumexp(logalphaEMpos, 2), 1);
end