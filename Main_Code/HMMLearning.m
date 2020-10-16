function HMModel = HMMLearning(ySeq, K, varargin)
%HMMLEARNING Learning/estimation of robust autoregressive hidden semi-Markov model (RARHSMM)
% parameters given input sequence(s)
%
%   Parameters
%   ----------
%   ySeq :                  single row vector, size (1, N) or cell, size (1, NSeq),
%                           Single univariate time series (sequence) or batch
%                           of univariate sequences (not necessarily of the same
%                           length). In bath case, sequences must be row
%                           vectors
%   K :                     unsigned integer
%                           Number of states/labels/regimes/modes in the
%                           sequences according to the model
%                           Example: 2
%   ARorder :               unsigned integer
%                           Autoregressive order of linear generalized t model
%                           on the mean of the observations
%                           Example: 'ARorder', 10
%   StateParameters :       structure
%                           Fields:
%                           - pi: probabilities of initial label
%                           - A: label transition matrix
%   DurationParameters :    structure
%                           Fields:
%                           - dmin: minimum duration (in samples) or regimes 
%                           (must be larger than ARorder)
%                           - dmax: maximum duration (in samples) of regimes
%   normalize :             logical value, default true
%                           Flag to zscore input sequence
%                           Example: 'normalize', false
%   robustIni :             logical value, default true
%                           Flag to use robust linear regression (M-estimators)
%                           for initial estimate of AR coefficients
%                           Example: 'robustIni', false
%   Fs :                    float, default 50
%                           Sampling frequency in Hz
%                           Example: 'Fs', 100
%   parallel :              logical value, default true
%                           Flag to use MATLAB parallel computing toolbox
%                           Example, 'parallel', false
%
%Example: loglike = HMMLearning(y, 2)
%Author: Carlos Loza (carlos.loza@utexas.edu)
%https://github.com/carlosloza/spindles-HMM

%% General parameters and settings
HMModel.StateParameters.K = K;              % number of regimes/modes
% Defaults
HMModel.normalize = 1;
HMModel.ARorder = 5;
HMModel.robustIni = true;
HMModel.Fs = 50;
% Check inputs and initial conditions
% NOTE: Provided parameters must have the same number of regimes as the K
% parameter. No formal check/validation is performed
for i = 1:length(varargin)
    if strcmpi(varargin{i}, 'ARorder')
        HMModel.ARorder = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'StateParameters') 
        HMModel.StateParameters = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'ObsParameters') 
        HMModel.ObsParameters = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'DurationParameters')
        HMModel.DurationParameters = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'normalize')
        HMModel.normalize = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'robustIni')
        HMModel.robustIni = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'Fs')
        HMModel.Fs = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'parallel')
        parflag = varargin{i + 1};
    end
end
% Single sequence as input case - create cell with one element
if ~iscell(ySeq)
    aux = ySeq;
    clear ySeq
    ySeq{1} = aux;
end
% Initialize loglikelihood and EM-related fields
HMModel.loglike = [];
HMModel.loglikeNorm = [];
HMModel.nSeq = numel(ySeq);         % Number of contionally iid input sequences given the model parameters
HMModel.EM.gamma = [];
HMModel.EM.eta_iIni = [];
HMModel.EM.xi = [];
HMModel.EM.sumxi = [];
HMModel.EM.Etau = [];
% Format input
HMModel.N = zeros(1, HMModel.nSeq);
for i = 1:HMModel.nSeq
    y = ySeq{i};
    HMModel.N(i) = size(y, 2);
    % Normalize each  training observation sequence 
    if HMModel.normalize
        y = zscore(y);
    end
    ySeq{i} = y;
end
% EM algorithm parameters
maxIt = 20;                             % Maximum number of EM iterations
convTh = 0.05;                          % Convergence threshold
fl = 1;                                 % Flag to keep running EM
It = 0;                                 % Number of iterations
% Cells to arrage fields at the end of the algorithm
sortCell = {'StateParameters', 'DurationParameters', 'ObsParameters',...
    'ARorder', 'loglike', 'loglikeNorm', 'robustIni', 'normalize', 'Fs'};
% Build auxiliary matrix of AR predictors (embedding matrix)
p = HMModel.ARorder;                    % Autoregressive order
nSeq = HMModel.nSeq;
for iSeq = 1:nSeq
    y = ySeq{iSeq};
    N = HMModel.N(iSeq);
    Ypaux = zeros(N - p, p);
    for i = 1:N - p
        Ypaux(i, :) = -fliplr(y(i : i + p - 1));
    end
    HMModel.DelayARMatrix(iSeq).Seq = Ypaux;
end
%% Initial conditions - tuned for sleep spindle modeling for Fs=50 Hz
HMModel = InitialConditionsSleepSpindles(ySeq, HMModel);
%% Expectation-Maximization algorithm
while fl
    % E STEP
    HMModel = ForwardBackward(ySeq, HMModel, parflag);
    HMModel = GeneralizedtEstep(ySeq, HMModel);
    % M STEP
    HMModel = MaxEM(ySeq, HMModel);  
    It = It + 1;
    % Check for convergence
    if It > 1
        if abs(HMModel.loglikeNorm(It) - HMModel.loglikeNorm(It-1))/abs(HMModel.loglikeNorm(It - 1)) <= convTh          
            HMModel = cleanFields(HMModel, sortCell);    
            break
        end
    end
    if It == maxIt
        HMModel = cleanFields(HMModel, sortCell); 
        warning('Solution did not converge after %d iterations', maxIt)
        break
    end        
end
end
%% Forward-backward algorithm 
% Logsumexp implementation for Explicit Duration Hidden Semi-Markov Model
function HMModel = ForwardBackward(ySeq, HMModel, parflag)
% default
if nargin == 2
    parflag = true;
end
% clear variables from previous EM iteration
HMModel.EM = rmfield(HMModel.EM, {'gamma';'eta_iIni';'xi';'sumxi'});
nSeq = HMModel.nSeq;                    % number of iid input sequences
K = HMModel.StateParameters.K;          % number of regimes/modes
iIni = HMModel.ARorder + 1;             % initial time sample for alpha-beta algorithm
dmax = HMModel.DurationParameters.dmax; % maximum duration (in samples) of regimes
% Work in log domain
Atrans = HMModel.StateParameters.A(:, :, 1);
logAtrans = log(Atrans);
logAtransT = logAtrans';
loglikeSeq = zeros(1, nSeq);            % log-likelihood per input sequence
if parflag
    % Parallel implementation - cells are necessary
    HMModelSeq = cell(1, nSeq);
    NSeq = HMModel.N;
    gammaEMSeq = cell(1, nSeq);
    eta_iIni_Seq = zeros(K, dmax, nSeq);
    xiEMSeq = cell(1, nSeq);
    sumxiEMSeq = zeros(K, K, nSeq);
    for i = 1:nSeq
        HMModelSeq{i} = HMModel;
        HMModelSeq{i}.DelayARMatrix = HMModelSeq{i}.DelayARMatrix(i).Seq;
    end
    % Main loop
    parfor iSeq = 1:nSeq
        N = NSeq(iSeq);
        [loglike, logalphaEM, auxEM] = HMMLikelihood(ySeq{iSeq}, HMModelSeq{iSeq}, 'normalize', false, 'returnAlpha', true);
        logpYZ = auxEM.logpYZ;
        logpDZ = auxEM.logpDZ;
        logalphaEMiIni = logalphaEM{iIni};
        % beta, eta, gamma, xi - Initializations
        logbetaprev = zeros(K, dmax);
        logbetaup = -inf(K, dmax);
        nback = N-1:-1:iIni;
        gammaEM = zeros(K, N, 'single');
        % auxiliary variable that accounts for transitions, i.e. d_{n,1} = 1
        logalphadmin = cell2mat(cellfun(@(x) x(:,1), logalphaEM, 'UniformOutput', false));
        xiEM = cell(1, N);
        xiEM(:) = {zeros(K, dmax, 'single')};
        sumxiEM = zeros(K, K);
        % beta, eta, gamma, xi - Last/first iteration
        aux = double(exp(logalphaEM{N} + logbetaprev - loglike));
        gammaEM(:, N) = sum(aux, 2);
        aux = exp((logalphadmin(:, N-1) + ...
            reshape((logpYZ(:, N) + logpDZ + logbetaprev), 1, K, dmax)...
            + logAtrans) - loglike);                % broadcasting here
        sumxiEM = sumxiEM + sum(aux, 3);            % auxiliary for easier/faster update of transition matrix
        xiEM{N} = squeeze(sum(aux, 1));             % marginalize z_{n-1} in advance to same memory
        logalphaEM = logalphaEM(1:end-1);           % release some memory
        % beta, eta, gamma, xi - remaining iterations
        for i = 1:numel(nback)
            % beta - eq (8) in paper
            logbetaaux = logsumexp(logbetaprev + logpDZ, 2);
            logbetaup(:, 1) = logsumexp(logpYZ(:, nback(i)+1) + logAtransT + logbetaaux, 1)';
            logbetaup(:, 2:dmax) = logpYZ(:, nback(i)+1) + logbetaprev(:, 1:dmax-1);
            % gamma - eq (10) in paper
            gammaEM(:, nback(i)) = sum(exp(logalphaEM{end} + logbetaup - loglike), 2);
            % xi - eq (11) in paper
            aux = zeros(K, K, dmax);
            if nback(i) > iIni
                aux(:, :, :) = exp((logalphadmin(:, nback(i)-1) + ...
                    reshape((logpYZ(:, nback(i)) + logpDZ + logbetaup), 1, K, dmax)...
                    + logAtrans) - loglike);        % broadcasting here
                sumxiEM = sumxiEM + sum(aux, 3);    % auxiliary for easier/faster update of transition matrix
                xiEM{nback(i)} = reshape(sum(aux, 1), K, dmax); % marginalize z_{n-1} in advance to same memory         
            end
            logbetaprev = logbetaup;                % update beta, unlike alphas in LLC, there is no need to save these betas after use
            logalphaEM = logalphaEM(1:nback(i)-1);
        end
        loglikeSeq(iSeq) = loglike;
        % eq (9) in paper
        eta_iIni = exp(logalphaEMiIni + logbetaup - loglike);   % the only eta that is needed for maximization step
        xiEM = cellfun(@single, xiEM, 'un', 0);     % cast to save memory
        % Save everything in parallel
        gammaEMSeq{iSeq} = gammaEM;
        eta_iIni_Seq(:, :, iSeq) = eta_iIni;
        xiEMSeq{iSeq} = xiEM;
        sumxiEMSeq(:, :, iSeq) = sumxiEM;
    end
    clear HMModelSeq
    % Update fields: Unpack everything from cells to structure
    for iSeq = 1:nSeq
        HMModel.EM(iSeq).gamma = gammaEMSeq{iSeq};
        HMModel.EM(iSeq).eta_iIni = eta_iIni_Seq(:, :, iSeq);
        HMModel.EM(iSeq).xi = xiEMSeq{iSeq};
        HMModel.EM(iSeq).sumxi = sumxiEMSeq(:, :, iSeq);
    end
    clear gammaEMSeq eta_iIni_Seq xiEMSeq sumxiEMSeq 
else
    % Non-parallel implementation - more straightforward but much more
    % demanding and slow
    % Main loop
    for iSeq = 1:nSeq
        N = HMModel.N(iSeq);
        HMModelSeq = HMModel;
        HMModelSeq.DelayARMatrix = HMModelSeq.DelayARMatrix(iSeq).Seq;
        [loglike, logalphaEM, auxEM] = HMMLikelihood(ySeq{iSeq}, HMModelSeq, 'normalize', false, 'returnAlpha', true);
        logpYZ = auxEM.logpYZ;
        logpDZ = auxEM.logpDZ;
        logalphaEMiIni = logalphaEM{iIni};
        % beta, eta, gamma, xi - Initializations
        logbetaprev = zeros(K, dmax);
        logbetaup = -inf(K, dmax);
        nback = N-1:-1:iIni;
        gammaEM = zeros(K, N, 'single');
        % auxiliary variable that accounts for transitions, i.e. d_{n,1} = 1
        logalphadmin = cell2mat(cellfun(@(x) x(:,1),  logalphaEM, 'UniformOutput', false));
        xiEM = cell(1, N);
        xiEM(:) = {zeros(K, dmax, 'single')};
        sumxiEM = zeros(K, K);
        % beta, eta, gamma, xi - Last/first iteration
        aux = double(exp(logalphaEM{N} + logbetaprev - loglike));
        gammaEM(:, N) = sum(aux, 2);
        aux = exp((logalphadmin(:, N-1) + ...
            reshape((logpYZ(:, N) + logpDZ + logbetaprev), 1, K, dmax)...
            + logAtrans) - loglike);                % broadcasting here
        sumxiEM = sumxiEM + sum(aux, 3);            % auxiliary for easier/faster update of transition matrix
        xiEM{N} = squeeze(sum(aux, 1));             % marginalize z_{n-1} in advance to same memory
        logalphaEM = logalphaEM(1:end-1);           % release some memory
        % beta, eta, gamma, xi - remaining iterations
        for i = 1:numel(nback)
            % beta - eq (8) in paper
            logbetaaux = logsumexp(logbetaprev + logpDZ, 2);
            logbetaup(:, 1) = logsumexp(logpYZ(:, nback(i)+1) + logAtransT + logbetaaux, 1)';
            logbetaup(:, 2:dmax) = logpYZ(:, nback(i)+1) + logbetaprev(:, 1:dmax-1);
            % gamma - eq (10) in paper
            gammaEM(:, nback(i)) = sum(exp(logalphaEM{end} + logbetaup - loglike), 2);
            % xi - eq (11) in paper
            aux = zeros(K, K, dmax);
            if nback(i) > iIni
                aux(:, :, :) = exp((logalphadmin(:, nback(i)-1) + ...
                    reshape((logpYZ(:, nback(i)) + logpDZ + logbetaup), 1, K, dmax)...
                    + logAtrans) - loglike);        % broadcasting here
                sumxiEM = sumxiEM + sum(aux, 3);    % auxiliary for easier/faster update of transition matrix
                xiEM{nback(i)} = reshape(sum(aux, 1), K, dmax); % marginalize z_{n-1} in advance to same memory
            end
            logbetaprev = logbetaup;                % update beta, unlike alphas in LLC, there is no need to save these betas after use
            logalphaEM = logalphaEM(1:nback(i)-1);
        end
        loglikeSeq(iSeq) = loglike;
        % eq (9) in paper
        eta_iIni = exp(logalphaEMiIni + logbetaup - loglike);   % the only eta that is needed for maximization step
        xiEM = cellfun(@single, xiEM, 'un', 0);     % cast to save memory
        % Update fields
        HMModel.EM(iSeq).gamma = gammaEM;
        HMModel.EM(iSeq).eta_iIni = eta_iIni;
        HMModel.EM(iSeq).xi = xiEM;
        HMModel.EM(iSeq).sumxi = sumxiEM;
        clear gammaEM xiEM logalphaEM
    end
end
HMModel.loglike = [HMModel.loglike sum(loglikeSeq)];    % add all log-likelihoods
HMModel.loglikeNorm = [HMModel.loglikeNorm sum(loglikeSeq./HMModel.N)]; % add all normalized log-likelihoods (key when input sequences have different lengths)
end
%% Expectations of latent variables from observation model
% eq (12) in paper
function HMModel = GeneralizedtEstep(ySeq, HMModel)
HMModel.EM = rmfield(HMModel.EM, {'Etau'});
nSeq = HMModel.nSeq;                % number of iid input sequences
K = HMModel.StateParameters.K;      % number of regimes/modes   
iIni = HMModel.ARorder + 1;         % initial time sample for estimation
for iSeq = 1:nSeq
    N = HMModel.N(iSeq);
    wnk = zeros(K, N);
    for k = 1:K
        nuk = HMModel.ObsParameters.nu(k);
        muk = (HMModel.DelayARMatrix(iSeq).Seq * ...
            HMModel.ObsParameters.meanParameters(k).Coefficients)';
        deltk = ((ySeq{iSeq}(iIni:end) - muk).^2)/...
            HMModel.ObsParameters.sigma(k)^2;
        wnk(k, iIni:end) = (nuk + 1)./(nuk + deltk);
    end
    HMModel.EM(iSeq).Etau = single(wnk);
end
end
%% Simple function to clean EM fields after convergence
function HMModel = cleanFields(HMModel, sortCell)
HMModel = rmfield(HMModel, {'N'; 'nSeq'; 'EM'});
aux = rmfield(HMModel.DurationParameters, 'flag');
HMModel = rmfield(HMModel, 'DurationParameters');
HMModel.DurationParameters = aux;
HMModel = rmfield(HMModel, 'DelayARMatrix');
HMModel = orderfields(HMModel, sortCell);
end