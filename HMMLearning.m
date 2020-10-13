function HMModel = HMMLearning(ySeq, K, varargin)
%% Batch implementation
% Defaults
HMModel.normalize = 1;
HMModel.Fs = 50;
HMModel.StateParameters.K = K;
HMModel.robustIni = false;
% Check inputs and initial conditions
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
    end
end
% Single sequence as input case
if ~iscell(ySeq)
    aux = ySeq;
    clear ySeq
    ySeq{1} = aux;
end
% Initialize loglikelihood and EM-related fields
HMModel.loglike = [];
HMModel.loglikeNorm = [];
HMModel.nSeq = numel(ySeq);
HMModel.EM.gamma = [];
HMModel.EM.eta_iIni = [];
HMModel.EM.xi = [];
HMModel.EM.sumxi = [];
HMModel.EM.Etau = [];
%% Format input and set EM convergence parameters
HMModel.N = zeros(1, HMModel.nSeq);
for i = 1:HMModel.nSeq
    y = ySeq{i};
    HMModel.N(i) = size(y, 2);
    % Normalize each  training observation sequence 
    % Indivisual z-scoring per dimension
    if HMModel.normalize
        y = zscore(y, [], 2);
    end
    ySeq{i} = y;
end
maxIt = 20;                             % Maximum number of EM iterations
convTh = 0.05;                          % Convergence threshold
fl = 1;
It = 0;
% Cells to arrage fields at the end of the algorithm
sortCell = {'StateParameters', 'DurationParameters', 'ObsParameters',...
    'ARorder', 'loglike', 'loglikeNorm', 'robustIni', 'normalize', 'Fs'};
%% Build auxiliary matrix of AR predictors
p = HMModel.ARorder;
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

%% Initial conditions
% Check if ALL state parameters are provided and at least the type of
% observation model is provided too. 
% This is NOT an exhastive search (it depends on the parametrization
% of the observations and durations for HSMM), so the user must be sure to provide
% all necessary initial conditions. If not, bad things will happen...

HMModel = InitialConditionsSleepSpindles(ySeq, HMModel);

%% Expectation-Maximization
while fl
    % E STEP
    HMModel = ForwardBackwardLogSumExpHSMMED(ySeq, HMModel);
    if strcmpi(HMModel.ObsParameters.model, 'Generalizedt')
        % Extra E-step for additional hidden variables of Generalized t
        % distribution for emissions
        HMModel = GeneralizedtEstep(ySeq, HMModel);
    end
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

%% Forward-backward algorithm - Logsumexp - Hidden Semi-Markov Model Explicit Duration
function HMModel = ForwardBackwardLogSumExpHSMMED(ySeq, HMModel)
% clear previous EM values
HMModel.EM = rmfield(HMModel.EM, {'gamma';'eta_iIni';'xi';'sumxi'});
nSeq = HMModel.nSeq;
K = HMModel.StateParameters.K;
iIni = HMModel.ARorder + 1;
dmax = HMModel.DurationParameters.dmax;
% Compute probabilities of observations for all possible hidden state cases
Atrans = HMModel.StateParameters.A(:, :, 1);
Astay = HMModel.StateParameters.A(:, :, 2);
logAtrans = log(Atrans);
logAtransT = logAtrans';
logAstay = log(Astay);
logAstayT = logAstay';
% Log-likelihood and auxiliary EM variables
loglikeSeq = zeros(1, nSeq);

%parfor options
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
parfor iSeq = 1:nSeq
    %iSeq
    %N = HMModel.N(iSeq);
    N = NSeq(iSeq);
%    HMModelSeq = HMModel;
%     if strcmpi(HMModel.type, 'ARHSMMED')
%         HMModelSeq.DelayARMatrix = HMModelSeq.DelayARMatrix(iSeq).Seq;
%     end
%    [loglike, logalphaEM, auxEM] = HMMLikelihood(ySeq{iSeq}, HMModelSeq, 'method', 'logsumexp', 'normalize', 0, 'returnAlpha', 1);
    [loglike, logalphaEM, auxEM] = HMMLikelihood(ySeq{iSeq}, HMModelSeq{iSeq}, 'normalize', 0, 'returnAlpha', 1);
    logpYZ = auxEM.logpYZ;
    logpDZ = auxEM.logpDZ;
    logalphaEMiIni = logalphaEM{iIni};
%    clear auxEM
    % beta, eta, gamma, xi - Initializations
    logbetaprev = zeros(K, dmax);   
    logbetaup = -inf(K, dmax);
    nback = N-1:-1:iIni;
    %etaEM = cell(1, N);
    %etaEM(:) = {zeros(K, dmax, 'single')};
    gammaEM = zeros(K, N, 'single');
    logalphadmin = cell2mat(cellfun(@(x) x(:,1),  logalphaEM, 'UniformOutput', false));
    xiEM = cell(1, N);
    xiEM(:) = {zeros(K, dmax, 'single')};
    sumxiEM = zeros(K, K);
    % beta, eta, gamma, xi - Last/first iteration
    aux = double(exp(logalphaEM{N} + logbetaprev - loglike));
    gammaEM(:, N) = sum(aux, 2);
    aux = exp((logalphadmin(:, N-1) + ...
        reshape((logpYZ(:, N) + logpDZ + logbetaprev), 1, K, dmax)...
        + logAtrans) - loglike);
    sumxiEM = sumxiEM + sum(aux, 3);
    xiEM{N} = squeeze(sum(aux, 1));
    logalphaEM = logalphaEM(1:end-1);           % this was good
    % beta, eta, gamma, xi - remaining iterations
    for i = 1:numel(nback)
        % beta
        logbetaaux = logsumexp(logbetaprev + logpDZ, 2);
        logbetaup(:, 1) = logsumexp(logpYZ(:, nback(i)+1) + logAtransT + logbetaaux, 1)';
        logbetaup(:, 2:dmax) = logpYZ(:, nback(i)+1) + logbetaprev(:, 1:dmax-1);      
        % gamma
        % % gammaEM(:, nback(i)) = sum(exp(logalphaEM{nback(i)} + logbetaup - loglike), 2);
        gammaEM(:, nback(i)) = sum(exp(logalphaEM{end} + logbetaup - loglike), 2);
        %etaEM{nback(i)} = single(aux);
        % xi
        aux = zeros(K, K, dmax);
        if nback(i) > iIni
            %nback(i)
            aux(:, :, :) = exp((logalphadmin(:, nback(i)-1) + ...
                reshape((logpYZ(:, nback(i)) + logpDZ + logbetaup), 1, K, dmax)...
                + logAtrans) - loglike);
            sumxiEM = sumxiEM + sum(aux, 3);
            xiEM{nback(i)} = reshape(sum(aux, 1), K, dmax);         % maybe squeeze instead?
        end
        logbetaprev = logbetaup;
        logalphaEM = logalphaEM(1:nback(i)-1);   % this was good
        % % logalphaEM{nback(i)} = [];               % this was added
    end
    loglikeSeq(iSeq) = loglike;
    eta_iIni = exp(logalphaEMiIni + logbetaup - loglike);
    % Convert to single
    xiEM = cellfun(@single, xiEM, 'un', 0);
    % Update fields
%     HMModel.EM(iSeq).gamma = gammaEM;
%     HMModel.EM(iSeq).eta_iIni = eta_iIni;
%     HMModel.EM(iSeq).xi = xiEM;  
%     HMModel.EM(iSeq).sumxi = sumxiEM;
%    clear gammaEM xiEM logalphaEM

    gammaEMSeq{iSeq} = gammaEM;
    eta_iIni_Seq(:, :, iSeq) = eta_iIni;
    xiEMSeq{iSeq} = xiEM;
    sumxiEMSeq(:, :, iSeq) = sumxiEM;
end
% PARFOR
clear HMModelSeq
for iSeq = 1:nSeq
    HMModel.EM(iSeq).gamma = gammaEMSeq{iSeq};
    HMModel.EM(iSeq).eta_iIni = eta_iIni_Seq(:, :, iSeq);
    HMModel.EM(iSeq).xi = xiEMSeq{iSeq};
    HMModel.EM(iSeq).sumxi = sumxiEMSeq(:, :, iSeq);
end
clear gammaEMSeq eta_iIni_Seq xiEMSeq sumxiEMSeq

HMModel.loglike = [HMModel.loglike sum(loglikeSeq)];
HMModel.loglikeNorm = [HMModel.loglikeNorm sum(loglikeSeq./HMModel.N)];
end
%%
function HMModel = cleanFields(HMModel, sortCell)
HMModel = rmfield(HMModel, {'N'; 'nSeq'; 'EM'});
aux = rmfield(HMModel.DurationParameters, 'flag');
HMModel = rmfield(HMModel, 'DurationParameters');
HMModel.DurationParameters = aux;
HMModel = rmfield(HMModel, 'DelayARMatrix');
HMModel = orderfields(HMModel, sortCell);
end