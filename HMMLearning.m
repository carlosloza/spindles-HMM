function HMModel = HMMLearning(ySeq, K, varargin)
%% Batch implementation
% Defaults
HMModel.normalize = 1;
HMModel.StateParameters.K = K;
HMModel.robustMstep = false;
HMModel.SleepSpindles = false;
% Check inputs and initial conditions
for i = 1:length(varargin)
    if strcmpi(varargin{i}, 'type')
        HMModel.type = varargin{i + 1};      % options: HMM, ARHMM, ARHMM, ARHSMMED
    elseif strcmpi(varargin{i}, 'ARorder')
        HMModel.ARorder = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'StateParameters') 
        HMModel.StateParameters = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'ObsParameters') 
        HMModel.ObsParameters = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'DurationParameters')
        HMModel.DurationParameters = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'Estep')
        HMModel.Estep = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'normalize')
        HMModel.normalize = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'robustMstep')
        HMModel.robustMstep = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'SleepSpindles')
        HMModel.SleepSpindles = varargin{i + 1};
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
% Default E-step: scaling-based implementation
if ~isfield(HMModel, 'Estep')
    HMModel.Estep = 'scaling';
end
% Select expectation function and implementation
switch HMModel.Estep
    case 'logsumexp'
        if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'ARHMM') 
            funcEstep = @ForwardBackwardLogSumExpHMM;
        elseif strcmpi(HMModel.type, 'HSMMED') || strcmpi(HMModel.type, 'ARHSMMED')
            funcEstep = @ForwardBackwardLogSumExpHSMMED;
        elseif strcmpi(HMModel.type, 'HSMMVT') || strcmpi(HMModel.type, 'ARHSMMVT')
            funcEstep = @ForwardBackwardLogSumExpHSMMVT;
        end
    case 'scaling'
        if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'ARHMM')
            funcEstep = @ForwardBackwardScaleHMM;
        elseif strcmpi(HMModel.type, 'HSMMED') || strcmpi(HMModel.type, 'ARHSMMED')
            funcEstep = @ForwardBackwardScaleHSMMED;
        end
end
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
%convTh = 0.01;                          % Convergence threshold
convTh = 0.05;
fl = 1;
It = 0;
% Cells to arrage fields at the end of the algorithm
switch HMModel.type
    case 'HMM'
        sortCell = {'type', 'StateParameters', 'ObsParameters', 'loglike', ...
            'loglikeNorm', 'Estep', 'robustMstep', 'normalize'};
    case 'ARHMM'
        if HMModel.SleepSpindles
            sortCell = {'type', 'StateParameters', 'ObsParameters', 'ARorder', 'loglike', ...
                'loglikeNorm', 'Estep', 'robustMstep', 'normalize', 'SleepSpindles', 'Fs'};
        else
            sortCell = {'type', 'StateParameters', 'ObsParameters', 'ARorder', 'loglike', ...
                'loglikeNorm', 'Estep', 'robustMstep', 'normalize'};
        end     
    case 'HSMMED'
        sortCell = {'type', 'StateParameters', 'DurationParameters', 'ObsParameters',...
            'loglike', 'loglikeNorm', 'Estep', 'robustMstep', 'normalize'};
    case 'ARHSMMED'
        if HMModel.SleepSpindles
            sortCell = {'type', 'StateParameters', 'DurationParameters', 'ObsParameters',...
                'ARorder', 'loglike', 'loglikeNorm', 'Estep', 'robustMstep', 'normalize', 'SleepSpindles', 'Fs'};
        else
            sortCell = {'type', 'StateParameters', 'DurationParameters', 'ObsParameters',...
                'ARorder', 'loglike', 'loglikeNorm', 'Estep', 'robustMstep', 'normalize'};
        end
    case 'HSMMVT'
        sortCell = {'type', 'StateParameters', 'DurationParameters', 'ObsParameters',...
            'loglike', 'loglikeNorm', 'Estep', 'robustMstep', 'normalize'};
    case 'ARHSMMVT'
        if HMModel.SleepSpindles
            sortCell = {'type', 'StateParameters', 'DurationParameters', 'ObsParameters',...
                'ARorder', 'loglike', 'loglikeNorm', 'Estep', 'robustMstep', 'normalize', 'SleepSpindles', 'Fs'};
        else
            sortCell = {'type', 'StateParameters', 'DurationParameters', 'ObsParameters',...
                'ARorder', 'loglike', 'loglikeNorm', 'Estep', 'robustMstep', 'normalize'};
        end   
end
%% Build auxiliary matrix of AR predictors
if strcmpi(HMModel.type, 'ARHMM') || strcmpi(HMModel.type, 'ARHSMMED') || strcmpi(HMModel.type, 'ARHSMMVT')
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
end

%% Initial conditions
% Check if ALL state parameters are provided and at least the type of
% observation model is provided too. 
% This is NOT an exhastive search (it depends on the parametrization
% of the observations and durations for HSMM), so the user must be sure to provide
% all necessary initial conditions. If not, bad things will happen...
if HMModel.SleepSpindles
    HMModel = InitialConditionsSleepSpindles(ySeq, HMModel);
else
    if isfield(HMModel.StateParameters, 'pi') &&...
            isfield(HMModel.StateParameters, 'A') &&...
            isfield(HMModel.ObsParameters, 'model')
        fprintf('Initial conditions are provided \n')
    else
        HMModel = InitialConditions(ySeq, HMModel);
    end
end


%%
% f = exp((HMModel.DurationParameters.dmax/HMModel.Fs)*...
%     (HMModel.DurationParameters.dmin:HMModel.DurationParameters.dmax)/(HMModel.DurationParameters.dmax-HMModel.DurationParameters.dmin));
% f = [(0.01/(HMModel.DurationParameters.dmax - HMModel.DurationParameters.dmin))*ones(1, HMModel.DurationParameters.dmax - HMModel.DurationParameters.dmin)...
%     0.99];
% f = f/sum(f);
% HMModel.DurationParameters.PNonParametric(1, :) = [zeros(1, HMModel.DurationParameters.dmin-1) f];
% 
% f1 = [zeros(1, 24) normpdf(25:HMModel.DurationParameters.dmax, 50, 3)];
% f1 = f1/sum(f1);
% f = [f1 zeros(1, HMModel.DurationParameters.dmax - numel(f1))];
% HMModel.DurationParameters.PNonParametric(2, :) = f;


%% Expectation-Maximization
while fl
    % E STEP
    
    
    %f = ones(1, HMModel.DurationParameters.dmax - HMModel.DurationParameters.dmin + 1);
%     f = exp((HMModel.DurationParameters.dmax/HMModel.Fs)*...
%         (HMModel.DurationParameters.dmin:HMModel.DurationParameters.dmax)/(HMModel.DurationParameters.dmax-HMModel.DurationParameters.dmin));
%     f = f/sum(f);
%     HMModel.DurationParameters.PNonParametric(1, :) = [zeros(1, HMModel.DurationParameters.dmin-1) f];
%     
%     f1 = [zeros(1, 24) normpdf(25:HMModel.DurationParameters.dmax, 50, 5)];
%     f1 = f1/sum(f1);
%     f = [f1 zeros(1, HMModel.DurationParameters.dmax - numel(f1))];
%     HMModel.DurationParameters.PNonParametric(2, :) = f;
    
    
    HMModel = funcEstep(ySeq, HMModel);
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
%% Forward-backward algorithm - Logsumexp - Hidden Markov Model
function HMModel = ForwardBackwardLogSumExpHMM(ySeq, HMModel)
% clear previous EM values
HMModel.EM = rmfield(HMModel.EM, {'gamma';'xi'});
nSeq = HMModel.nSeq;
K = HMModel.StateParameters.K;
iIni = HMModel.ARorder + 1;
logA = log(HMModel.StateParameters.A);   
logAT = logA';
% Log-likelihood and auxiliary EM variables
loglikeSeq = zeros(1, nSeq);
for iSeq = 1:nSeq
    N = HMModel.N(iSeq);
    HMModelSeq = HMModel;
    if strcmpi(HMModel.type, 'ARHMM')
        HMModelSeq.DelayARMatrix = HMModelSeq.DelayARMatrix(iSeq).Seq;
    end
    [loglike, logalphaEM, auxEM] = HMMLikelihood(ySeq{iSeq}, HMModelSeq, 'method', 'logsumexp', 'normalize', 0);
    logpYZ = auxEM.logpYZ;
    % Backward iterations
    logbetaEM = zeros(K, N);
    nback = N-1:-1:iIni;
    % First iteration
    logbetaEM(:, end) = 0;
    % Remaining iterations
    for i = 1:numel(nback)
        logbetaEM(:, nback(i)) = logsumexp(...
            logbetaEM(:, nback(i)+1) + logpYZ(:, nback(i)+1) + logAT, 1)';
    end
    % Conditional and joint conditionals probabilities
    gammaEM = exp((logalphaEM + logbetaEM) - loglike);
    xiEM = -inf(K, K, N);                                                        
    xiaux = reshape(logpYZ(:, 2:end) + logbetaEM(:, 2:end), [1, K, N-1]) + logA;
    xiEM(:,:, 2:end) = reshape(logalphaEM(:, 1:end-1), [K, 1, N-1]) + xiaux;
    xiEM = exp(xiEM - loglike);
    loglikeSeq(iSeq) = loglike;
    % Update fields    
    HMModel.EM(iSeq).gamma = gammaEM;
    HMModel.EM(iSeq).xi = xiEM;  
end
HMModel.loglike = [HMModel.loglike sum(loglikeSeq)];
HMModel.loglikeNorm = [HMModel.loglikeNorm sum(loglikeSeq./HMModel.N)];
end
%% Forward-backward algorithm - Scaling - Hidden Markov Model
function HMModel = ForwardBackwardScaleHMM(ySeq, HMModel)
% clear previous EM values
HMModel.EM = rmfield(HMModel.EM, {'gamma';'xi'});
nSeq = HMModel.nSeq;
K = HMModel.StateParameters.K;
iIni = HMModel.ARorder + 1;
A = HMModel.StateParameters.A;   
AT = A';
% Log-likelihood and auxiliary EM variables
loglikeSeq = zeros(1, nSeq);
for iSeq = 1:nSeq
    N = HMModel.N(iSeq);
    HMModelSeq = HMModel;
    if strcmpi(HMModel.type, 'ARHMM')
        HMModelSeq.DelayARMatrix = HMModelSeq.DelayARMatrix(iSeq).Seq;
    end
    [loglike, alphaEM, auxEM] = HMMLikelihood(ySeq{iSeq}, HMModelSeq, 'method', 'scaling', 'normalize', 0);
    pYZ = auxEM.pYZ;
    coeff = auxEM.coeff;
    % Backward iterations
    betaEM = zeros(K, N);
    nback = N-1:-1:iIni;
    % First iteration
    betaEM(:, end) = 1;
    % Remaining iterations - beta
    for i = 1:numel(nback)
        aux = ((betaEM(:, nback(i)+1).*pYZ(:, nback(i)+1))' * AT)';
        betaEM(:, nback(i)) = aux/coeff(nback(i)+1);
    end
    % Conditional and joint conditionals probabilities
    gammaEM = alphaEM.*betaEM;
    xiEM = zeros(K, K, N);
    for i = iIni+1:N                                                                 % TODO - vectorize this!
        xiEM(:, :, i) = (alphaEM(:, i-1)/coeff(i)) .* ...
            (pYZ(:, i)') .* A .* betaEM(:,i)';
    end
    loglikeSeq(iSeq) = loglike;
    % Update fields
    HMModel.EM(iSeq).gamma = gammaEM;
    HMModel.EM(iSeq).xi = xiEM;  
end
HMModel.loglike = [HMModel.loglike sum(loglikeSeq)];
HMModel.loglikeNorm = [HMModel.loglikeNorm sum(loglikeSeq./HMModel.N)];
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
    if strcmpi(HMModel.type, 'ARHSMMED')
        HMModelSeq{i}.DelayARMatrix = HMModelSeq{i}.DelayARMatrix(i).Seq;
    end
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
    [loglike, logalphaEM, auxEM] = HMMLikelihood(ySeq{iSeq}, HMModelSeq{iSeq}, 'method', 'logsumexp', 'normalize', 0, 'returnAlpha', 1);
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
HMModel.loglikeNorm = [HMModel.loglikeNorm sum(loglikeSeq./HMModel.N)]
end

%% Forward-backward algorithm - Scaling - Hidden Semi-Markov Model Explicit Duration
function HMModel = ForwardBackwardScaleHSMMED(ySeq, HMModel)
% clear previous EM values
HMModel.EM = rmfield(HMModel.EM, {'gamma';'eta_iIni';'xi';'sumxi'});
nSeq = HMModel.nSeq;
K = HMModel.StateParameters.K;
iIni = HMModel.ARorder + 1;
dmax = HMModel.DurationParameters.dmax;
% Compute probabilities of observations for all possible hidden state cases
Atrans = HMModel.StateParameters.A(:, :, 1);
Astay = HMModel.StateParameters.A(:, :, 2);
AtransT = Atrans';
AstayT = Astay';
% Log-likelihood and auxiliary EM variables
loglikeSeq = zeros(1, nSeq);
for iSeq = 1:nSeq
    %iSeq
    N = HMModel.N(iSeq);
    HMModelSeq = HMModel;
    if strcmpi(HMModel.type, 'ARHSMMED')
        HMModelSeq.DelayARMatrix = HMModelSeq.DelayARMatrix(iSeq).Seq;
    end
    [loglike, alphaEM, auxEM] = HMMLikelihood(ySeq{iSeq}, HMModelSeq, 'method', 'scaling', 'normalize', 0, 'returnAlpha', 1);
    pYZ = auxEM.pYZ;
    pDZ = auxEM.pDZ;
    alphaEMiIni = alphaEM{iIni};
    coeff = auxEM.coeff;
    clear auxEM
    % beta, eta, gamma, xi - Initializations
    betaprev = ones(K, dmax);
    betaup = zeros(K, dmax);
    nback = N-1:-1:iIni;
    %etaEM = cell(1, N);
    %etaEM(:) = {zeros(K, dmax, 'single')};
    gammaEM = zeros(K, N, 'double');
    alphadmin = cell2mat(cellfun(@(x) x(:,1),  alphaEM, 'UniformOutput', false));
    xiEM = cell(1, N);
    sumxiEM = zeros(K, K);
    % beta, eta, gamma, xi - Last/first iteration
    %etaEM{N} = alphaEM{N};
    %gammaEM(:, N) = full(sum(alphaEM{N}, 2));
    gammaEM(:, N) = sum(alphaEM{N}, 2);
    aux = (alphadmin(:, N-1)/coeff(N)) .* ...
        (reshape((pYZ(:, N) .* pDZ .* betaprev), 1, K, dmax) .* Atrans);
    sumxiEM = sumxiEM + sum(aux, 3);
    xiEM{N} = sparse(squeeze(sum(double(aux), 1)));
    %xiEM{N} = squeeze(sum(aux, 1));
    % % alphaEM = alphaEM(1:end-1);         % this was good
    % beta, eta, gamma, xi - remaining iterations
    for i = 1:numel(nback)
        % beta
        betaaux = sum(betaprev .* pDZ, 2);
        betaup(:, 1) = ((pYZ(:, nback(i)+1) .* betaaux)' * AtransT)';
        betaup(:, 2:dmax) = (pYZ(:, nback(i)+1) .* AstayT) * betaprev(:, 1:dmax-1);
        betaup = betaup/coeff(nback(i)+1);
        %clear betaaux
        % eta
        % % aux = alphaEM{end} .* betaup;       % this was good
        aux = alphaEM{nback(i)} .* betaup;
        %aux = sparse(aux);
        %etaEM{nback(i)} = aux;
        % gamma
        %gammaEM(:, nback(i)) = full(sum(aux, 2));
        gammaEM(:, nback(i)) = sum(aux, 2);
        %clear aux
        % xi
        aux = zeros(K, K, dmax);
        if nback(i) > iIni
            aux(:, :, :) = (alphadmin(:, nback(i)-1)/coeff(nback(i))) .* ...
                (reshape(pYZ(:, nback(i)) .* pDZ .* betaup, 1, K, dmax)...
                .* Atrans);
            sumxiEM = sumxiEM + sum(aux, 3);
            xiEM{nback(i)} = sparse(reshape(sum(aux, 1), K, dmax));
        end
        betaprev = betaup;
        % % alphaEM = alphaEM(1:nback(i)-1);        % this was good
        alphaEM{nback(i)} = [];               % this was added
    end
    %gammaEM = sparse(gammaEM);
    loglikeSeq(iSeq) = loglike;
    eta_iIni = alphaEMiIni;
    % Update fields
    HMModel.EM(iSeq).gamma = gammaEM;
    HMModel.EM(iSeq).eta_iIni = eta_iIni;
    HMModel.EM(iSeq).xi = xiEM;  
    HMModel.EM(iSeq).sumxi = sumxiEM;
end
HMModel.loglike = [HMModel.loglike sum(loglikeSeq)];
HMModel.loglikeNorm = [HMModel.loglikeNorm sum(loglikeSeq./HMModel.N)];
end

%% Forward-backward algorithm - Logsumexp - Hidden Semi-Markov Model Variable Transition
function HMModel = ForwardBackwardLogSumExpHSMMVT(ySeq, HMModel)
% clear previous EM values
HMModel.EM = rmfield(HMModel.EM, {'gamma';'eta_iIni';'xi';'sumxi'});
nSeq = HMModel.nSeq;
K = HMModel.StateParameters.K;
iIni = HMModel.ARorder + 1;
dmax = HMModel.DurationParameters.dmax;
% Compute probabilities of observations for all possible hidden state cases
logA = log(HMModel.StateParameters.A);
% Log-likelihood and auxiliary EM variables
loglikeSeq = zeros(1, nSeq);
for iSeq = 1:nSeq
    iSeq
    N = HMModel.N(iSeq);
    distInt = HMModel.DurationParameters.DurationIntervals;
    distIntaux = distInt;
    distIntaux(end) = dmax + 1;
    HMModelSeq = HMModel;
    if strcmpi(HMModel.type, 'ARHSMMVT')
        HMModelSeq.DelayARMatrix = HMModelSeq.DelayARMatrix(iSeq).Seq;
    end
    [loglike, logalphaEM, auxEM] = HMMLikelihood(ySeq{iSeq}, HMModelSeq, 'method', 'logsumexp', 'normalize', 0, 'returnAlpha', 1);
    logpYZ = auxEM.logpYZ;
    logpDZ = auxEM.logpDZ;
    logalphaEMiIni = logalphaEM{iIni};
    clear auxEM
    % beta, eta, gamma, xi - Initializations
    idxbeta = zeros(1, K*dmax);
    idxbeta(1:K) = 1:K;
    for i = 2:dmax
        idxbeta((i-1)*K+1:i*K) = K*dmax + idxbeta((i-1)*K) + 1:K*dmax + idxbeta((i-1)*K) + K;
    end
    
    idxbeta2 = zeros(K, dmax);
    idxbeta2(:, 1) = (1:K)';
    for i = 2:dmax
        idxbeta2(:, i) = (dmax + 1)*K + idxbeta2(1, i - 1):...
            (dmax + 1)*K + idxbeta2(1, i - 1) + K - 1;
    end
    
    logbetaprev = zeros(K, dmax, dmax);   
    logbetaup = -inf(K, dmax, dmax);
    nback = N-1:-1:iIni;  
    % gamma  
    gammaEM = zeros(K, N, 'double');
    auxgamma = exp(logalphaEM{N} + logbetaprev - loglike);
    gammaEM(:, N) = sum(sum(auxgamma, 2), 3);
    % xi
    xiEM = cell(1, N);         
    logalphadmin = cellfun(@(x) x(:,1,:),  logalphaEM, 'UniformOutput', false);
    aux1xi = -inf(K, K, dmax);
    for j = 1:numel(distInt)-1
        aux1xi(:, :, distIntaux(j):distIntaux(j+1)-1) = ...
            logalphadmin{N-1}(:, :, distIntaux(j):distIntaux(j+1)-1) + logA(:, :, j);
    end
    %aux2xi = reshape(logbetaprev(idxbeta), K, dmax) + logsumexp(logsumexp(aux1xi, 3), 1)';
    aux2xi = logbetaprev(idxbeta2) + logsumexp(logsumexp(aux1xi, 3), 1)';
    xiEM{N} = exp((logpYZ(:, N) + logpDZ + aux2xi) - loglike);   
    % sumxi
    sumxiEM = zeros(K, K, numel(distInt)-1);
    %aux1sumxi = logsumexp(logpDZ + reshape(logbetaprev(idxbeta), K, dmax), 2);
    aux1sumxi = logsumexp(logpDZ + logbetaprev(idxbeta2), 2);
    for j = 1:numel(distInt)-1
        sumxiEM(:,:, j) = sumxiEM(:,:, j) + sum(exp((logpYZ(:, N)' + ...
            (logalphaEM{N-1}(:, 1, distIntaux(j):distIntaux(j+1)-1) + ...
            (aux1sumxi' + logA(:,:,j)))) - loglike), 3);
    end
    % beta, eta, gamma, xi - remaining iterations
    for i = 1:numel(nback)
        % beta
        %logbetaaux1 = logsumexp(logpDZ + reshape(logbetaprev(idxbeta), K, dmax), 2);
        logbetaaux1 = logsumexp(logpDZ + logbetaprev(idxbeta2), 2);
        for j = 1:numel(distInt)-1
            logbetaup(:, 1, distIntaux(j):distIntaux(j+1)-1) = ...
                repmat(squeeze(logsumexp(logpYZ(:, nback(i)+1) + ...
                reshape(logbetaaux1 + logA(:, :, j)', K, 1, K), 1)), 1, 1, distIntaux(j+1) - distIntaux(j));
        end       
        logbetaup(:, 2:end, :) = logpYZ(:, nback(i)+1) + logbetaprev(:, 1:dmax-1, :);
        % gamma
        auxgamma = exp(logalphaEM{nback(i)} + logbetaup - loglike);
        gammaEM(:, nback(i)) = sum(sum(auxgamma, 2), 3);      
        % xi and sumxi
        if nback(i) > 1
            aux1xi = -inf(K, K, dmax);
            for j = 1:numel(distInt)-1
                aux1xi(:, :, distIntaux(j):distIntaux(j+1)-1) = ...
                    logalphadmin{nback(i)-1}(:, :, distIntaux(j):distIntaux(j+1)-1) + logA(:, :, j);
            end
            %aux2xi = reshape(logbetaup(idxbeta), K, dmax) + logsumexp(logsumexp(aux1xi, 3), 1)';
            aux2xi = logbetaup(idxbeta2) + logsumexp(logsumexp(aux1xi, 3), 1)';
            xiEM{nback(i)} = exp((logpYZ(:, nback(i)) + logpDZ + aux2xi) - loglike);          
            %aux1sumxi = logsumexp(logpDZ + reshape(logbetaup(idxbeta), K, dmax), 2);
            aux1sumxi = logsumexp(logpDZ + logbetaup(idxbeta2), 2);
            for j = 1:numel(distInt)-1
                sumxiEM(:,:, j) = sumxiEM(:,:, j) + sum(exp((logpYZ(:, nback(i))' + ...
                    (logalphaEM{nback(i)-1}(:, 1, distIntaux(j):distIntaux(j+1)-1) + ...
                    (aux1sumxi' + logA(:,:,j)))) - loglike), 3);
            end
        end
        logbetaprev = logbetaup;
        logalphaEM{nback(i)} = [];               % this was added
    end
    %gammaEM = sparse(gammaEM);
    loglikeSeq(iSeq) = loglike;
    eta_iIni = sum(exp(logalphaEMiIni + logbetaup - loglike), 3);
    % Update fields
    HMModel.EM(iSeq).gamma = gammaEM;
    HMModel.EM(iSeq).eta_iIni = eta_iIni;
    HMModel.EM(iSeq).xi = xiEM;  
    HMModel.EM(iSeq).sumxi = sumxiEM;
    clear gammaEM xiEM logalphaEM
end
HMModel.loglike = [HMModel.loglike sum(loglikeSeq)]
HMModel.loglikeNorm = [HMModel.loglikeNorm sum(loglikeSeq./HMModel.N)];
end

%%
function HMModel = cleanFields(HMModel, sortCell)
if HMModel.SleepSpindles == 0
    HMModel = rmfield(HMModel,'SleepSpindles');
end
HMModel = rmfield(HMModel, {'N'; 'nSeq'; 'EM'});
if strcmpi(HMModel.type, 'HSMMED') || strcmpi(HMModel.type, 'ARHSMMED') || ...
        strcmpi(HMModel.type, 'HSMMVT') || strcmpi(HMModel.type, 'ARHSMMVT')
    aux = rmfield(HMModel.DurationParameters, 'flag');
    HMModel = rmfield(HMModel, 'DurationParameters');
    HMModel.DurationParameters = aux;
end
if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'HSMMED') || strcmpi(HMModel.type, 'HSMMVT')
    HMModel = rmfield(HMModel, 'ARorder');
end
if strcmpi(HMModel.type, 'ARHMM') || strcmpi(HMModel.type, 'ARHSMMED') || strcmpi(HMModel.type, 'ARHSMMVT')
    HMModel = rmfield(HMModel, 'DelayARMatrix');
end
% if isfield(HMModel, 'Fs')
%     HMModel = rmfield(HMModel,'Fs');
% end
HMModel = orderfields(HMModel, sortCell);
end