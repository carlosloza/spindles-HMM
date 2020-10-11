function HSMModel = HSMMLearning(ySeq, K, varargin)
%% Batch implementation - ySeq is a cell now
HSMModel.normalize = 1;
HSMModel.StateParameters.K = K;
% Check inputs and initial conditions
for i = 1:length(varargin)
    if strcmpi(varargin{i}, 'type')
        HSMModel.type = varargin{i + 1};      % options: HSMMED, ARHSMMED
    elseif strcmpi(varargin{i}, 'ARorder')
        HSMModel.ARorder = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'StateParameters') 
        HSMModel.StateParameters = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'ObsParameters') 
        HSMModel.ObsParameters = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'DurationParameters')
        HSMModel.DurationParameters = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'Estep')
        HSMModel.Estep = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'normalize')
        HSMModel.normalize = varargin{i + 1};
    end
end
% Single sequence as input case
if ~iscell(ySeq)
    aux = ySeq;
    clear ySeq
    ySeq{1} = aux;
end
% Initialize loglikelihood and EM-related fields
HSMModel.loglike = [];
HSMModel.loglikeNorm = [];
HSMModel.nSeq = numel(ySeq);
HSMModel.EM.gamma = [];
HSMModel.EM.eta = [];
HSMModel.EM.xi = [];  
HSMModel.EM.sumxi = [];
% Default E-step: scaling-based implementation
if ~isfield(HSMModel, 'Estep')
    HSMModel.Estep = 'scaling';
end
if strcmpi(HSMModel.Estep, 'logsumexp')
    funcEstep = @ForwardBackwardLogSumExp;
elseif strcmpi(HSMModel.Estep, 'scaling')
    funcEstep = @ForwardBackwardScale;
else
    % Misspelling case
    HSMModel.Estep = 'scaling';
    funcEstep = @ForwardBackwardScale;
end
%% Format input and set EM convergence parameters
HSMModel.N = zeros(1, HSMModel.nSeq);
for i = 1:HSMModel.nSeq
    y = ySeq{i};
    HSMModel.N(i) = size(y, 2);
    % Observations must be row vectors
    if iscolumn(y)
        y = y';
    end
    % Normalize each  training observation sequence 
    if HSMModel.normalize
        y = zscore(y, [], 2);
    end
    ySeq{i} = y;
end
maxIt = 20;                             % Maximum number of EM iterations
convTh = 0.01;                          % Convergence threshold
fl = 1;
It = 0;
if strcmpi(HSMModel.type, 'HSMMED')
    sortCell = {'type', 'StateParameters', 'DurationParameters', 'ObsParameters',...
        'loglike', 'loglikeNorm', 'Estep', 'normalize'};
elseif strcmpi(HSMModel.type, 'ARHSMMED')
    sortCell = {'type', 'StateParameters', 'DurationParameters', 'ObsParameters',...
        'ARorder', 'loglike', 'loglikeNorm', 'Estep', 'normalize'};
end
%% Initial conditions
HSMModel = InitialConditions(ySeq, HSMModel);
%% ExpeItation-Maximization
while fl
    % E STEP
    HSMModel = funcEstep(ySeq, HSMModel);
    % M STEP
    HSMModel = MaxEM(ySeq, HSMModel);  
    It = It + 1;
    % Check for convergence
    if It > 1
        %if abs(HSMModel.loglike(It) - HSMModel.loglike(It-1))/abs(HSMModel.loglike(It - 1)) <= convTh
        if abs(HSMModel.loglikeNorm(It) - HSMModel.loglikeNorm(It-1))/abs(HSMModel.loglikeNorm(It - 1)) <= convTh
            HSMModel = rmfield(HSMModel, {'N', 'nSeq', 'EM'});
            aux = rmfield(HSMModel.DurationParameters, 'flag');
            HSMModel = rmfield(HSMModel, 'DurationParameters');
            HSMModel.DurationParameters = aux;
            if strcmpi(HSMModel.type, 'ARHSMMED')
                HSMModel = rmfield(HSMModel, 'DelayARMatrix');
            elseif strcmpi(HSMModel.type, 'HSMMED')
                HSMModel = rmfield(HSMModel, 'ARorder');
            end
            HSMModel = orderfields(HSMModel, sortCell);
            break
        end
    end
    if It == maxIt
        HSMModel = rmfield(HSMModel, {'N', 'nSeq', 'EM'});
        aux = rmfield(HSMModel.DurationParameters, 'flag');
        HSMModel = rmfield(HSMModel, 'DurationParameters');
        HSMModel.DurationParameters = aux;
        if strcmpi(HSMModel.type, 'ARHSMMED')
            HSMModel = rmfield(HSMModel, 'DelayARMatrix');
        elseif strcmpi(HSMModel.type, 'HSMMED')
            HSMModel = rmfield(HSMModel, 'ARorder');
        end
        HSMModel = orderfields(HSMModel, sortCell);
        break
    end        
end

end
% %% Initial conditions
% function HSMModel = InitialConditions(y, HSMModel)
% N = HSMModel.N;
% K = HSMModel.StateParameters.K;
% % HSMMED
% if strcmpi(HSMModel.type, 'HSMMED')
%     % Parameters of observation model, i.e. emissions
%     if ~isfield(HSMModel, 'ObsParameters')
%         % Gaussian model for observations
%         HSMModel.ObsParameters.model = 'Gaussian';
%         [idx, C] = kmeans(y', K, 'Replicates', 10);
%         disp('Initial conditions done')
%         HSMModel.ObsParameters.mu = C';
%         for k = 1:K       
%             HSMModel.ObsParameters.sigma(1, k) = std(y(idx == k));
%         end
%     else
%         % Initial conditions are provided
%         fprintf('Initial conditions for observations are provided \n')
%     end
%     % Parameters of hidden markov chain, i.e. states
%     % Probabilities of initial state
%     if ~isfield(HSMModel, 'StateParameters')
%         HSMModel.StateParameters = [];
%     else
%         % Initial conditions are provided
%         fprintf('Initial conditions for states are provided \n')
%     end
%     if ~isfield(HSMModel.StateParameters, 'pi')
%         if exist('idx', 'var')
%             for k = 1:K
%                 HSMModel.StateParameters.pi(k, 1) = sum(idx == k);
%             end
%             HSMModel.StateParameters.pi = HSMModel.StateParameters.pi/...
%                 sum(HSMModel.StateParameters.pi);
%         else
%             HSMModel.StateParameters.pi = 1/K*ones(K, 1);
%         end       
%     end
%     if isfield(HSMModel, 'ARorder')
%         disp('Hey! vanilla HMM is not autoregressive!')                                 % TODO
%     end
%     HSMModel.ARorder = 0;
% end
% 
% % ARHSMMED
% if strcmpi(HSMModel.type, 'ARHSMMED')
%     p = HSMModel.ARorder;
%     % Build auxiliary matrix of AR predictors
%     Yp = zeros(N - p, p);
%     for i = 1:N-p
%         Yp(i, :) = -fliplr(y(i : i + p - 1));
%     end    
%     HSMModel.DelayARMatrix = Yp;
%     % Parameters of observation model, i.e. emissions
%     if ~isfield(HSMModel, 'ObsParameters')
%         % Observation noise
%         Ypini = Yp(1:4*p, :);
%         Yini = y(p+1:5*p)';
%         aini = (Ypini'*Ypini)\(Ypini'*Yini);
%         sig = std(Ypini*aini - Yini)*ones(1, K);
%         sig2 = sig(1)^2;
%         % Kalman filter
%         % Tracking parameter alph, a small value tracks the modes in the observations
%         % in a better way, a larger value smooths the states stimates, but
%         % compromises tracking
%         alph = 0.1;             
%         evTh = 0.5;             % Threshold for low evidence observations
%         KalmanARModel = KalmanAR(y, p, 'alph', alph, 'sig2', sig2);
%         a = KalmanARModel.a;
%         evX = KalmanARModel.evX;
%         % Cluster AR coefficients without low evidence
%         idxTh = evX > evTh;
%         a = a(:, idxTh);
%         [idx, C] = kmeans(a', K, 'Replicates', 10);
%         HSMModel.ObsParameters.ARcoeff = C';
% %         GMModel = fitgmdist(a', K, 'Replicates', 10, 'RegularizationValue', 0.1, 'CovarianceType', 'diagonal');     
% %         HSMModel.ObsParameters.ARcoeff = GMModel.mu';
%         HSMModel.ObsParameters.sigma = sig;
%     end   
%     % Parameters of hidden markov chain, i.e. states
%     % Probabilities of initial state
%     if ~isfield(HSMModel, 'StateParameters')
%         HSMModel.StateParameters = [];
%     end
%     if ~isfield(HSMModel.StateParameters, 'pi')
%         if exist('idx', 'var')
%             %HSMModel.StateParameters.pi = GMModel.ComponentProportion';  
%             for k = 1:K
%                 HSMModel.StateParameters.pi(k, 1) = sum(idx == k);
%             end
%             HSMModel.StateParameters.pi = HSMModel.StateParameters.pi/...
%                 sum(HSMModel.StateParameters.pi);
%         else
%             HSMModel.StateParameters.pi = 1/K*ones(K, 1);
%         end
%     end
% end
% % Transition matrices for both HSMMED and ARHSMMED cases
% if ~isfield(HSMModel.StateParameters, 'A')
%     Atrans = 0.1*eye(K);                     % Small probability of self-transitions
%     for i = 1:K
%         aux = gamrnd(1, 1, 1, K - 1);   % Sample from dirichlet distribution
%         aux = (1 - Atrans(i,i))*aux./sum(aux);
%         Atrans(i, 1:K ~= i) = aux;
%     end   
% %     Atrans = zeros(K, K);           % no self-transitions
% %     for i = 1:K
% %         aux = gamrnd(1, 1, 1, K - 1);   % Sample from dirichlet distribution
% %         aux = (1 - Atrans(i,i))*aux./sum(aux);
% %         Atrans(i, 1:K ~= i) = aux;
% %     end  
%     HSMModel.StateParameters.A(:, :, 1) = Atrans;
%     HSMModel.StateParameters.A(:, :, 2) = eye(K);
% end
% 
% % dmax for HSMMED abd ARHSMMED
% if strcmpi(HSMModel.type, 'HSMMED') || strcmpi(HSMModel.type, 'ARHSMMED')
%     if ~isfield(HSMModel.DurationParameters, 'dmin')
%         % Heuristic - might be painfully slow and biased!
%         HSMModel.DurationParameters.dmin = 1;
%     end
%     if ~isfield(HSMModel.DurationParameters, 'dmax')
%         % Heuristic - might be painfully slow
%         HSMModel.DurationParameters.dmax = round(N/K);
%     end
%     dmax = HSMModel.DurationParameters.dmax;
%     dmin = HSMModel.DurationParameters.dmin;
%     % Parameters of duration model
% %     if ~isfield(HSMModel.DurationParameters, 'PNonParametric')
% %         aux = [zeros(1, dmin-1) ones(1, dmax - dmin + 1)];
% %         aux = aux/sum(aux);
% %         HSMModel.DurationParameters.PNonParametric = repmat(aux, K, 1);
% %         % Auxiliary flag
% %         HSMModel.DurationParameters.flag = 1;
% %     else
% %         HSMModel.DurationParameters.flag = 0;
% %     end
%     HSMModel.DurationParameters.flag = 1;
%     if strcmpi(HSMModel.DurationParameters.model, 'NonParametric')
%         if ~isfield(HSMModel.DurationParameters, 'PNonParametric')
%             % Start with non-parametric model, then refine estimation
%             aux = [zeros(1, dmin-1) ones(1, dmax - dmin + 1)];
%             aux = aux/sum(aux);
%             HSMModel.DurationParameters.PNonParametric = repmat(aux, K, 1);
%             HSMModel.DurationParameters.flag = 0;
%         end
%     else
%         if strcmpi(HSMModel.DurationParameters.model, 'Poisson')
%             if ~isfield(HSMModel.DurationParameters, 'lambda')
%                 aux = [zeros(1, dmin-1) ones(1, dmax - dmin + 1)];
%                 aux = aux/sum(aux);
%                 HSMModel.DurationParameters.PNonParametric = repmat(aux, K, 1);
%                 % Auxiliary flag
%                 HSMModel.DurationParameters.flag = 1;
%             else
%                 HSMModel.DurationParameters.flag = 0;
%             end
%         elseif strcmpi(HSMModel.DurationParameters.model, 'Gaussian')
%             if ~isfield(HSMModel.DurationParameters, 'mu')
%                 aux = [zeros(1, dmin-1) ones(1, dmax - dmin + 1)];
%                 aux = aux/sum(aux);
%                 HSMModel.DurationParameters.PNonParametric = repmat(aux, K, 1);
%                 % Auxiliary flag
%                 HSMModel.DurationParameters.flag = 1;
%             else
%                 HSMModel.DurationParameters.flag = 0;
%             end
%         end
%     end
%     % Pre-compute factorials for Poisson model
%     if strcmpi(HSMModel.DurationParameters.model, 'Poisson')
%         HSMModel.DurationParameters.dlogfact = log(factorial(1:dmax));
%     end
% end
% end

%% Forward-backward algorithm - Logsumexp
function HSMModel = ForwardBackwardLogSumExp(ySeq, HSMModel)
% clear previous EM values
HSMModel.EM = rmfield(HSMModel.EM, 'gamma');
HSMModel.EM = rmfield(HSMModel.EM, 'eta'); 
HSMModel.EM = rmfield(HSMModel.EM, 'xi'); 
HSMModel.EM = rmfield(HSMModel.EM, 'sumxi'); 
nSeq = HSMModel.nSeq;
K = HSMModel.StateParameters.K;
iIni = HSMModel.ARorder + 1;
dmax = HSMModel.DurationParameters.dmax;
% Compute probabilities of observations for all possible hidden state cases
Atrans = HSMModel.StateParameters.A(:, :, 1);
Astay = HSMModel.StateParameters.A(:, :, 2);
logAtrans = log(Atrans);
logAtransT = logAtrans';
logAstay = log(Astay);
logAstayT = logAstay';
% Log-likelihood and auxiliary EM variables
loglikeSeq = zeros(1, nSeq);
for iSeq = 1:nSeq
    N = HSMModel.N(iSeq);
    HSMModelSeq = HSMModel;
    if strcmpi(HSMModel.type, 'ARHSMMED')
        HSMModelSeq.DelayARMatrix = HSMModelSeq.DelayARMatrix(iSeq).Seq;
    end
    [loglike, logalphaEM, auxEM] = HSMMLikelihood(ySeq{iSeq}, HSMModelSeq, 'method', 'logsumexp', 'normalize', 0, 'returnAlpha', 1);
    logpYZ = auxEM.logpYZ;
    logpDZ = auxEM.logpDZ;
    clear auxEM
    % beta, eta, gamma, xi - Initializations
    logbetaprev = zeros(K, dmax);   
    logbetaup = -inf(K, dmax);
    nback = N-1:-1:iIni;
    etaEM = cell(1, N);
    gammaEM = zeros(K, N);
    logalphadmin = cell2mat(cellfun(@(x) x(:,1),  logalphaEM, 'UniformOutput', false));
    xiEM = cell(1, N);
    sumxiEM = zeros(K, K);
    % beta, eta, gamma, xi - Last/first iteration
    etaEM{N} = sparse(exp(logalphaEM{N} + logbetaprev - loglike));
    gammaEM(:, N) = sum(etaEM{N}, 2);
    aux = exp((logalphadmin(:, N-1) + ...
        reshape((logpYZ(:, N) + logpDZ + logbetaprev), 1, K, dmax)...
        + logAtrans) - loglike);
    sumxiEM = sumxiEM + sum(aux, 3);
    xiEM{N} = sparse(squeeze(sum(aux, 1)));
    logalphaEM = logalphaEM(1:end-1);
    % beta, eta, gamma, xi - remaining iterations
    for i = 1:numel(nback)
        % beta
        logbetaaux = logsumexp(logbetaprev + logpDZ, 2);
        logbetaup(:, 1) = logsumexp(logpYZ(:, nback(i)+1) + logAtransT + logbetaaux, 1)';
        logbetaup(:, 2:dmax) = squeeze(logsumexp(reshape(logpYZ(:, nback(i)+1) ...
            + logAstayT, K, 1, K) + logbetaprev(:, 1:dmax-1), 1))';
        clear logbetaaux
        % eta
        aux = exp(logalphaEM{end} + logbetaup - loglike);
        aux = sparse(aux);
        etaEM{nback(i)} = aux;
        % gamma
        gammaEM(:, nback(i)) = full(sum(aux, 2));
        % xi
        aux = zeros(K, K, dmax);
        if nback(i) > 1
            aux(:, :, :) = exp((logalphadmin(:, nback(i)-1) + ...
                reshape((logpYZ(:, nback(i)) + logpDZ + logbetaup), 1, K, dmax)...
                + logAtrans) - loglike);
            sumxiEM = sumxiEM + sum(aux, 3);
            xiEM{nback(i)} = sparse(reshape(sum(aux, 1), K, dmax));
        end
        logbetaprev = logbetaup;
        logalphaEM = logalphaEM(1:nback(i)-1);
    end
    gammaEM = sparse(gammaEM);
    loglikeSeq(iSeq) = loglike;
    % Update fields
    HSMModel.EM(iSeq).gamma = gammaEM;
    HSMModel.EM(iSeq).eta = etaEM;
    HSMModel.EM(iSeq).xi = xiEM;  
    HSMModel.EM(iSeq).sumxi = sumxiEM;
end
HSMModel.loglike = [HSMModel.loglike sum(loglikeSeq)];
HSMModel.loglikeNorm = [HSMModel.loglikeNorm sum(loglikeSeq./HSMModel.N)];
end

%% Forward-backward algorithm - Scaling
function HSMModel = ForwardBackwardScale(ySeq, HSMModel)
% clear previous EM values
HSMModel.EM = rmfield(HSMModel.EM, 'gamma');
HSMModel.EM = rmfield(HSMModel.EM, 'eta'); 
HSMModel.EM = rmfield(HSMModel.EM, 'xi'); 
HSMModel.EM = rmfield(HSMModel.EM, 'sumxi'); 
nSeq = HSMModel.nSeq;
K = HSMModel.StateParameters.K;
iIni = HSMModel.ARorder + 1;
dmax = HSMModel.DurationParameters.dmax;
% Compute probabilities of observations for all possible hidden state cases
Atrans = HSMModel.StateParameters.A(:, :, 1);
Astay = HSMModel.StateParameters.A(:, :, 2);
AtransT = Atrans';
AstayT = Astay';
% Log-likelihood and auxiliary EM variables
loglikeSeq = zeros(1, nSeq);
for iSeq = 1:nSeq
    N = HSMModel.N(iSeq);
    HSMModelSeq = HSMModel;
    if strcmpi(HSMModel.type, 'ARHSMMED')
        HSMModelSeq.DelayARMatrix = HSMModelSeq.DelayARMatrix(iSeq).Seq;
    end
    [loglike, alphaEM, auxEM] = HSMMLikelihood(ySeq{iSeq}, HSMModelSeq, 'method', 'scaling', 'normalize', 0, 'returnAlpha', 1);
    pYZ = auxEM.pYZ;
    pDZ = auxEM.pDZ;
    coeff = auxEM.coeff;
    clear auxEM
    % beta, eta, gamma, xi - Initializations
    betaprev = ones(K, dmax);
    betaup = zeros(K, dmax);
    nback = N-1:-1:iIni;
    etaEM = cell(1, N);
    gammaEM = zeros(K, N);
    alphadmin = full(cell2mat(cellfun(@(x) x(:,1),  alphaEM, 'UniformOutput', false)));
    xiEM = cell(1, N);
    sumxiEM = zeros(K, K);
    % beta, eta, gamma, xi - Last/first iteration
    etaEM{N} = alphaEM{N};
    %gammaEM(:, N) = full(sum(aux, 2));
    gammaEM(:, N) = full(sum(alphaEM{N}, 2));
    aux = (alphadmin(:, N-1)/coeff(N)) .* ...
        (reshape((pYZ(:, N) .* pDZ .* betaprev), 1, K, dmax) .* Atrans);
    sumxiEM = sumxiEM + sum(aux, 3);
    xiEM{N} = sparse(squeeze(sum(aux, 1)));
    alphaEM = alphaEM(1:end-1);
    % beta, eta, gamma, xi - remaining iterations
    for i = 1:numel(nback)
        % beta
        betaaux = sum(betaprev .* pDZ, 2);
        betaup(:, 1) = ((pYZ(:, nback(i)+1) .* betaaux)' * AtransT)';
        betaup(:, 2:dmax) = (pYZ(:, nback(i)+1) .* AstayT) * betaprev(:, 1:dmax-1);
        betaup = betaup/coeff(nback(i)+1);
        clear betaaux
        % eta
        aux = alphaEM{end} .* betaup;
        aux = sparse(aux);
        etaEM{nback(i)} = aux;
        % gamma
        gammaEM(:, nback(i)) = full(sum(aux, 2));
        clear aux
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
        alphaEM = alphaEM(1:nback(i)-1);
    end
    gammaEM = sparse(gammaEM);
    loglikeSeq(iSeq) = loglike;
    % Update fields
    HSMModel.EM(iSeq).gamma = gammaEM;
    HSMModel.EM(iSeq).eta = etaEM;
    HSMModel.EM(iSeq).xi = xiEM;  
    HSMModel.EM(iSeq).sumxi = sumxiEM;
end
HSMModel.loglike = [HSMModel.loglike sum(loglikeSeq)];
HSMModel.loglikeNorm = [HSMModel.loglikeNorm sum(loglikeSeq./HSMModel.N)];
end