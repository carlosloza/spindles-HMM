function [loglike, alphaEM, auxEM] = HMMLikelihood(y, HMModel, varargin)
K = HMModel.StateParameters.K;
N = size(y, 2);

iIni = HMModel.ARorder + 1;

dmax = HMModel.DurationParameters.dmax;
if ~isfield(HMModel.DurationParameters, 'flag')    %%
    HMModel.DurationParameters.flag = 0;
end

% defaults
normflag = 1;
alphaflag = 0;
% Check inputs
for i = 1:length(varargin)
    if strcmpi(varargin{i}, 'normalize')
        normflag = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'returnAlpha')
        alphaflag = varargin{i + 1};
    end
end
if normflag
    y = zscore(y);                          % Normalize observations
end
if ~isfield(HMModel, 'DelayARMatrix')
    p = HMModel.ARorder;
    % Build auxiliary matrix of AR predictors
    Yp = zeros(N - p, p);
    for i = 1:N-p
        Yp(i, :) = -fliplr(y(i : i + p - 1));
    end
    HMModel.DelayARMatrix = Yp;
end

logpYZ = LogProbObsAllZ(y, K, HMModel);
logpDZ = LogProbDurAllZ(K, HMModel);

%%
nforw = iIni+1:N;
auxEM.logpYZ = logpYZ;
auxEM.logpDZ = logpDZ;
% Compute probabilities of observations for all possible hidden state cases
logpzIni = log(HMModel.StateParameters.pi);
Atrans = HMModel.StateParameters.A(:, :, 1);
logAtrans = log(Atrans);
if alphaflag
    alphaEM = cell(1, N);    % inside each cell element: states x durations
    alphaEM(:) = {-inf(K, dmax, 'double')};
    % First iteration
    alphaEM{iIni} = double(logpYZ(:, iIni) + logpDZ + logpzIni);
    logalphaEMprev = alphaEM{iIni};
    % Remaining iterations
    for i = 1:numel(nforw)
        logalphaauxdmin = logsumexp(logalphaEMprev(:, 1) + logAtrans, 1)';
        logalphaEMpos = -inf(K, dmax);
        logalphaEMpos(:, dmax) = logpYZ(:, nforw(i)) + logpDZ(:, dmax) + logalphaauxdmin;
        logalphaEMpos(:, 1:dmax-1) = logpYZ(:, nforw(i)) + ...
            logsumexp(cat(3, logpDZ(:,1:dmax-1) + logalphaauxdmin, logalphaEMprev(:, 2:dmax)), 3);
        logalphaEMprev = logalphaEMpos;
        alphaEM{nforw(i)} = logalphaEMpos;
    end
else
    alphaEM = [];
    % First iteration
    logalphaEMprev = double(logpYZ(:, iIni) + logpDZ + logpzIni);
    % Remaining iterations
    for i = 1:numel(nforw)
        logalphaauxdmin = logsumexp(logalphaEMprev(:, 1) + logAtrans, 1)';
        logalphaEMpos = -inf(K, dmax);
        logalphaEMpos(:, dmax) = logpYZ(:, nforw(i)) + logpDZ(:, dmax) + logalphaauxdmin;
        logalphaEMpos(:, 1:dmax-1) = logpYZ(:, nforw(i)) + ...
            logsumexp(cat(3, logpDZ(:,1:dmax-1) + logalphaauxdmin, logalphaEMprev(:, 2:dmax)), 3);
        logalphaEMprev = logalphaEMpos;
    end
end
% log-likelihood
loglike = logsumexp(logsumexp(logalphaEMpos, 2), 1);

end