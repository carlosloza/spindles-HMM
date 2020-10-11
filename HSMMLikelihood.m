function [loglike, alphaEM, auxEM] = HSMMLikelihood(y, HSMModel, varargin)
K = HSMModel.StateParameters.K;
N = numel(y);
dmax = HSMModel.DurationParameters.dmax;  %%
iIni = HSMModel.ARorder + 1;
% for compatibility with other functions
if ~isfield(HSMModel.DurationParameters, 'flag')    %%
    HSMModel.DurationParameters.flag = 0;
end
% defaults
Estep = 'scaling';
normflag = 1;
alphaflag = 0;
% Check inputs
for i = 1:length(varargin)
    if strcmpi(varargin{i}, 'method')
        Estep = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'normalize')
        normflag = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'returnAlpha')
        alphaflag = varargin{i + 1};
    end
end
% Observations must be row vectors
if iscolumn(y)
    y = y';
end
if normflag
    y = zscore(y);                          % Normalize observations
end
if strcmpi(HSMModel.type, 'ARHSMMED')
    if ~isfield(HSMModel, 'DelayARMatrix')
        p = HSMModel.ARorder;
        % Build auxiliary matrix of AR predictors
        Yp = zeros(N - p, p);
        for i = 1:N-p
            Yp(i, :) = -fliplr(y(i : i + p - 1));
        end
        HSMModel.DelayARMatrix = Yp;
    end
end
logpYZ = LogProbObsAllZ(y, K, HSMModel);
logpDZ = LogProbDurAllZ(K, HSMModel);
%%
nforw = iIni+1:N;
if strcmpi(Estep, 'logsumexp')
    auxEM.logpYZ = logpYZ;
    auxEM.logpDZ = logpDZ;
    % Compute probabilities of observations for all possible hidden state cases
    logpzIni = log(HSMModel.StateParameters.pi);
    Atrans = HSMModel.StateParameters.A(:, :, 1);
    Astay = HSMModel.StateParameters.A(:, :, 2);
    logAtrans = log(Atrans);
    logAstay = log(Astay);
    if alphaflag
        alphaEM = cell(1, N);    % inside each cell element: states x durations
        alphaEM(:) = {-inf(K, dmax)};
        % First iteration
        alphaEM{iIni} = logpYZ(:, iIni) + logpDZ + logpzIni;
        logalphaEMprev = alphaEM{iIni};
        % Remaining iterations
        for i = 1:numel(nforw)
            logalphaauxdmin = logsumexp(logalphaEMprev(:, 1) + logAtrans, 1)';
            logalphaaux = squeeze(logsumexp(logalphaEMprev(:, 2:dmax) + ...
                reshape(logAstay, K, 1, K), 1));
            logalphaEMpos = -inf(K, dmax);
            logalphaEMpos(:, dmax) = logpYZ(:, nforw(i)) + logpDZ(:, dmax) + logalphaauxdmin;
            logalphaEMpos(:, 1:dmax-1) = logpYZ(:, nforw(i)) + squeeze(logsumexp(cat(1, ...
                reshape(logalphaaux, 1, dmax - 1, K),...
                reshape((logpDZ(:,1:dmax-1) + logalphaauxdmin)', 1, dmax-1, K)), 1))';
            logalphaEMprev = logalphaEMpos;
            alphaEM{nforw(i)} = logalphaEMpos;
        end
    else
        alphaEM = [];
        % First iteration
        logalphaEMprev = logpYZ(:, iIni) + logpDZ + logpzIni;
        % Remaining iterations
        for i = 1:numel(nforw)
            i
            logalphaauxdmin = logsumexp(logalphaEMprev(:, 1) + logAtrans, 1)';
            logalphaaux = squeeze(logsumexp(logalphaEMprev(:, 2:dmax) + ...
                reshape(logAstay, K, 1, K), 1));
            logalphaEMpos = -inf(K, dmax);
            logalphaEMpos(:, dmax) = logpYZ(:, nforw(i)) + logpDZ(:, dmax) + logalphaauxdmin;
            logalphaEMpos(:, 1:dmax-1) = logpYZ(:, nforw(i)) + squeeze(logsumexp(cat(1, ...
                reshape(logalphaaux, 1, dmax - 1, K),...
                reshape((logpDZ(:,1:dmax-1) + logalphaauxdmin)', 1, dmax-1, K)), 1))';
            logalphaEMprev = logalphaEMpos;
        end
    end  
    % log-likelihood
    loglike = logsumexp(logsumexp(logalphaEMpos, 2), 1);
elseif strcmpi(Estep, 'scaling')
    pYZ = exp(logpYZ);
    pDZ = exp(logpDZ);
    auxEM.pYZ = pYZ;
    auxEM.pDZ = pDZ;
    % Compute probabilities of observations for all possible hidden state cases
    pzIni = HSMModel.StateParameters.pi;
    Atrans = HSMModel.StateParameters.A(:, :, 1);
    Astay = HSMModel.StateParameters.A(:, :, 2);
    coeff = -inf(1, N);         % scaling coefficients
    if alphaflag
        alphaEM = cell(1, N);       % inside each cell element: states x durations
        alphaEM(:) = {zeros(K, dmax)};
        % First iteration
        aux = pYZ(:, iIni) .* pDZ .* pzIni;
        coeff(iIni) = sum(aux(:));
        aux = aux/coeff(iIni);
        alphaEM{iIni} = aux;
        % Remaining iterations
        alphaEMprev = alphaEM{iIni};
        for i = 1:numel(nforw)
            aux = zeros(K, dmax);
            alphaauxdmin = (alphaEMprev(:, 1)'*Atrans)';
            alphaaux = (alphaEMprev(:, 2:dmax)' * Astay)';
            aux(:, dmax) = pYZ(:, nforw(i)) .* pDZ(:, dmax) .* alphaauxdmin;
            aux(:, 1:dmax-1) = pYZ(:, nforw(i)).*...
                (alphaaux + pDZ(:,1:dmax-1) .* alphaauxdmin);
            coeff(nforw(i)) = sum(aux(:));
            aux = aux/coeff(nforw(i));
            alphaEM{nforw(i)} = aux;
            alphaEMprev = aux;
        end
    else
        alphaEM = [];       % inside each cell element: states x durations
        % First iteration
        aux = pYZ(:, iIni) .* pDZ .* pzIni;
        coeff(iIni) = sum(aux(:));
        aux = aux/coeff(iIni);
        % Remaining iterations
        alphaEMprev = aux;
        for i = 1:numel(nforw)
            i
            aux = zeros(K, dmax);
            alphaauxdmin = (alphaEMprev(:, 1)'*Atrans)';
            alphaaux = (alphaEMprev(:, 2:dmax)' * Astay)';
            aux(:, dmax) = pYZ(:, nforw(i)) .* pDZ(:, dmax) .* alphaauxdmin;
            aux(:, 1:dmax-1) = pYZ(:, nforw(i)).*...
                (alphaaux + pDZ(:,1:dmax-1) .* alphaauxdmin);
            coeff(nforw(i)) = sum(aux(:));
            aux = aux/coeff(nforw(i));
            alphaEMprev = aux;
        end
    end
    % log-likelihood
    loglike = sum(log(coeff(iIni:end))); 
    auxEM.coeff = coeff;
else
    disp('Error')                   % TODO
end

end