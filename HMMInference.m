function [z, loglike, drem] = HMMInference(y, HMModel, varargin)
normflag = 1;
for i = 1:length(varargin)
    if strcmpi(varargin{i}, 'normalize')
        normflag = varargin{i + 1};
    end
end
K = HMModel.StateParameters.K;
N = size(y, 2);
pmin = 2e-300;                              % To avoid underflow
dmax = HMModel.DurationParameters.dmax;
% for compatibility with other functions
if ~isfield(HMModel.DurationParameters, 'flag')
    HMModel.DurationParameters.flag = 0;
end
iIni = HMModel.ARorder + 1;
%% Format input
% Observations must be row vectors
if normflag
    y = zscore(y);                          % Normalize observations
end
%%
p = HMModel.ARorder;
% Build auxiliary matrix of AR predictors
Yp = zeros(N - p, p);
for i = 1:N-p
    Yp(i, :) = -fliplr(y(i : i + p - 1));
end
HMModel.DelayARMatrix = Yp;
%% Common variables for all models
pzIni = double(HMModel.StateParameters.pi);
pzIni(pzIni < pmin) = pmin;
logpzIni = log(pzIni);
logpYZ = LogProbObsAllZ(y, K, HMModel);
%% Viterbi decoding
Atrans = HMModel.StateParameters.A(:, :, 1);
Atrans(Atrans < pmin) = pmin;
logAtrans = log(Atrans);
logpDZ = LogProbDurAllZ(K, HMModel);
% First iteration
psi_d = cell(1, N);
psi_z = cell(1, N);
deltprev = logpYZ(:, iIni) + logpDZ + logpzIni;
idx = zeros(K, dmax - 1);
idx(:, 1) = (K*dmax+1:K*dmax+K)';
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