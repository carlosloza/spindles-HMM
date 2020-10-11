function [z, ds, loglike] = HSMMInference(y, HSMModel, varargin)
normflag = 1;
for i = 1:length(varargin)
    if strcmpi(varargin{i}, 'normalize')
        normflag = varargin{i + 1};
    end
end
N = size(y, 2);
z = zeros(1, N);
ds = zeros(1, N);
pmin = 2e-300;                              % To avoid underflow
iIni = HSMModel.ARorder + 1;
K = HSMModel.StateParameters.K;
dmax = HSMModel.DurationParameters.dmax;
% for compatibility with other functions
if ~isfield(HSMModel.DurationParameters, 'flag')
    HSMModel.DurationParameters.flag = 0;
end
%% Format input
% Observations must be row vectors
if iscolumn(y)
    y = y';
end
if normflag
    y = zscore(y);                          % Normalize observations
end
%% 
pzIni = HSMModel.StateParameters.pi;
pzIni(pzIni < pmin) = pmin;
logpzIni = log(pzIni);

Atrans = HSMModel.StateParameters.A(:, :, 1);
Atrans(Atrans < pmin) = pmin;
Astay = HSMModel.StateParameters.A(:, :, 2);
Astay(Astay < pmin) = pmin;
logAtrans = log(Atrans);
logAstay = log(Astay);
if strcmpi(HSMModel.type, 'ARHSMMED')
    p = HSMModel.ARorder;
    % Build auxiliary matrix of AR predictors
    Yp = zeros(N - p, p);
    for i = 1:N-p
        Yp(i, :) = -fliplr(y(i : i + p - 1));
    end    
    HSMModel.DelayARMatrix = Yp;
end
logpYZ = LogProbObsAllZ(y, K, HSMModel);
logpDZ = LogProbDurAllZ(K, HSMModel);

% % Forward direction
% psi = zeros(K, 2, dmax, N);
% % First iteration
% deltprev = logpYZ(:, iIni) + logpDZ + logpzIni;
% % Rest of iterations
% for i = iIni + 1:N
%     aux = zeros(K, K, dmax);   
%     aux(:, :, 1) = deltprev(:, 1) + logAtrans;
%     aux(:, :, 2:dmax) = reshape(deltprev(:, 2:dmax), K, 1, dmax-1) + logAstay;  
%     for j = 1:dmax              % dn
%         aux1 = -inf(size(aux));
%         aux1(:, :, 1) = logpDZ(:, j)' + aux(:, :, 1);   % dn_1 = 1
%         if j < dmax
%             aux1(:, :, j+1) = aux(:, :, j+1);
%         end
%         for k = 1:K 
%             aaa = squeeze(aux1(:,k,:));
%             b = max(aaa(:));
%             deltprev(k, j) = logpYZ(k, i) + b;
%             [s,r] = find(aaa == b, 1);
%             psi(k, :, j, i) = [s r];
%         end     
%     end
% end
% % Maximum probability and corresponding stat for last obervation (?)
% a = deltprev(:,:);
% loglike = max(a(:));

% FAST
% Forward direction
psi = zeros(K, dmax, 2, N, 'single');
% First iteration
deltprev = logpYZ(:, iIni) + logpDZ + logpzIni;
% Rest of iterations
for i = iIni + 1:N
    aux1 = zeros(K, K, dmax);  
    aux2 = zeros(K, K, dmax-1, 2);
    aux1(:, :, 1) = deltprev(:, 1) + logAtrans;
    aux1(:, :, 2:dmax) = reshape(deltprev(:, 2:dmax), K, 1, dmax-1) + logAstay;    
    % not dmax
    aux2(:,:,:,1) = reshape(logpDZ(:, 1:dmax-1), 1, K, dmax-1) + aux1(:, :, 1);
    aux2(:,:,:,2) = aux1(:, :, 2:dmax);
    [max1, idxmax1] = max(aux2, [], 4);
    [max2, idxmax2] = max(max1, [], 1);
    max2 = squeeze(max2);
    idxmax2 = squeeze(idxmax2);
    deltprev(:, 1:dmax-1) = logpYZ(:, i) + max2;
    % Indices
    auxpsi = reshape(idxmax1(K*(0:1:numel(idxmax1)/K-1) + idxmax2(:)'), K, dmax-1);
    % Correct for maximum at d = j+1
    idx = find(auxpsi == 2);
    auxpsi(idx) = ceil(idx/K) + 1;
    psi(:, 1:dmax-1, 1, i) = idxmax2;
    psi(:, 1:dmax-1, 2, i) = auxpsi;    
    % dmax
    [maxdmax, idxdmax] = max(logpDZ(:, dmax)' + aux1(:, :, 1), [], 1);
    deltprev(:, dmax) = logpYZ(:, i) + maxdmax';
    psi(:, dmax, 1, i) = idxdmax';
    psi(:, dmax, 2, i) = 1;
end
[loglike, idxloglike] = max(deltprev(:));
[z(end), ds(end)] = ind2sub([K dmax], idxloglike);
% Backtracking
for i = N-1:-1:iIni
    aux = psi(z(i+1), ds(i+1), :, i+1);
    z(i) = aux(1);
    ds(i) = aux(2);
end

% [z(end), ds(end)] = find(a == loglike, 1); % this works!
% % Backtracking
% for i = N-1:-1:iIni
%     aux = psi(z(i+1), :, ds(i+1), i+1);
%     z(i) = aux(1);
%     ds(i) = aux(2);
% end
end