function [z, loglike, drem] = HMMInference(y, HMModel, varargin)
normflag = 1;
for i = 1:length(varargin)
    if strcmpi(varargin{i}, 'normalize')
        normflag = varargin{i + 1};
    end
end
K = HMModel.StateParameters.K;
N = size(y, 2);
z = zeros(1, N);
pmin = 2e-300;                              % To avoid underflow
if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'ARHMM')
    flagtype = 1;           % HMM variants
elseif strcmpi(HMModel.type, 'HSMMED') || strcmpi(HMModel.type, 'ARHSMMED')
    flagtype = 2;           % HSMMED variants
    drem = zeros(1, N);
    dmax = HMModel.DurationParameters.dmax;
    % for compatibility with other functions
    if ~isfield(HMModel.DurationParameters, 'flag')
        HMModel.DurationParameters.flag = 0;
    end
elseif strcmpi(HMModel.type, 'HSMMVT') || strcmpi(HMModel.type, 'ARHSMMVT')
    flagtype = 3;           % HSMMED variants
    drem = zeros(1, N);
    dmax = HMModel.DurationParameters.dmax;
    % for compatibility with other functions
    if ~isfield(HMModel.DurationParameters, 'flag')
        HMModel.DurationParameters.flag = 0;
    end
end
if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'HSMMED') || strcmpi(HMModel.type, 'HSMMVT')
    iIni = 1;
else
    iIni = HMModel.ARorder + 1;
end
%% Format input
% Observations must be row vectors
% if iscolumn(y)
%     y = y';
% end
if normflag
    y = zscore(y);                          % Normalize observations
end

%%
if strcmpi(HMModel.type, 'ARHMM') || strcmpi(HMModel.type, 'ARHSMMED') || strcmpi(HMModel.type, 'ARHSMMVT')
    p = HMModel.ARorder;
    % Build auxiliary matrix of AR predictors
    Yp = zeros(N - p, p);
    for i = 1:N-p
        Yp(i, :) = -fliplr(y(i : i + p - 1));
    end       
    % normalize
    %Yp = normalize(Yp, 1, 'zscore');
    HMModel.DelayARMatrix = Yp;
end
%% Common variables for all models
pzIni = double(HMModel.StateParameters.pi);
pzIni(pzIni < pmin) = pmin;
logpzIni = log(pzIni);
logpYZ = LogProbObsAllZ(y, K, HMModel);
%% 
switch flagtype
    case 1              % HMM and ARHMM
        A = HMModel.StateParameters.A;
        A(A < pmin) = pmin;
        logA = log(A);
        % Forward direction
        delt = zeros(K, N);
        psi = zeros(K, N);
        % First iteration
        delt(:, iIni) = logpzIni + logpYZ(:, iIni);
        % Rest of iterations
        for i = iIni + 1:N
            [aux1, aux2] = max(delt(:, i-1) + logA, [], 1);
            delt(:, i) = logpYZ(:, i) + aux1';
            psi(:, i) = aux2';
        end
        % Maximum probability and corresponding stat for last obervation
        [loglike, z(end)] = max(delt(:, end));
        % Backtracking
        for i = N-1:-1:iIni
            z(i) = psi(z(i+1), i+1);
        end
        drem = [];
    case 2
        Atrans = HMModel.StateParameters.A(:, :, 1);
        Atrans(Atrans < pmin) = pmin;
        Astay = HMModel.StateParameters.A(:, :, 2);
        Astay(Astay < pmin) = pmin;
        logAtrans = log(Atrans);
        logAstay = log(Astay);
        logpDZ = LogProbDurAllZ(K, HMModel);
%         tic
%         % Forward direction
%         psi = zeros(K, dmax, 2, N, 'single');
%         % First iteration
%         deltprev = logpYZ(:, iIni) + logpDZ + logpzIni;
%         % Rest of iterations
%         for i = iIni + 1:N
%             aux1 = zeros(K, K, dmax);
%             aux2 = zeros(K, K, dmax-1, 2);
%             aux1(:, :, 1) = deltprev(:, 1) + logAtrans;
%             aux1(:, :, 2:dmax) = reshape(deltprev(:, 2:dmax), K, 1, dmax-1) + logAstay;
%             % not dmax
%             aux2(:,:,:,1) = reshape(logpDZ(:, 1:dmax-1), 1, K, dmax-1) + aux1(:, :, 1);
%             aux2(:,:,:,2) = aux1(:, :, 2:dmax);
%             [max1, idxmax1] = max(aux2, [], 4);
%             [max2, idxmax2] = max(max1, [], 1);
%             max2 = squeeze(max2);
%             idxmax2 = squeeze(idxmax2);
%             deltprev(:, 1:dmax-1) = logpYZ(:, i) + max2;
%             % Indices
%             auxpsi = reshape(idxmax1(K*(0:1:numel(idxmax1)/K-1) + idxmax2(:)'), K, dmax-1);
%             % Correct for maximum at d = j+1
%             idx = find(auxpsi == 2);
%             auxpsi(idx) = ceil(idx/K) + 1;
%             psi(:, 1:dmax-1, 1, i) = idxmax2;
%             psi(:, 1:dmax-1, 2, i) = auxpsi;
%             % dmax
%             [maxdmax, idxdmax] = max(logpDZ(:, dmax)' + aux1(:, :, 1), [], 1);
%             deltprev(:, dmax) = logpYZ(:, i) + maxdmax';
%             psi(:, dmax, 1, i) = idxdmax';
%             psi(:, dmax, 2, i) = 1;
%         end
%         [loglike, idxloglike] = max(deltprev(:));
%         [z(end), drem(end)] = ind2sub([K dmax], idxloglike);
%         % Backtracking
%         for i = N-1:-1:iIni
%             aux = psi(z(i+1), drem(i+1), :, i+1);
%             z(i) = aux(1);
%             drem(i) = aux(2);
%         end
%         toc
%        tic
        % implementation for large data
        % First iteration
        psi_d = cell(1, N);
        %psi_d = zeros(K, dmax, N, 'uint16');
        psi_z = cell(1, N);
        %psi_z = zeros(K, dmax, N, 'uint16');
        deltprev = logpYZ(:, iIni) + logpDZ + logpzIni;        
        idx = zeros(K, dmax - 1);
        idx(:, 1) = (K*dmax+1:K*dmax+K)';
        for i = 2:dmax-1
            idx(:, i) = (dmax + 1)*K + idx(1, i - 1):...
                (dmax + 1)*K + idx(1, i - 1) + K - 1;
        end       
        % Rest of iterations
        auxidx_z_n1 = repmat((1:K)', 1, dmax-1);
        %aux = -inf(K, dmax, dmax);
        aux1 = repmat(2:dmax, K, 1);
        for i = iIni + 1:N
%             if mod(i, 100) == 0
%                 disp(num2str(i))
%             end
            % maximum over z_n1 (previous z)
            [mx1, auxidx] = max(deltprev(:, 1) + logAtrans, [], 1);  % K x 1   
            max_z_n1 = [mx1' deltprev(:, 2:dmax)];
            psi_z{i} = [auxidx' auxidx_z_n1];
            % maximum over d_n1 (previous duration)
%             aux(:, :, 1) = logpDZ + max_z_n1(:, 1);
%             aux(idx) = max_z_n1(:, 2:dmax);
%             [max_d_n1, psi_d{i}] = max(aux, [], 3);
            
            % alternative faster implementation
            [max_d_n1, idxtemp] = max(cat(3, logpDZ + max_z_n1(:, 1),...
                [max_z_n1(:, 2:dmax) -inf(K, 1)]), [], 3);
            idxdmax = idxtemp(:,dmax);
            idxtemp = idxtemp(:, 1:dmax-1);
            idxtemp(idxtemp == 1) = nan;
            idxtemp(idxtemp == 2) = 0;
            idxtemp = idxtemp + aux1;
            idxtemp(isnan(idxtemp)) = 1;
            psi_d{i} = [idxtemp idxdmax];

%             if norm(max_d_n1_alt(:) - max_d_n1_alt(:)) ~= 0 ||...
%                     norm(psi_d{i}(:) - c3(:)) ~= 0
%                 asd = 1;
%             end
            
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
        
        
%        toc
        asd = 1;
    case 3
        distInt = HMModel.DurationParameters.DurationIntervals;
        distIntaux = distInt;
        distIntaux(end) = dmax + 1;
        A = HMModel.StateParameters.A;
        A(A < pmin) = pmin;
        logA = log(A);
        logpDZ = LogProbDurAllZ(K, HMModel);
        % First iteration
        deltprev = repmat(logpYZ(:, iIni) + logpDZ + logpzIni, 1, 1, dmax);
        % Rest of iterations
        for i = iIni + 1:N
            % max over zn1 (z_{n-1})
            aux1 = -inf(K, K, dmax);
            deltr_n1 = deltprev(:, 1, :);
            for j = 1:numel(distInt)-1
                aux1(:, :, distIntaux(j):distIntaux(j+1)-1) = ...
                    deltr_n1(:,:, distIntaux(j):distIntaux(j+1)-1) + logA(:, :, j);
            end
            [mx1, idx1] = max(aux1, [], 1);
            mx1 = reshape(mx1, 3, 1, 20);
            idx1 = reshape(idx1, 3, 1, 20);
            mx2 = deltprev(:, 2:end, :);
            idx2 = repmat((1:3)', 1, 19, 20);
            max_zn1 = cat(2, mx2, mx1);
            idx_zn1 = cat(2, idx2, idx1);
            % max over 
        end
end

end