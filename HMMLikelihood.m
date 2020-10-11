function [loglike, alphaEM, auxEM] = HMMLikelihood(y, HMModel, varargin)
K = HMModel.StateParameters.K;
N = size(y, 2);

if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'ARHMM')
    flagtype = 1;           % HMM variants
elseif strcmpi(HMModel.type, 'HSMMED') || strcmpi(HMModel.type, 'ARHSMMED')
    flagtype = 2;           % HSMMED variants
elseif strcmpi(HMModel.type, 'HSMMVT') || strcmpi(HMModel.type, 'ARHSMMVT')
    flagtype = 3;           % HSMMVT variants
end

if ~isfield(HMModel, 'ARorder')
    iIni = 1;
else
    iIni = HMModel.ARorder + 1;
end
if flagtype == 2 || flagtype == 3
    dmax = HMModel.DurationParameters.dmax;
    if ~isfield(HMModel.DurationParameters, 'flag')    %%
        HMModel.DurationParameters.flag = 0;
    end
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
if normflag
    y = zscore(y);                          % Normalize observations
end
if strcmpi(HMModel.type, 'ARHMM') || strcmpi(HMModel.type, 'ARHSMMED') || strcmpi(HMModel.type, 'ARHSMMVT')
    if ~isfield(HMModel, 'DelayARMatrix')
        p = HMModel.ARorder;
        % Build auxiliary matrix of AR predictors
        Yp = zeros(N - p, p);
        for i = 1:N-p
            Yp(i, :) = -fliplr(y(i : i + p - 1));
        end
        HMModel.DelayARMatrix = Yp;
    end
end
logpYZ = LogProbObsAllZ(y, K, HMModel);
if flagtype == 2 || flagtype == 3
    logpDZ = LogProbDurAllZ(K, HMModel);
end
%%
nforw = iIni+1:N;
if flagtype == 1   
    switch Estep
        case 'logsumexp'
            % Compute probabilities of observations for all possible hidden state cases
            auxEM.logpYZ = logpYZ;
            logpzIni = log(HMModel.StateParameters.pi);
            logA = log(HMModel.StateParameters.A);
            alphaEM = zeros(K, N);
            % First iteration
            alphaEM(:, iIni) =  auxEM.logpYZ(:, iIni) + logpzIni;
            %nforw = iIni+1:N;
            % Remaining iterations
            for i = 1:numel(nforw)
                alphaEM(:, nforw(i)) = auxEM.logpYZ(:, nforw(i)) + ...
                    logsumexp(alphaEM(:, nforw(i)-1) + logA, 1)';
            end
            % log-likelihood
            loglike = logsumexp(alphaEM(:, end), 1);
        case 'scaling'
            % Compute probabilities of observations for all possible hidden state cases
            auxEM.pYZ = exp(logpYZ);
            pzIni = HMModel.StateParameters.pi;
            A = HMModel.StateParameters.A;
            alphaEM = zeros(K, N);
            coeff = zeros(1, N);
            % First iteration
            alphaEM(:, iIni) =  auxEM.pYZ(:, iIni) .* pzIni;
            coeff(iIni) = sum(alphaEM(:, iIni));
            alphaEM(:, iIni) = alphaEM(:, iIni)/coeff(iIni);
            %nforw = iIni+1:N;
            % Remaining iterations - alpha
            for i = 1:numel(nforw)
                aux = auxEM.pYZ(:, nforw(i)).*(alphaEM(:, nforw(i)-1)' * A)';
                coeff(nforw(i)) = sum(aux);
                alphaEM(: , nforw(i)) = aux/coeff(nforw(i));
            end
            % log-likelihood
            loglike = sum(log(coeff(iIni:end)));
            auxEM.coeff = coeff;
    end
elseif flagtype == 2
    switch Estep
        case 'logsumexp'
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
                    %logalphaaux = logalphaEMprev(:, 2:dmax);   % get rid of this               
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
                    %logalphaaux = logalphaEMprev(:, 2:dmax); 
                    logalphaEMpos = -inf(K, dmax);
                    logalphaEMpos(:, dmax) = logpYZ(:, nforw(i)) + logpDZ(:, dmax) + logalphaauxdmin;
                    logalphaEMpos(:, 1:dmax-1) = logpYZ(:, nforw(i)) + ...
                        logsumexp(cat(3, logpDZ(:,1:dmax-1) + logalphaauxdmin, logalphaEMprev(:, 2:dmax)), 3); 
                    logalphaEMprev = logalphaEMpos;
                end
            end
            % log-likelihood
            loglike = logsumexp(logsumexp(logalphaEMpos, 2), 1);         
        case 'scaling'
            pYZ = exp(logpYZ);
            pDZ = exp(logpDZ);
            auxEM.pYZ = pYZ;
            auxEM.pDZ = pDZ;
            % Compute probabilities of observations for all possible hidden state cases
            pzIni = HMModel.StateParameters.pi;
            Atrans = HMModel.StateParameters.A(:, :, 1);
            Astay = HMModel.StateParameters.A(:, :, 2);
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
                    %alphaaux = (alphaEMprev(:, 2:dmax)' * Astay)';
                    alphaaux = alphaEMprev(:, 2:dmax);
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
                    aux = zeros(K, dmax);
                    alphaauxdmin = (alphaEMprev(:, 1)'*Atrans)';
                    %alphaaux = (alphaEMprev(:, 2:dmax)' * Astay)';
                    alphaaux = alphaEMprev(:, 2:dmax);
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
    end
elseif flagtype == 3
    distInt = HMModel.DurationParameters.DurationIntervals;
    distIntaux = distInt;
    distIntaux(end) = dmax + 1;
    switch Estep
        case 'logsumexp'
            auxEM.logpYZ = double(logpYZ);
            auxEM.logpDZ = double(logpDZ);
            % Compute probabilities of observations for all possible hidden state cases
            logpzIni = log(HMModel.StateParameters.pi);
            logA = log(HMModel.StateParameters.A);
            if alphaflag
                alphaEM = cell(1, N);    % inside each cell element: states x durations
                alphaEM(:) = {-inf(K, dmax, dmax, 'double')};
                % First iteration
                alphaEM{iIni} = repmat(logpYZ(:, iIni) + logpDZ + logpzIni, 1, 1, dmax);
                logalphaEMprev = alphaEM{iIni};
                % Remaining iterations
                %b2 = zeros(K, dmax-1);               
                idxalpha1 = zeros(K, dmax - 1);
                idxalpha1(:, 1) = (K+1:2*K)';
                for i = 2:dmax-1
                    idxalpha1(:, i) = (dmax + 1)*K + idxalpha1(1, i - 1):...
                        (dmax + 1)*K + idxalpha1(1, i - 1) + K - 1;
                end
                idxalpha2 = zeros(K, dmax - 1);
                idxalpha2(:, 1) = (1:K)';
                for i = 2:dmax - 1
                    idxalpha2(:, i) = (dmax + 1)*K + idxalpha2(1, i - 1):...
                        (dmax + 1)*K + idxalpha2(1, i - 1) + K - 1;
                end
%                 logArep = zeros(K, K, dmax);
%                 for i = 1:numel(distInt)-1
%                     logArep(:, :, distIntaux(i):distIntaux(i+1)-1) = repmat(logA(:, :, i), 1, 1, distIntaux(i+1) - distIntaux(i));
%                 end
                
                logArep2 = zeros(K, K, dmax);
                for i = 1:numel(distInt)-1
                    logArep2(:, :, distIntaux(i):distIntaux(i+1)-1) = repmat(logA(:, :, i)', 1, 1, distIntaux(i+1) - distIntaux(i));
                end
                
                
                %aux1 = -inf(K, K, dmax);
                for i = 1:numel(nforw)
                    %i
                    logalphaEMpos = -inf(K, dmax, dmax);                   
                    alphar_n1 = logalphaEMprev(:, 1, :);
                    %aux1 = alphar_n1 + logArep;
%                     for j = 1:numel(distInt)-1
%                         aux1(:, :, distIntaux(j):distIntaux(j+1)-1) = ...
%                             alphar_n1(:,:, distIntaux(j):distIntaux(j+1)-1) + logA(:, :, j);
%                     end 
                    
                    aux2 = reshape(alphar_n1, 1, K, dmax) + logArep2;
                    logalphaauxdmin = logsumexp(reshape(aux2, K, K*dmax), 2);
                    
                    %logalphaauxdmin = logsumexp(logsumexp(aux1, 3), 1)';        
                    logalphaEMpos(:, dmax, dmax) = logpYZ(:, nforw(i)) + logpDZ(:, dmax) + logalphaauxdmin;                  
                    logalphaEMpos(:, 1:dmax-1,:) = logpYZ(:, nforw(i)) + logalphaEMprev(:, 2:dmax, :);    
                    b1 = logpDZ(:,1:dmax-1) + logalphaauxdmin;
%                     for ii = 1:dmax-1
%                         b2(:,ii) = logalphaEMprev(:, ii+1, ii);
%                     end     
                    b2 = logalphaEMprev(idxalpha1);
                    b3 = logpYZ(:, nforw(i)) + logsumexp(cat(3, b1, b2), 3);  
%                     for ii = 1:dmax-1
%                         logalphaEMpos(:,ii,ii) = b3(:, ii);
%                     end
                    logalphaEMpos(idxalpha2) = b3;
                    logalphaEMprev = logalphaEMpos;
                    alphaEM{nforw(i)} = logalphaEMpos;
                end
            else
            end
            % log-likelihood
            loglike = logsumexp(logsumexp(logsumexp(logalphaEMpos, 1), 2), 3);
        case 'scaling'
    end
end

% if strcmpi(Estep, 'logsumexp')
%     % Compute probabilities of observations for all possible hidden state cases
%     auxEM.logpYZ = logpYZ;
%     logpzIni = log(HMModel.StateParameters.pi);
%     logA = log(HMModel.StateParameters.A);   
%     alphaEM = zeros(K, N);
%     % First iteration
%     alphaEM(:, iIni) =  auxEM.logpYZ(:, iIni) + logpzIni;
%     nforw = iIni+1:N;
%     % Remaining iterations
%     for i = 1:numel(nforw)
%         alphaEM(:, nforw(i)) = auxEM.logpYZ(:, nforw(i)) + ...
%             logsumexp(alphaEM(:, nforw(i)-1) + logA, 1)';
%     end
%     % log-likelihood
%     loglike = logsumexp(alphaEM(:, end), 1);
% elseif strcmpi(Estep, 'scaling')
%     % Compute probabilities of observations for all possible hidden state cases
%     auxEM.pYZ = exp(logpYZ);
%     pzIni = HMModel.StateParameters.pi;
%     A = HMModel.StateParameters.A;   
%     alphaEM = zeros(K, N);
%     coeff = zeros(1, N); 
%     % First iteration
%     alphaEM(:, iIni) =  auxEM.pYZ(:, iIni) .* pzIni;
%     coeff(iIni) = sum(alphaEM(:, iIni));
%     alphaEM(:, iIni) = alphaEM(:, iIni)/coeff(iIni);
%     nforw = iIni+1:N;
%     % Remaining iterations - alpha
%     for i = 1:numel(nforw)
%         aux = auxEM.pYZ(:, nforw(i)).*(alphaEM(:, nforw(i)-1)' * A)';
%         coeff(nforw(i)) = sum(aux);
%         alphaEM(: , nforw(i)) = aux/coeff(nforw(i));
%     end
%     % log-likelihood
%     loglike = sum(log(coeff(iIni:end)));
%     auxEM.coeff = coeff;
% else
%     disp('Error')                   % TODO
% end

end