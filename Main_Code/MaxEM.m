function HMModel = MaxEM(ySeq, HMModel)
%MAXEM Maximization step (M-step) of EM algorithm
%   HMModel = MAXEM(ySeq, HMModel) updates learnable parameters of HMModel
%   according to maximum likelihood using the sequences in ySeq.
%Author: Carlos Loza (carlos.loza@utexas.edu)
%% Hyperparameters and general parameters
iIni = HMModel.ARorder + 1;             % initial time sample for learning
K = HMModel.StateParameters.K;          % number of regimes/modes
p = HMModel.ARorder;                    % autoregressive order
dmax = HMModel.DurationParameters.dmax; % maximum duration of regimes (D in paper)
nSeq = HMModel.nSeq;                    % number of sequences in ySeq
F = @(x, c) -psi(x/2) + log(x/2) + c;   % auxiliary function for estimation of degrees of freedom
%% Initial state probabilities - eq (14) in paper
aux = cell2mat(arrayfun(@(x) x.gamma(:, iIni), HMModel.EM, 'UniformOutput', false));
HMModel.StateParameters.pi = double(sum(aux,2)/sum(aux(:)));
%% Transition probabilities - eq (15) in paper. 
A = zeros(K, K);
for iSeq = 1:nSeq
    for k = 1:K
        % Easier/faster unnormalized implementation
        A(k, :) = A(k, :) + HMModel.EM(iSeq).sumxi(k, :);
    end
end
A = A./sum(A,2);                        % Normalization
HMModel.StateParameters.A = A;
%% Observation parameters - AR coefficients, scales and degrees of freedom
auxdim = sum(horzcat(HMModel.N)) - p*numel(horzcat(HMModel.N));
% AR coefficients
% Compute weigths for AR coefficients estimation
Ckv = zeros(K, auxdim);             % Weight matrix (W in paper)
for k = 1:K
    idxini = 1;
    for iSeq = 1:nSeq
        idxend = idxini + HMModel.N(iSeq) - p - 1;
        Ckv(k, idxini:idxend) = sqrt(HMModel.EM(iSeq).gamma(k, iIni: end).*...
            HMModel.EM(iSeq).Etau(k, iIni: end));
        idxini = idxend + 1;
    end
end
for k = 1:K
    Ytil = zeros(auxdim, 1);
    idxini = 1;
    for iSeq = 1:nSeq
        idxend = idxini + HMModel.N(iSeq) - p - 1;
        Ytil(idxini:idxend) = (Ckv(k, idxini:idxend) .* ySeq{iSeq}(iIni:end))';
        idxini = idxend + 1;
    end
    Xtil = Ckv(k,:) .* vertcat(HMModel.DelayARMatrix.Seq)';
    mdl = fitlm(Xtil', Ytil', 'Intercept', false);  % eq (17) in paper
    HMModel.ObsParameters.meanParameters(k).Coefficients = table2array(mdl.Coefficients(:,1));
end
% Scale parameters of generalized t additive noise
sumgammk = zeros(nSeq, K);
sumgammwnk = zeros(nSeq, K);
sigkaux = zeros(nSeq, K);
for iSeq = 1:nSeq
    for k = 1:K
        muk = (HMModel.DelayARMatrix(iSeq).Seq * HMModel.ObsParameters.meanParameters(k).Coefficients)';
        sumgammk(iSeq, k) = sum(HMModel.EM(iSeq).gamma(k, iIni:end));
        sumgammwnk(iSeq, k) = HMModel.EM(iSeq).gamma(k, iIni:end)*HMModel.EM(iSeq).Etau(k, iIni:end)';
        sigkaux(iSeq, k) = sum(HMModel.EM(iSeq).gamma(k, iIni:end).*...
            HMModel.EM(iSeq).Etau(k, iIni:end).*(ySeq{iSeq}(iIni:end) - muk).^2);   % eq (18) in paper
    end
end
HMModel.ObsParameters.sigma = sqrt(sum(sigkaux, 1)./sum(sumgammk, 1));
% Degrees of freedom of generalized t additive noise
auxnu = zeros(1, K);
for iSeq = 1:nSeq
    for k = 1:K
        auxnu(k) = auxnu(k) + (sum(HMModel.EM(iSeq).gamma(k, iIni:end).*...
            (log(HMModel.EM(iSeq).Etau(k, iIni:end)) - HMModel.EM(iSeq).Etau(k, iIni:end))))/...
            sumgammk(iSeq, k);
    end
end
auxnu = auxnu./nSeq;
for k = 1:K
    % eq (19) in paper and solver
    % eq (19) uses eq (13) plus optimization (partial derivative equal
    % to zero) and further simplifications
    c = 1 + psi((HMModel.ObsParameters.nu(k) + 1)/2) - log((HMModel.ObsParameters.nu(k) + 1)/2) + auxnu(k);
    fun = @(x) F(x, c);
    HMModel.ObsParameters.nu(k) = fzero(fun, [0.1 100]);
end
%% Duration parameters - eq (16) in paper
aux = zeros(K, dmax);
for iSeq = 1:nSeq
    xi = HMModel.EM(iSeq).xi(:, iIni+1:end);
    for k = 1:K
        % Easier/faster unnormalized implementation
        aux(k, :) = aux(k, :) + (HMModel.EM(iSeq).eta_iIni(k, :) + ...
            full(sum(cell2mat(cellfun(@(x) x(k, :), xi, 'UniformOutput', false)'), 1)));
    end
end
% Normalization
HMModel.DurationParameters.PNonParametric = aux./sum(aux, 2);
end