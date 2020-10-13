%% Maximization step of EM
% Batch implementation - ySeq is a cell now
function HMModel = MaxEM(ySeq, HMModel)
iIni = HMModel.ARorder + 1;
K = HMModel.StateParameters.K;
nSeq = HMModel.nSeq;
F = @(x, c) -psi(x/2) + log(x/2) + c;
%% Initial state probabilities
aux = cell2mat(arrayfun(@(x) x.gamma(:, iIni), HMModel.EM, 'UniformOutput', false));
HMModel.StateParameters.pi = full(sum(aux,2)/sum(aux(:)));
%% Transition probabilities
A = zeros(K, K);
for iSeq = 1:nSeq
    for k = 1:K
        A(k, :) = A(k, :) + HMModel.EM(iSeq).sumxi(k, :);
    end
end
A = A./sum(A,2);
HMModel.StateParameters.A(:, :, 1) = A;

%% Observation parameters
p = HMModel.ARorder;
% AR coefficients and standard deviation of emissions
auxdim = sum(horzcat(HMModel.N)) - p*numel(horzcat(HMModel.N));
if strcmpi(HMModel.ObsParameters.meanModel, 'Linear')
    % Compute weigths for AR coefficients estimation
    Ckv = zeros(K, auxdim);
    for k = 1:K
        idxini = 1;
        switch HMModel.ObsParameters.model
            case 'Gaussian'
                for iSeq = 1:nSeq
                    idxend = idxini + HMModel.N(iSeq) - p - 1;
                    Ckv(k, idxini:idxend) = sqrt(HMModel.EM(iSeq).gamma(k, iIni: end));
                    idxini = idxend + 1;
                end
            case 'Generalizedt'
                for iSeq = 1:nSeq
                    idxend = idxini + HMModel.N(iSeq) - p - 1;
                    Ckv(k, idxini:idxend) = sqrt(HMModel.EM(iSeq).gamma(k, iIni: end).*...
                        HMModel.EM(iSeq).Etau(k, iIni: end));
                    idxini = idxend + 1;
                end
        end
    end
    % These next lines will only work for Gaussian and Generalizedt
    % observation models
    for k = 1:K
        Ytil = zeros(auxdim, 1);
        idxini = 1;
        for iSeq = 1:nSeq
            idxend = idxini + HMModel.N(iSeq) - p - 1;
            Ytil(idxini:idxend) = (Ckv(k, idxini:idxend) .* ySeq{iSeq}(iIni:end))';
            idxini = idxend + 1;
        end
        Xtil = Ckv(k,:) .* vertcat(HMModel.DelayARMatrix.Seq)';        
        mdl = fitlm(Xtil', Ytil', 'Intercept', false);
        HMModel.ObsParameters.meanParameters(k).Coefficients = table2array(mdl.Coefficients(:,1));
    end
    % else - Non-linear models of the mean - TODO?
end

% Sigma and nu (for Generalizedt)
if strcmpi(HMModel.ObsParameters.meanModel, 'Linear')
    switch HMModel.ObsParameters.model
        case 'Gaussian'
            sumgammk = zeros(nSeq, K);
            sigkaux = zeros(nSeq, K);
            for iSeq = 1:nSeq
                for k = 1:K
                    muk = (HMModel.DelayARMatrix(iSeq).Seq * HMModel.ObsParameters.meanParameters(k).Coefficients)';
                    sumgammk(iSeq, k) = sum(HMModel.EM(iSeq).gamma(k, iIni:end));
                    sigkaux(iSeq, k) = sum(HMModel.EM(iSeq).gamma(k, iIni:end) .* (ySeq{iSeq}(iIni:end) - muk).^2);
                end
            end
            HMModel.ObsParameters.sigma = sqrt(sum(sigkaux, 1)./sum(sumgammk, 1));
        case 'Generalizedt'
            sumgammk = zeros(nSeq, K);
            sumgammwnk = zeros(nSeq, K);
            sigkaux = zeros(nSeq, K);
            for iSeq = 1:nSeq
                for k = 1:K
                    muk = (HMModel.DelayARMatrix(iSeq).Seq * HMModel.ObsParameters.meanParameters(k).Coefficients)';
                    sumgammk(iSeq, k) = sum(HMModel.EM(iSeq).gamma(k, iIni:end));
                    sumgammwnk(iSeq, k) = HMModel.EM(iSeq).gamma(k, iIni:end)*HMModel.EM(iSeq).Etau(k, iIni:end)';
                    sigkaux(iSeq, k) = sum(HMModel.EM(iSeq).gamma(k, iIni:end).*...
                        HMModel.EM(iSeq).Etau(k, iIni:end).*(ySeq{iSeq}(iIni:end) - muk).^2);
                end
            end
            HMModel.ObsParameters.sigma = sqrt(sum(sigkaux, 1)./sum(sumgammk, 1));
            % Degrees of freedom
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
                c = 1 + psi((HMModel.ObsParameters.nu(k) + 1)/2) - log((HMModel.ObsParameters.nu(k) + 1)/2) + auxnu(k);
                fun = @(x) F(x, c);
                HMModel.ObsParameters.nu(k) = fzero(fun, [0.1 100]);
            end
    end
    % else - Non-linear models of the mean - TODO?
end
%HMModel.ObsParameters.sigma = sigk;
% Duration for HSMM
HMModel = OptimizeDuration(HMModel);


end