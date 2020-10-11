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
if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'ARHMM')
    A = zeros(K, K);
    for iSeq = 1:nSeq
        for k = 1:K        
            aux = squeeze(HMModel.EM(iSeq).xi(k, :, iIni+1:end));
            A(k, :) = A(k, :) + sum(aux,2)';
        end
    end
    A = A./sum(A,2);
    HMModel.StateParameters.A(:, :, 1) = A;
elseif strcmpi(HMModel.type, 'HSMMED') || strcmpi(HMModel.type, 'ARHSMMED')
    A = zeros(K, K);
    for iSeq = 1:nSeq
        for k = 1:K
            A(k, :) = A(k, :) + HMModel.EM(iSeq).sumxi(k, :);
        end
    end
    A = A./sum(A,2);
    HMModel.StateParameters.A(:, :, 1) = A;
elseif strcmpi(HMModel.type, 'HSMMVT') || strcmpi(HMModel.type, 'ARHSMMVT')
    distInt = HMModel.DurationParameters.DurationIntervals;
    A = zeros(K, K, numel(distInt)-1);
    for iSeq = 1:nSeq
        for j = 1:numel(distInt)-1
            for k = 1:K
                A(k, :, j) = A(k, :, j) + HMModel.EM(iSeq).sumxi(k, :, j);
            end
        end
    end
    A = A./sum(A,2);
    A(isnan(A)) = 0;
    HMModel.StateParameters.A = A;
end

%% Observation parameters
if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'HSMMED') || strcmpi(HMModel.type, 'HSMMVT')
    switch HMModel.ObsParameters.model
        case 'Gaussian'
            % Mean and standard deviation of emissions
            sumgammk = zeros(nSeq, K);
            mukaux = zeros(nSeq, K);
            for iSeq = 1:nSeq
                for k = 1:K
                    sumgammk(iSeq, k) = sum(HMModel.EM(iSeq).gamma(k, iIni:end));
                    mukaux(iSeq, k) = sum(HMModel.EM(iSeq).gamma(k, :).*ySeq{iSeq});
                end
            end
            HMModel.ObsParameters.mu = sum(mukaux, 1)./sum(sumgammk, 1);
            sigkaux = zeros(nSeq, K);
            for iSeq = 1:nSeq
                for k = 1:K
                    sigkaux(iSeq, k) = ...
                        sum(HMModel.EM(iSeq).gamma(k, :).*(ySeq{iSeq} - HMModel.ObsParameters.mu(k)).^2);
                end
            end
            HMModel.ObsParameters.sigma = sqrt(sum(sigkaux, 1)./sum(sumgammk, 1));
        case 'Generalizedt'
            sumgammk = zeros(nSeq, K);
            sumgammwnk = zeros(nSeq, K);
            mukaux = zeros(nSeq, K);
            for iSeq = 1:nSeq
                for k = 1:K
                    sumgammk(iSeq, k) = sum(HMModel.EM(iSeq).gamma(k, :));
                    sumgammwnk(iSeq, k) = HMModel.EM(iSeq).gamma(k, :)*HMModel.EM(iSeq).Etau(k, :)';
                    mukaux(iSeq, k) = sum(HMModel.EM(iSeq).gamma(k, :).*HMModel.EM(iSeq).Etau(k, :).*ySeq{iSeq});
                end
            end
            HMModel.ObsParameters.mu = sum(mukaux, 1)./sum(sumgammwnk, 1);
            sigkaux = zeros(nSeq, K);
            for iSeq = 1:nSeq
                for k = 1:K
                    sigkaux(iSeq, k) = sum(HMModel.EM(iSeq).gamma(k, :).*...
                        HMModel.EM(iSeq).Etau(k, :).*(ySeq{iSeq} - HMModel.ObsParameters.mu(k)).^2);
                end
            end
            HMModel.ObsParameters.sigma = sqrt(sum(sigkaux, 1)./sum(sumgammk, 1));
            %HMModel.ObsParameters.sigma = sqrt(sum(sigkaux, 1)./sum(sumgammwnk, 1));
            % Degrees of freedom
            auxnu = zeros(1, K);
            for iSeq = 1:nSeq
                for k = 1:K
                    auxnu(k) = auxnu(k) + (sum(HMModel.EM(iSeq).gamma(k, :).*...
                        (log(HMModel.EM(iSeq).Etau(k, :)) - HMModel.EM(iSeq).Etau(k, :))))/...
                        sumgammk(iSeq, k);
                end
            end
            auxnu = auxnu./nSeq;
            for k = 1:K
                c = 1 + psi((HMModel.ObsParameters.nu(k) + 1)/2) - log((HMModel.ObsParameters.nu(k) + 1)/2) + auxnu(k);
                fun = @(x) F(x, c);
                HMModel.ObsParameters.nu(k) = fzero(fun, [0.5 100]);
            end
        case 'MultivariateGaussian'
            % mean vectors andcovariance matrices
            d = size(ySeq{1}, 1);
            sumgammk = zeros(nSeq, K);
            mukaux = zeros(nSeq, d, K);
            for iSeq = 1:nSeq
                for k = 1:K
                    sumgammk(iSeq, k) = sum(HMModel.EM(iSeq).gamma(k, iIni:end));
                    mukaux(iSeq, :, k) = sum(HMModel.EM(iSeq).gamma(k, :).*ySeq{iSeq}, 2);
                end
            end
            HMModel.ObsParameters.mu = squeeze(sum(mukaux, 1))./sum(sumgammk, 1);
            sigkaux = zeros(d, d, K);
            for iSeq = 1:nSeq
                N = HMModel.N(iSeq);
                for k = 1:K
                    aux = ySeq{iSeq} - HMModel.ObsParameters.mu(:, k);
                    b1 = reshape(aux, d, 1, N);
                    b2 = reshape(aux, 1, d, N);
                    a = reshape(HMModel.EM(iSeq).gamma(k, :), 1, 1, N) .* (b1.*b2);
                    sigkaux(:, :, k) = sigkaux(:, :, k) + ...
                        sum(a, 3);
                    
                    %                 sigkaux(iSeq, k) = ...
                    %                     sum(HMModel.EM(iSeq).gamma(k, :).*(ySeq{iSeq} - HMModel.ObsParameters.mu(k)).^2);
                end
            end
            HMModel.ObsParameters.sigma = sigkaux./reshape(sum(sumgammk, 1),1,1,K);
    end
elseif strcmpi(HMModel.type, 'ARHMM') || strcmpi(HMModel.type, 'ARHSMMED') || strcmpi(HMModel.type, 'ARHSMMVT')
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
            if HMModel.robustMstep
                mdl = fitlm(Xtil', Ytil', 'Intercept', false, 'RobustOpts', 'welsch'); 
            else
                mdl = fitlm(Xtil', Ytil', 'Intercept', false);
            end
            HMModel.ObsParameters.meanParameters(k).Coefficients = table2array(mdl.Coefficients(:,1));
        end
        % else - Non-linear models of the mean - TODO?
    end
    
        clf
        [HH, FF] = freqz(1, [1; HMModel.ObsParameters.meanParameters(1).Coefficients], 1024, HMModel.Fs);
        plot(FF, 20*log10(abs(HH)))
        hold on
        [HH, FF] = freqz(1, [1; HMModel.ObsParameters.meanParameters(2).Coefficients], 1024, HMModel.Fs);
        plot(FF, 20*log10(abs(HH)), 'r')
        title(['loglike = ' num2str(HMModel.loglike(end))])
        ylim([-15 20])
        pause(0.1)
    
    
%     for k = 1:K
%         Ckv = zeros(1, auxdim);
%         Ytil = zeros(auxdim, 1);
%         idxini = 1;
%         for iSeq = 1:nSeq
%             idxend = idxini + HMModel.N(iSeq) - p - 1;
%             Ckv(idxini:idxend) = sqrt(HMModel.EM(iSeq).gamma(k, iIni: end));
%             Ytil(idxini:idxend) = (Ckv(idxini:idxend) .* ySeq{iSeq}(iIni:end))';
%             idxini = idxend + 1;
%         end
%         Xtil = Ckv .* vertcat(HMModel.DelayARMatrix.Seq)';
%         if strcmpi(HMModel.ObsParameters.meanModel, 'Linear')
%             if HMModel.robustMstep
%                 mdl = fitlm(Xtil', Ytil', 'Intercept', false, 'RobustOpts', 'welsch'); 
%             else
%                 mdl = fitlm(Xtil', Ytil', 'Intercept', false);
%             end
%             HMModel.ObsParameters.meanParameters(k).Coefficients = table2array(mdl.Coefficients(:,1));
%             % else - Non-linear models of the mean - TODO?
%         end
%     end
    %HMModel.ObsParameters.ARcoeff = ak;
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
end
% Duration for HSMM
if strcmpi(HMModel.type, 'HSMMED') || strcmpi(HMModel.type, 'ARHSMMED') ||...
        strcmpi(HMModel.type, 'HSMMVT') || strcmpi(HMModel.type, 'ARHSMMVT')
    HMModel = OptimizeDuration(HMModel);
end

end