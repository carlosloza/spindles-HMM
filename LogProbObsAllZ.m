%% Probabilities of observations for all possible hidden state cases
function logpYZ = LogProbObsAllZ(y, K, HMModel)
log2pi = log(2*pi);
if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'HSMMED') || strcmpi(HMModel.type, 'HSMMVT')
    switch HMModel.ObsParameters.model
        case 'Gaussian'
            logpYZ = zeros(K, numel(y));
            for k = 1:K
                logpYZ(k, :) = -0.5*log2pi - log(HMModel.ObsParameters.sigma(k))...
                    - ((y - HMModel.ObsParameters.mu(k)).^2)/(2*HMModel.ObsParameters.sigma(k).^2);
            end
        case 'Generalizedt'
            logpYZ = zeros(K, numel(y));                     
            for k = 1:K
                muk = HMModel.ObsParameters.mu(k);
                sigk = HMModel.ObsParameters.sigma(k);
                nuk = HMModel.ObsParameters.nu(k);
                logpYZ(k, :) = log(pdf('tLocationScale', y, muk, sigk, nuk));
            end
        case 'MultivariateGaussian'
            logpYZ = zeros(K, size(y, 2));
            for k = 1:K
                logpYZ(k, :) = log(mvnpdf(y', HMModel.ObsParameters.mu(:,k)', ...
                    HMModel.ObsParameters.sigma(:,:,k)'))';
            end
    end
elseif strcmpi(HMModel.type, 'ARHMM') || strcmpi(HMModel.type, 'ARHSMMED') || strcmpi(HMModel.type, 'ARHSMMVT')
    iIni = HMModel.ARorder + 1;
    logpYZ = -inf(K, numel(y));
    switch HMModel.ObsParameters.meanModel
        case 'Linear'                                           
            switch HMModel.ObsParameters.model
                case 'Gaussian'
                    for k = 1:K                       
                        muk = (HMModel.DelayARMatrix * ...
                            HMModel.ObsParameters.meanParameters(k).Coefficients)';
%                         muk = (HMModel.DelayARMatrix * ...
%                             table2array(HMModel.ObsParameters.meanModel(k).model.Coefficients(:,1)))';
                        sigk = HMModel.ObsParameters.sigma(k);
                        logpYZ(k, iIni:end) = -0.5*log2pi - log(sigk)...
                            - ((y(iIni:end) - muk).^2)/(2*sigk.^2);
                    end
                case 'Generalizedt'
                    for k = 1:K
                        muk = (HMModel.DelayARMatrix * ...
                            HMModel.ObsParameters.meanParameters(k).Coefficients)';
%                         muk = (HMModel.DelayARMatrix * ...
%                             table2array(HMModel.ObsParameters.meanModel(k).model.Coefficients(:,1)))';
                        sigk = HMModel.ObsParameters.sigma(k);
                        nuk = HMModel.ObsParameters.nu(k);
                        logpYZ(k, iIni:end) = log(pdf('tLocationScale', y(iIni:end) - muk, 0, sigk, nuk));
                    end
            end
        case 'SVM'
            switch HMModel.ObsParameters.model
                case 'Gaussian'
                    for k = 1:K
                        muk = predict(HMModel.ObsParameters.meanModel(k).model, HMModel.DelayARMatrix)';
                        sigk = HMModel.ObsParameters.sigma(k);
                        logpYZ(k, iIni:end) = -0.5*log2pi - log(sigk)...
                            - ((y(iIni:end) - muk).^2)/(2*sigk.^2);
                    end
                case 'Generalizedt'
                    for k = 1:K
                        muk = predict(HMModel.ObsParameters.meanModel(k).model, HMModel.DelayARMatrix)';
                        sigk = HMModel.ObsParameters.sigma(k);
                        nuk = HMModel.ObsParameters.nu(k);
                        logpYZ(k, iIni:end) = log(pdf('tLocationScale', y(iIni:end) - muk, 0, sigk, nuk));
                    end
            end
        case 'MLP'
            switch HMModel.ObsParameters.model
                case 'Gaussian'
                    for k = 1:K
                        muk = HMModel.ObsParameters.meanModel(k).model(HMModel.DelayARMatrix');
                        sigk = HMModel.ObsParameters.sigma(k);
                        logpYZ(k, iIni:end) = -0.5*log2pi - log(sigk)...
                            - ((y(iIni:end) - muk).^2)/(2*sigk.^2);
                    end
                case 'Generalizedt'
                    for k = 1:K
                        muk = HMModel.ObsParameters.meanModel(k).model(HMModel.DelayARMatrix');
                        sigk = HMModel.ObsParameters.sigma(k);
                        nuk = HMModel.ObsParameters.nu(k);
                        logpYZ(k, iIni:end) = log(pdf('tLocationScale', y(iIni:end) - muk, 0, sigk, nuk));
                    end
            end
    end
    
    %meanK = (DelayARMatrix * HMModel.ObsParameters.ARcoeff)'; 
    %for k = 1:K
        %logpYZ(k, iIni:end) = -0.5*log2pi - log(HMModel.ObsParameters.sigma(k))...
        %    - ((y(iIni:end) - meanK(k, :)).^2)/(2*HMModel.ObsParameters.sigma(k).^2);
    %end
end
end