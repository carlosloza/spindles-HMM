function HMModel = GeneralizedtEstep(ySeq, HMModel)
HMModel.EM = rmfield(HMModel.EM, {'Etau'});
nSeq = HMModel.nSeq;
K = HMModel.StateParameters.K;
iIni = HMModel.ARorder + 1;
for iSeq = 1:nSeq
    N = HMModel.N(iSeq);
    wnk = zeros(K, N);
    if strcmpi(HMModel.type, 'ARHMM') || strcmpi(HMModel.type, 'ARHSMMED') || strcmpi(HMModel.type, 'ARHSMMVT')
        for k = 1:K
            nuk = HMModel.ObsParameters.nu(k);
            muk = (HMModel.DelayARMatrix(iSeq).Seq * ...
                HMModel.ObsParameters.meanParameters(k).Coefficients)';
            deltk = ((ySeq{iSeq}(iIni:end) - muk).^2)/...
                HMModel.ObsParameters.sigma(k)^2;
            wnk(k, iIni:end) = (nuk + 1)./(nuk + deltk);
        end       
    elseif strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'HSMMED')
        for k = 1:K
            nuk = HMModel.ObsParameters.nu(k);
            deltk = ((ySeq{iSeq}(iIni:end) - HMModel.ObsParameters.mu(k)).^2)/...
                HMModel.ObsParameters.sigma(k)^2;
            wnk(k, :) = (nuk + 1)./(nuk + deltk);
        end
    end  
    HMModel.EM(iSeq).Etau = single(wnk);
end
end