%% Log of probabilities of durations given hidden state and d_(n-1) = 1, i.e. hidden state transition
function logpDZ = LogProbDurGivenZ(K, HSMModel)
if HSMModel.DurationParameters.flag
    logpDZ = log(HSMModel.DurationParameters.PNonParametric);
else
    if strcmpi(HSMModel.DurationParameters.model, 'Poisson') && ~HSMModel.DurationParameters.flag
        if ~isfield(HSMModel.DurationParameters, 'dlogfact')
            HSMModel.DurationParameters.dlogfact = log(factorial(1:HSMModel.DurationParameters.dmax));
        end     
        logpDZ = -inf(K, HSMModel.DurationParameters.dmax);
        dEf = 1:HSMModel.DurationParameters.dmax;
        for k = 1:K           
            logpDZ(k, dEf(1):dEf(end)) = log(HSMModel.DurationParameters.lambda(k))*dEf - ...
                HSMModel.DurationParameters.lambda(k) - HSMModel.DurationParameters.dlogfact;
        end
    elseif strcmpi(HSMModel.DurationParameters.model, 'Gaussian') && ~HSMModel.DurationParameters.flag
        logpDZ = -inf(K, HSMModel.DurationParameters.dmax);
        log2pi = log(2*pi);
        dEf = 1:HSMModel.DurationParameters.dmax;
        for k = 1:K
            logpDZ(k, :) = -0.5*log2pi - log(HSMModel.DurationParameters.sigma(k))...
             - ((dEf - HSMModel.DurationParameters.mu(k)).^2)/(2*HSMModel.DurationParameters.sigma(k).^2);
        end
    elseif strcmpi(HSMModel.DurationParameters.model, 'NonParametric')
        logpDZ = log(HSMModel.DurationParameters.PNonParametric);
    end
end

end