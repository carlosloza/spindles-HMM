function HSMModel = OptimizeDuration(HSMModel)
K = HSMModel.StateParameters.K;
dmax = HSMModel.DurationParameters.dmax;
iIni = HSMModel.ARorder + 1;
nSeq = HSMModel.nSeq;
if strcmpi(HSMModel.DurationParameters.model, 'NonParametric') || HSMModel.DurationParameters.flag
    aux = zeros(K, dmax);
    for iSeq = 1:nSeq
        xi = HSMModel.EM(iSeq).xi(:, iIni+1:end);
        for k = 1:K
            aux(k, :) = aux(k, :) + (HSMModel.EM(iSeq).eta_iIni(k, :) + ...
                full(sum(cell2mat(cellfun(@(x) x(k, :), xi, 'UniformOutput', false)'), 1)));          
        end
    end
    HSMModel.DurationParameters.PNonParametric = aux./sum(aux, 2);
    if ~strcmpi(HSMModel.DurationParameters.model, 'NonParametric')
        % Estimate initial values for parametric model - MIGHT HAVE TO BE
        % ROBUST INITIAL ESTIMATES
        if strcmpi(HSMModel.DurationParameters.model, 'Poisson')
            % Rate
            HSMModel.DurationParameters.lambda = (1:dmax)*...
                (HSMModel.DurationParameters.PNonParametric)';
        elseif strcmpi(HSMModel.DurationParameters.model, 'Gaussian')
            % Mean
            HSMModel.DurationParameters.mu = (1:dmax)*...
                (HSMModel.DurationParameters.PNonParametric)';
            % Standard deviation
            for k = 1:K
                HSMModel.DurationParameters.sigma(k) = ...
                sqrt((((1:dmax) - HSMModel.DurationParameters.mu(k)).^2)*...
                    (HSMModel.DurationParameters.PNonParametric(k, :))');
            end            
        end
        HSMModel.DurationParameters.flag = 0;
    end
end
end