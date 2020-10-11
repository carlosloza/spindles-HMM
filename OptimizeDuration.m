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
            
%             aux(k, :) = aux(k, :) + (HSMModel.EM(iSeq).eta{iIni}(k, :) + ...
%                 full(sum(cell2mat(cellfun(@(x) x(k, :), xi, 'UniformOutput', false)'), 1)));
            
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
elseif strcmpi(HSMModel.DurationParameters.model, 'Poisson')
    dEf = 1:dmax;  
    num = zeros(1, K);
    den = zeros(1, K);
    for iSeq = 1:nSeq
        xiEM = HSMModel.EM(iSeq).xi(iIni+1:end);
%         num = num + dEf*HSMModel.EM(iSeq).eta{iIni}' + ...
%             sum(cell2mat(cellfun(@(x) ((1:size(x,2))*x')', xiEM, 'UniformOutput', false)), 2)';
        num = num + dEf*HSMModel.EM(iSeq).eta_iIni' + ...
            sum(cell2mat(cellfun(@(x) ((1:size(x,2))*x')', xiEM, 'UniformOutput', false)), 2)';
        
%         den = den + full(sum(HSMModel.EM(iSeq).eta{iIni},2))' + ...
%             full(sum(cell2mat(cellfun(@(x) sum(x,2), xiEM, 'UniformOutput', false)), 2))';
        den = den + full(sum(HSMModel.EM(iSeq).eta_iIni,2))' + ...
            full(sum(cell2mat(cellfun(@(x) sum(x,2), xiEM, 'UniformOutput', false)), 2))';
    end
    HSMModel.DurationParameters.lambda = num./den;
elseif strcmpi(HSMModel.DurationParameters.model, 'Gaussian')
    dEf = 1:dmax;
    % mean
    num = zeros(1, K);
    den = zeros(1, K);
    for iSeq = 1:nSeq
        xiEM = HSMModel.EM(iSeq).xi(iIni+1:end);
%         num = num + dEf*HSMModel.EM(iSeq).eta{iIni}' + ...
%             sum(cell2mat(cellfun(@(x) ((1:size(x,2))*x')', xiEM, 'UniformOutput', false)), 2)';

        num = num + dEf*HSMModel.EM(iSeq).eta_iIni' + ...
            sum(cell2mat(cellfun(@(x) ((1:size(x,2))*x')', xiEM, 'UniformOutput', false)), 2)';
        
%         den = den + full(sum(HSMModel.EM(iSeq).eta{iIni},2))' + ...
%             full(sum(cell2mat(cellfun(@(x) sum(x,2), xiEM, 'UniformOutput', false)), 2))';

        den = den + full(sum(HSMModel.EM(iSeq).eta_iIni,2))' + ...
            full(sum(cell2mat(cellfun(@(x) sum(x,2), xiEM, 'UniformOutput', false)), 2))';
    end
    HSMModel.DurationParameters.mu = num./den;
    % sigma
    aux = zeros(1, K);
    for iSeq = 1:nSeq
        xiEM = HSMModel.EM(iSeq).xi(iIni+1:end);
        for k = 1:K
            muk = HSMModel.DurationParameters.mu(k);
%             aux(k) = aux(k) + ((dEf - muk).^2)*HSMModel.EM(iSeq).eta{iIni}(k,:)' + ...
%                 sum(cell2mat(cellfun(@(x) (((dEf - muk).^2)*x(k,:)')'...
%                 ,xiEM, 'UniformOutput', false)), 2)';
            aux(k) = aux(k) + ((dEf - muk).^2)*HSMModel.EM(iSeq).eta_iIni(k,:)' + ...
                sum(cell2mat(cellfun(@(x) (((dEf - muk).^2)*x(k,:)')'...
                ,xiEM, 'UniformOutput', false)), 2)';
        end
    end
%     aux = zeros(1, K);
%     for k = 1:K
%         muk = HSMModel.DurationParameters.mu(k);
%         aux(k) = ((dEf - muk).^2)*HSMModel.EM.eta{iIni}(k,:)' + ...
%             sum(cell2mat(cellfun(@(x) (((dEf - muk).^2)*x(k,:)')'...
%             ,xiEM, 'UniformOutput', false)), 2)';
%     end
    HSMModel.DurationParameters.sigma = sqrt(aux./den);
end
end