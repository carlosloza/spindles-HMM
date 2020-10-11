%% Initial conditions
function HMModel = InitialConditions(ySeq, HMModel)
nSeq = HMModel.nSeq;
K = HMModel.StateParameters.K;
if ~isfield(HMModel, 'ObsParameters')
    if size(ySeq{1}, 1) == 1
            % Default univariate Gaussian model for observations
            HMModel.ObsParameters.model = 'Gaussian';
        else
            % Default multivariate Gaussian model for observations
            HMModel.ObsParameters.model = 'MultivariateGaussian';
    end
end
%% HMM, HSMMED and HSMMVT
if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'HSMMED') || strcmpi(HMModel.type, 'HSMMVT')
    % Parameters of observation model, i.e. emissions
    if size(ySeq{1}, 1) == 1
        % Univariate observations
        yall = cell2mat(ySeq);
        switch HMModel.ObsParameters.model
            case 'Gaussian'
                [idx, C] = kmeans(yall', K, 'Replicates', 10);
                HMModel.ObsParameters.mu = C';
                for k = 1:K
                    HMModel.ObsParameters.sigma(1, k) = std(yall(idx == k), 1);
                end
            case 'Generalizedt'
                [idx, ~] = kmeans(yall', K, 'Replicates', 10, 'Distance', 'cityblock');     % more robust that euclidean distance
                for k = 1:K
                    pd = fitdist(yall(idx == k)', 'tLocationScale');
                    HMModel.ObsParameters.mu(k) = pd.mu;
                    HMModel.ObsParameters.sigma(k) = pd.sigma;
                    HMModel.ObsParameters.nu(k) = pd.nu;
                end
        end
        disp('Initial conditions done')
    else
        % Multivariate observations
        yall = cell2mat(ySeq);
        switch HMModel.ObsParameters.model
            case 'MultivariateGaussian'
                [idx, C] = kmeans(yall', K, 'Replicates', 10);
                disp('Initial conditions done')
                HMModel.ObsParameters.mu = C';
                for k = 1:K
                    aux = yall(:, idx == k) - HMModel.ObsParameters.mu(:,k);
                    b1 = reshape(aux, size(aux,1), 1, size(aux,2));
                    b2 = reshape(aux, 1, size(aux,1), size(aux,2));
                    a = b1.*b2;
                    HMModel.ObsParameters.sigma(:, :, k) = sum(a, 3)./size(aux, 2);
                end
        end
    end
    if isfield(HMModel, 'ARorder')
        warning('Vanilla HMM is not autoregressive! This parameter is ignored')
    end
    HMModel.ARorder = 0;
end

%% ARHMM, ARHSMMED and ARHSMMVT
if strcmpi(HMModel.type, 'ARHMM') || strcmpi(HMModel.type, 'ARHSMMED') || strcmpi(HMModel.type, 'ARHSMMVT')
    if ~isfield(HMModel.ObsParameters, 'meanModel')
        % Default - Linear AR model
        HMModel.ObsParameters.meanModel = 'Linear';
    end
    p = HMModel.ARorder;
    % Build auxiliary matrix of AR predictors
    sigmaIniaux = round(mean(horzcat(HMModel.N))/100);
    %sigmaIniaux = 4;
    Yp = zeros(sum(horzcat(HMModel.N)) - nSeq*p, p);
    Ypini = zeros(sigmaIniaux*p*nSeq, p);
    Yini = zeros(sigmaIniaux*p*nSeq, 1);
    idxini = 1;
    for iSeq = 1:nSeq
        y = ySeq{iSeq};
        N = HMModel.N(iSeq);
        idxend = idxini + N - p - 1;
        Ypaux = HMModel.DelayARMatrix(iSeq).Seq;    
        Yp(idxini:idxend, :) = Ypaux;
        idxini = idxend + 1;
        Ypini((iSeq - 1)*sigmaIniaux*p + 1: iSeq*sigmaIniaux*p, :) = ...
            Ypaux(1:sigmaIniaux*p, :);
        Yini((iSeq - 1)*sigmaIniaux*p + 1: iSeq*sigmaIniaux*p) = ...
            y(p+1:(sigmaIniaux + 1)*p)';
    end
%     for iSeq = 1:nSeq
%         y = ySeq{iSeq};
%         N = HMModel.N(iSeq);
%         Ypaux = zeros(N - p, p);
%         for i = 1:N - p
%             Ypaux(i, :) = -fliplr(y(i : i + p - 1));
%         end
%         idxend = idxini + N - p - 1;
%         Yp(idxini:idxend, :) = Ypaux;
%         idxini = idxend + 1;
%         HMModel.DelayARMatrix(iSeq).Seq = Ypaux;
%         Ypini((iSeq - 1)*sigmaIniaux*p + 1: iSeq*sigmaIniaux*p, :) = ...
%             Ypaux(1:sigmaIniaux*p, :);
%         Yini((iSeq - 1)*sigmaIniaux*p + 1: iSeq*sigmaIniaux*p) = ...
%             y(p+1:(sigmaIniaux + 1)*p)';
%     end
    % Parameters of observation model, i.e. emissions
    aSeq = [];
    % Initial observation noise under linear model
    aini = (Ypini'*Ypini)\(Ypini'*Yini);
    sig = ones(1, K)*std(Ypini*aini - Yini, 1);
    sig2 = sig(1)^2;
    % AR Kalman filter
    alph = 0.1;
    evTh = 0.5;             % Threshold for low evidence observations
    %evTh = 0.1;
    idxevThSeq = cell(1, nSeq);
    yauxall = [];
    Xauxall = [];
    for iSeq = 1:nSeq
        KalmanARModel = KalmanAR(ySeq{iSeq}, p, 'alph', alph, 'sig2', sig2);
        %KalmanARModel = KalmanAR(ySeq{iSeq}, p, 'alph', alph);
        a = KalmanARModel.a;
        evX = KalmanARModel.evX;
        % Only keep AR coefficients without low evidence
        idxevThSeq{iSeq} = find(evX > evTh);
        idxevThSeq{iSeq} = setdiff(idxevThSeq{iSeq}, HMModel.N(iSeq));
        aSeq = [aSeq; a(:, idxevThSeq{iSeq})'];
        yidx = ySeq{iSeq}(idxevThSeq{iSeq} + 1);
        yauxall = [yauxall yidx];
        Xidx = zeros(p, numel(yidx));
        for i = 1:numel(yidx)
            Xidx(:, i) = -fliplr(ySeq{iSeq}(idxevThSeq{iSeq}(i)-p+1:idxevThSeq{iSeq}(i)));
        end
        Xauxall = [Xauxall Xidx];
        %aSeq = [aSeq; a(:, evX > evTh)'];
    end
    Xauxall = Xauxall';
    switch HMModel.ObsParameters.model
        case 'Gaussian'
            [idx, C] = kmeans(aSeq, K, 'Replicates', 10);
        case 'Generalizedt'
            [idx, C] = kmeans(aSeq, K, 'Replicates', 10, 'Distance', 'cityblock');     % more
    end
    for k = 1:K
        HMModel.ObsParameters.meanParameters(k).Coefficients = C(k, :)';
        err = (Xauxall(idx == k,:) * HMModel.ObsParameters.meanParameters(k).Coefficients)' -...
            yauxall(idx == k);
        switch HMModel.ObsParameters.model
            case 'Gaussian'
                HMModel.ObsParameters.sigma(k) = std(err, 1);
            case 'Generalizedt'
                pd = fitdist(err', 'tLocationScale');
                HMModel.ObsParameters.sigma(k) = pd.sigma;
                HMModel.ObsParameters.nu(k) = pd.nu;
        end
    end
    %HMModel.ObsParameters.sigma = sig;
end
%% Probabilities of initial state
if ~isfield(HMModel.StateParameters, 'pi')
    if exist('idx', 'var')
        %HMModel.StateParameters.pi = GMModel.ComponentProportion';
        for k = 1:K
            HMModel.StateParameters.pi(k, 1) = sum(idx == k);
        end
        HMModel.StateParameters.pi = HMModel.StateParameters.pi/...
            sum(HMModel.StateParameters.pi);
%     else
%         HMModel.StateParameters.pi = 1/K*ones(K, 1);  % this is never reached - I think
    end
end

%% Transition matrices for all cases
if ~isfield(HMModel.StateParameters, 'A')
    if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'ARHMM')
        % Transition matrix - encourage persistent states
        A = 0.5*eye(K);                     % Probability of self-transitions
        for i = 1:K
            aux = gamrnd(1, 1, 1, K - 1);   % Sample from dirichlet distribution
            aux = (1 - A(i,i))*aux./sum(aux);
            A(i, 1:K ~= i) = aux;
        end
        HMModel.StateParameters.A = A;
    elseif strcmpi(HMModel.type, 'HSMMED') || strcmpi(HMModel.type, 'ARHSMMED')
        % Transition matrix - encourage persistent states
        A = 0.5*eye(K);                     % Probability of self-transitions
        for i = 1:K
            aux = gamrnd(1, 1, 1, K - 1);   % Sample from dirichlet distribution
            aux = (1 - A(i,i))*aux./sum(aux);
            A(i, 1:K ~= i) = aux;
        end
%         A = zeros(K, K);                     % Probability of self-transitions
%         for i = 1:K
%             aux = gamrnd(1, 1, 1, K);   % Sample from dirichlet distribution
%             aux = aux./sum(aux);
%             A(i, :) = aux;
%         end
        HMModel.StateParameters.A(:, :, 1) = A;
        HMModel.StateParameters.A(:, :, 2) = eye(K);
    elseif strcmpi(HMModel.type, 'HSMMVT') || strcmpi(HMModel.type, 'ARHSMMVT')
        if ~isfield(HMModel.StateParameters, 'A')
            distInt = HMModel.DurationParameters.DurationIntervals;
            A = zeros(K, K, numel(distInt) - 1);
            for j = 1:numel(distInt) - 1
                for i = 1:K
                    aux = gamrnd(1, 1, 1, K);
                    A(i, :, j) = aux/sum(aux);
                end
            end
            HMModel.StateParameters.A = A;
        end
    end
end

%% Duration parameters for HSMMED and ARHSMMED
if strcmpi(HMModel.type, 'HSMMED') || strcmpi(HMModel.type, 'ARHSMMED') ||...
    strcmpi(HMModel.type, 'HSMMVT') || strcmpi(HMModel.type, 'ARHSMMVT')
    if ~isfield(HMModel.DurationParameters, 'dmin')
        % First check if model is autoregressive
        if strcmpi(HMModel.type, 'ARHSMMED') || strcmpi(HMModel.type, 'ARHSMMVT')
            HMModel.DurationParameters.dmin = HMModel.ARorder + 1;
        else
            % Heuristic - might be painfully slow and biased!
            HMModel.DurationParameters.dmin = 1;
        end      
    else
        if strcmpi(HMModel.type, 'ARHSMMED') || strcmpi(HMModel.type, 'ARHSMMVT')
            if HMModel.DurationParameters.dmin <= HMModel.ARorder
                warning('Minimum duration in ARHSM models must be larger than AR order. Setting proper value')
                HMModel.DurationParameters.dmin = HMModel.ARorder + 1;
            end
        end
        
    end
    if ~isfield(HMModel.DurationParameters, 'dmax')
        % Heuristic - might be painfully slow!
        HMModel.DurationParameters.dmax = round(mean(horzcat(HMModel.N))/K);
    end
    dmax = HMModel.DurationParameters.dmax;
    dmin = HMModel.DurationParameters.dmin;
    HMModel.DurationParameters.flag = 1;
    if strcmpi(HMModel.DurationParameters.model, 'NonParametric')
        if ~isfield(HMModel.DurationParameters, 'PNonParametric')
            % Start with non-parametric model, then refine estimation
            aux = [zeros(1, dmin-1) ones(1, dmax - dmin + 1)];
            aux = aux/sum(aux);
            HMModel.DurationParameters.PNonParametric = repmat(aux, K, 1);
            HMModel.DurationParameters.flag = 0;
        end
    else
        switch HMModel.DurationParameters.model
            case 'Poisson'
                if ~isfield(HMModel.DurationParameters, 'lambda')
                    aux = [zeros(1, dmin-1) ones(1, dmax - dmin + 1)];
                    aux = aux/sum(aux);
                    HMModel.DurationParameters.PNonParametric = repmat(aux, K, 1);
                    % Auxiliary flag
                    HMModel.DurationParameters.flag = 1;
                else
                    HMModel.DurationParameters.flag = 0;
                end
            case  'Gaussian'
                if ~isfield(HMModel.DurationParameters, 'mu')
                    aux = [zeros(1, dmin-1) ones(1, dmax - dmin + 1)];
                    aux = aux/sum(aux);
                    HMModel.DurationParameters.PNonParametric = repmat(aux, K, 1);
                    % Auxiliary flag
                    HMModel.DurationParameters.flag = 1;
                else
                    HMModel.DurationParameters.flag = 0;
                end
        end
    end
    % Pre-compute factorials for Poisson model
    if strcmpi(HMModel.DurationParameters.model, 'Poisson')
        HMModel.DurationParameters.dlogfact = log(factorial(1:dmax));
    end
end
end