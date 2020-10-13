%% Initial conditions for sleep spindles modeling
function HMModel = InitialConditionsSleepSpindles(ySeq, HMModel)
nSeq = HMModel.nSeq;
K = HMModel.StateParameters.K;
%% ARHSMMED
if ~isfield(HMModel.ObsParameters, 'meanParameters')
    HMModel.ObsParameters.meanParameters = [];
    if ~isfield(HMModel.ObsParameters, 'meanModel')
        % Default - Linear AR model
        HMModel.ObsParameters.meanModel = 'Linear';
    end
    if ~isfield(HMModel.ObsParameters.meanParameters, 'Coefficients')
        p = HMModel.ARorder;
        % Build auxiliary matrix of AR predictors
        sigmaIniaux = round(mean(horzcat(HMModel.N))/1000);
        sigmaIniaux = 10;      % 10
        %Yp = zeros(sum(horzcat(HMModel.N)) - nSeq*p, p);
        Ypini = zeros(sigmaIniaux*p*nSeq, p);
        Yini = zeros(sigmaIniaux*p*nSeq, 1);
        %idxini = 1;
        for iSeq = 1:nSeq
            y = ySeq{iSeq};
            Ypaux = HMModel.DelayARMatrix(iSeq).Seq;
            Ypini((iSeq - 1)*sigmaIniaux*p + 1: iSeq*sigmaIniaux*p, :) = ...
                Ypaux(1:sigmaIniaux*p, :);
            Yini((iSeq - 1)*sigmaIniaux*p + 1: iSeq*sigmaIniaux*p) = ...
                y(p+1:(sigmaIniaux + 1)*p)';
        end
        % Parameters of observation model, i.e. emissions
        aSeq = [];
        % Initial observation noise under linear model
        %aini = (Ypini'*Ypini)\(Ypini'*Yini);
        mdl = fitlm(Ypini, Yini, 'Intercept', false, 'RobustOpts', 'welsch');
        aini = table2array(mdl.Coefficients(:,1));
        
        sig = ones(1, K)*std(Ypini*aini - Yini, 1);
        % No more Kalman
        a1 = [1; aini]';
        b1 = [zeros(1, p-1) 1];
        [z1, p1, k1] = tf2zp(b1, a1);
        z2 = z1;
        k2 = k1;
        p2 = p1;
        p2(1:2) = 1*abs(p2(1:2)).*exp(1i*angle(p2(1:2)));    % 1.0        % peak at DC
        p2(3:4) = 1.7*abs(p2(3:4)).*exp(1i*1.1*angle(p2(3:4))); % 1.7       % peak at spindles
        p2(5) = 0.05*p2(5);                 % 0.05    % "peak" at Nyquist frequency
        [b2, a2] = zp2tf(z2, p2, k2);
        C(1, :) = a1(2:end);
        C(2, :) = a2(2:end);
        
        sig(1) = 0.8*sig(2);
        nuGent = [4, 10];
        for k = 1:K
            HMModel.ObsParameters.meanParameters(k).Coefficients = C(k, :)';
            HMModel.ObsParameters.sigma(k) = sig(k);
            HMModel.ObsParameters.nu(k) = nuGent(k);
        end
    end
end

%% Probabilities of initial state
if ~isfield(HMModel.StateParameters, 'pi')
    HMModel.StateParameters.pi = [1 0]';            % always start with non-spindle
end

%% Transition matrices for all cases
if ~isfield(HMModel.StateParameters, 'A')
    A = [0.5 0.5; 1 0];
    HMModel.StateParameters.A(:, :, 1) = A;
    HMModel.StateParameters.A(:, :, 2) = eye(K);
%     % Transition matrix - encourage persistent states
%     A = 0.8*eye(K);
%     for i = 1:K
%         aux = gamrnd(1, 1, 1, K - 1);   % Sample from dirichlet distribution
%         aux = (1 - A(i,i))*aux./sum(aux);
%         A(i, 1:K ~= i) = aux;
%     end
%     if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'ARHMM')
%         HMModel.StateParameters.A = A;
%     elseif strcmpi(HMModel.type, 'HSMMED') || strcmpi(HMModel.type, 'ARHSMMED')
%         HMModel.StateParameters.A(:, :, 1) = A;
%         HMModel.StateParameters.A(:, :, 2) = eye(K);
%     end
end

%% Duration parameters for HSMMED and ARHSMMED
if ~isfield(HMModel.DurationParameters, 'dmin')
    % First check if model is autoregressive
    HMModel.DurationParameters.dmin = HMModel.ARorder + 1;
else
    
    if HMModel.DurationParameters.dmin <= HMModel.ARorder
        warning('Minimum duration in ARHSM models must be larger than AR order. Setting proper value')
        HMModel.DurationParameters.dmin = HMModel.ARorder + 1;
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
        for k = 1:K
            if k == 1
                f = ones(1, dmax - dmin + 1);
                %f = exp((dmax/HMModel.Fs)*(dmin:dmax)/(dmax-dmin));
                f = f/sum(f);
                HMModel.DurationParameters.PNonParametric(k, :) = [zeros(1, dmin-1) f];
            else
                if isfield(HMModel.DurationParameters, 'Ini')
                    f1 = [zeros(1, 24) normpdf(25:dmax, HMModel.DurationParameters.Ini(1), HMModel.DurationParameters.Ini(2))];
                else
                    f1 = [zeros(1, 24) normpdf(25:dmax, 50, 5)];
                end
                f1 = f1/sum(f1);
                f = [f1 zeros(1, dmax - numel(f1))];
                HMModel.DurationParameters.PNonParametric(k, :) = f;
            end
            clear f
        end
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