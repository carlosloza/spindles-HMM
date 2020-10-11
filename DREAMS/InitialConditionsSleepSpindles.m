%% Initial conditions for sleep spindles modeling
function HMModel = InitialConditionsSleepSpindles(ySeq, HMModel)
nSeq = HMModel.nSeq;
K = HMModel.StateParameters.K;
if ~isfield(HMModel, 'ObsParameters')
    if size(ySeq{1}, 1) == 1
            % Default univariate Gaussian model for observations
            HMModel.ObsParameters.model = 'Gaussian';
            %HMModel.ObsParameters.meanParameters = [];
        else
            % Default multivariate Gaussian model for observations
            HMModel.ObsParameters.model = 'MultivariateGaussian';
    end
end
%% HMM and HSMMED - DON'T CARE ABOUT THIS
if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'HSMMED')
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

%% ARHMM and ARHSMMED
if strcmpi(HMModel.type, 'ARHMM') || strcmpi(HMModel.type, 'ARHSMMED') || strcmpi(HMModel.type, 'ARHSMMVT')
    %if ~isfield(HMModel.ObsParameters.meanParameters, 'Coefficients')
    if ~isfield(HMModel.ObsParameters, 'meanParameters')
        HMModel.ObsParameters.meanParameters = [];
        if ~isfield(HMModel.ObsParameters, 'meanModel')
            % Default - Linear AR model
            HMModel.ObsParameters.meanModel = 'Linear';
        end
        %if ~isfield(HMModel.ObsParameters, 'meanParameters')
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
                %N = HMModel.N(iSeq);
                %idxend = idxini + N - p - 1;
                %Yp(idxini:idxend, :) = Ypaux;
                %idxini = idxend + 1;
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
%             sig2 = sig(1)^2;                
%             sig2 = 0.01;                % 0.01 good
%             % AR Kalman filter
%             alph = 0.1;
%             %alph = 0.05;
%             evTh = 0.5;             % Threshold for low evidence
%             %observations 0.5 good
%             %evTh = 0.9;
%             idxevThSeq = cell(1, nSeq);
%             yauxall = [];
%             Xauxall = [];
%             for iSeq = 1:nSeq
%                 KalmanARModel = KalmanAR(ySeq{iSeq}, p, 'alph', alph, 'sig2', sig2);
%                 a = KalmanARModel.a;
%                 evX = KalmanARModel.evX;
%                 % Only keep AR coefficients without low evidence
%                 idxevThSeq{iSeq} = find(evX > evTh);
%                 idxevThSeq{iSeq} = setdiff(idxevThSeq{iSeq}, HMModel.N(iSeq));
%                 aSeq = [aSeq; a(:, idxevThSeq{iSeq})'];
%                 yidx = ySeq{iSeq}(idxevThSeq{iSeq} + 1);
%                 yauxall = [yauxall yidx];
%                 Xidx = zeros(p, numel(yidx));
%                 for i = 1:numel(yidx)
%                     Xidx(:, i) = -fliplr(ySeq{iSeq}(idxevThSeq{iSeq}(i)-p+1:idxevThSeq{iSeq}(i)));
%                 end
%                 Xauxall = [Xauxall Xidx];
%                 %aSeq = [aSeq; a(:, evX > evTh)'];
%             end
%             Xauxall = Xauxall';
%             switch HMModel.ObsParameters.model
%                 case 'Gaussian'
%                     [idx, C] = kmeans(aSeq, K, 'Replicates', 10);
%                 case 'Generalizedt'
%                     [idx, C] = kmeans(aSeq, K, 'Replicates', 10, 'Distance', 'cityblock');     % more
%             end
%             % Assign states according to PSD differences
%             logPSD = zeros(1024, 2);
%             for k = 1:K
%                 [H, F] = freqz(1, [1; C(k, :)'], 1024, HMModel.Fs);
%                 logPSD(:, k) = 20*log10(abs(H));
%             end
%             if diff(sum(logPSD(F >=11 & F <=16, :))) < 0
%                 % swith labels
%                 C = flipud(C);
%                 idx = -idx + 3;
%             end
           
            %sig(2) = 0.8*sig(1);            % 0.8 helped A LOT for Gaussian
            
            sig(1) = 0.8*sig(2);
            nuGent = [4, 10];
            for k = 1:K
                HMModel.ObsParameters.meanParameters(k).Coefficients = C(k, :)';
                %err = (Xauxall(idx == k,:) * HMModel.ObsParameters.meanParameters(k).Coefficients)' -...
                %    yauxall(idx == k);
                switch HMModel.ObsParameters.model
                    case 'Gaussian'
                        HMModel.ObsParameters.sigma(k) = sig(k);
                    case 'Generalizedt'
                        % this will throw an error
                        %pd = fitdist(err', 'tLocationScale');
                        %HMModel.ObsParameters.sigma(k) = pd.sigma;
                        %HMModel.ObsParameters.nu(k) = pd.nu;
                        HMModel.ObsParameters.sigma(k) = sig(k);
                        HMModel.ObsParameters.nu(k) = nuGent(k);
                end
            end
            
            % Correct initial conditions of spindles
            %HMModel.ObsParameters.sigma(2) = 0.9*HMModel.ObsParameters.sigma(1);
            %HMModel.ObsParameters.nu(2) = 2.5*HMModel.ObsParameters.nu(1);
            
        end
        
%         figure
%         [HH, FF] = freqz(1, [1; HMModel.ObsParameters.meanParameters(1).Coefficients], 1024, HMModel.Fs);
%         plot(FF, 20*log10(abs(HH)))
%         hold on
%         [HH, FF] = freqz(1, [1; HMModel.ObsParameters.meanParameters(2).Coefficients], 1024, HMModel.Fs);
%         plot(FF, 20*log10(abs(HH)), 'r')
%         ylim([-15 20])
%         pause(0.1)
%         
%         asd = 1;
        %HMModel.ObsParameters.sigma = sig;
    end
end

%% Probabilities of initial state
if ~isfield(HMModel.StateParameters, 'pi')
    HMModel.StateParameters.pi = [1 0]';            % always start with non-spindle
end

%% Transition matrices for all cases
if ~isfield(HMModel.StateParameters, 'A')
    switch HMModel.type
        case 'ARHMM'
            A = [0.9 0.1; 0.1 0.9];
        case 'ARHSMMED'
            % TO - DO
            %A = [0.9 0.1; 1 0];
            A = [0.5 0.5; 1 0];
        case 'ARHSMMVT'
            % Do nothing - pass this as a parameter
%             dmin = HMModel.DurationParameters.dmin;
%             dmax = HMModel.DurationParameters.dmax;
%             distInt = HMModel.DurationParameters.DurationIntervals;
%             distIntaux = distInt;
%             distIntaux(end) = dmax + 1;
    end
%     % Transition matrix - encourage persistent states
%     A = 0.8*eye(K);
%     for i = 1:K
%         aux = gamrnd(1, 1, 1, K - 1);   % Sample from dirichlet distribution
%         aux = (1 - A(i,i))*aux./sum(aux);
%         A(i, 1:K ~= i) = aux;
%     end
    if strcmpi(HMModel.type, 'HMM') || strcmpi(HMModel.type, 'ARHMM')
        HMModel.StateParameters.A = A;
    elseif strcmpi(HMModel.type, 'HSMMED') || strcmpi(HMModel.type, 'ARHSMMED')
        HMModel.StateParameters.A(:, :, 1) = A;
        HMModel.StateParameters.A(:, :, 2) = eye(K);
    end
end

%% Duration parameters for HSMMED and ARHSMMED
if strcmpi(HMModel.type, 'HSMMED') || strcmpi(HMModel.type, 'ARHSMMED') || strcmpi(HMModel.type, 'ARHSMMVT')
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
            for k = 1:K
                if k == 1
                    f = ones(1, dmax - dmin + 1); 
                    %f = exp((dmax/HMModel.Fs)*(dmin:dmax)/(dmax-dmin));
                    f = f/sum(f);
                    HMModel.DurationParameters.PNonParametric(k, :) = [zeros(1, dmin-1) f];
                    %f = normpdf(dmin:dmax, dmax, 100);
                else
                    %f = [zeros(1, 49) normpdf(50:dmax, 100, 50)];        % Only works for Fs = 100 Hz!!!!!!!!! 
                    %f = [zeros(1, 24) normpdf(25:dmax, 50, 25)];        % Only works for Fs = 50 Hz!
                    %f = f/sum(f);
                    %HMModel.DurationParameters.PNonParametric(k, :) = f;
                    %f1 = [zeros(1, 24) normpdf(25:dmax, 50, 30)];       
                    %f1 = [zeros(1, 24) normpdf(25:dmax, 50, 10)]; 
                    % Only works for Fs = 50 Hz!
                    %auxf = round(HMModel.DurationParameters.Ini(1) - 2*HMModel.DurationParameters.Ini(2));
                    %f1 = [zeros(1, auxf) normpdf(auxf+1:dmax, HMModel.DurationParameters.Ini(1), HMModel.DurationParameters.Ini(2))];
                    %f1 = [zeros(1, 24) ones(1, 150 - 25 + 1)];          % Only works for Fs = 50 Hz!
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
end