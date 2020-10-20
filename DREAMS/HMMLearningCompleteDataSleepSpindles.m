function HMModel = HMMLearningCompleteDataSleepSpindles(TrainingStructure, HMModel)
%HMMLEARNINGCOMPLETEDATASLEEPSPINDLES Learning/estimation of robust 
% autoregressive hidden semi-Markov model (RARHSMM)parameters given input
% sequence(s) AND label(s)
%
%   Parameters
%   ----------
%   TrainingStructure :     structure 
%                           Fields:
%                           y : row vector(s) with possibly different
%                           lengths. Univariate observation sequence(s)
%                           z : row vector(s) with possibly different
%                           lengths BUT same lengths as corresponding entries
%                           in "y" field. Univariate label/state sequence(s)
%   HMModel :               structure
%                           Hyperparameters such as number of regimes,
%                           normalization flag, robut initialization flag,
%                           minimum and maximum regime durations and so on.
%                           It has the same format as the HMModel structure
%                           output. This structure is meant to be a concise
%                           way to provide hyperparameters
%
%   Returns
%   -------
%   HMModel :               structure
%                           Fields: 
%                           - StateParameters: structure with number of
%                           regimes (K), initial probabilities vector (pi),
%                           state transition matrix (A)
%                           - DurationParameters: structure with minimum 
%                           (dmin) duration of regimes, maximum duration of 
%                           regimes (duration in samples), and probability 
%                           mass function of regime durations (non-parametric 
%                           density)
%                           - ObsParameters: structure with autoregressive
%                           coefficients (meanParameters), scale, and
%                           degrees of freedom of observation noise
%                           - ARorder: autoregressive order
%                           - robustIni: flag to use robust linear regression 
%                           for initial estimate of AR coefficients (same as 
%                           input to HMMLearning function)
%                           - normalize: flag to zscore input sequence(s)
%                           (same as input to HMMLearning function)
%                           - Fs: sampling frequency in Hz
%                           (same as input to HMMLearning function)
%
%Example: HMModel = HMMLearningCompleteDataSleepSpindles(yzStruct, HMModel)
%Note: Requires Statistics and Machine Learning Toolbox 
%Author: Carlos Loza (carlos.loza@utexas.edu)
%https://github.com/carlosloza/spindles-HMM

%% General parameters and settings
nSeq = size(TrainingStructure, 2);          % Number of contionally iid input sequences given the model parameters
K = HMModel.StateParameters.K;              % Number of regimes/modes 
iIni = HMModel.ARorder + 1;                 % initial time sample for learning
% Format input
for i = 1:nSeq
    y = TrainingStructure(i).y;
    % Normalize each training observation sequence 
    if HMModel.normalize
        y = zscore(y);                         
    end
    TrainingStructure(i).y = y;
    TrainingStructure(i).N = size(y, 2);
end

%% Parameters of observation model (i.e., emissions)
p = HMModel.ARorder;                        % Autoregressive order
Ypcell = cell(1, nSeq);
labelsall = [];                             % vector with all labels concatenates
yauxall = [];                               % vector with all univariate observations concatenated
for train_i = 1:nSeq
    y = TrainingStructure(train_i).y;
    N = TrainingStructure(train_i).N;
    % Build auxiliary matrix of AR predictors (embedding amtrix)
    Yp = zeros(N - p, p);
    for i = 1:N-p
        Yp(i, :) = -fliplr(y(i : i + p - 1));
    end
    Ypcell{train_i} = Yp;
    labelsall = [labelsall TrainingStructure(train_i).z(iIni:end)];
    yauxall = [yauxall TrainingStructure(train_i).y(iIni:end)];
end
clear Yp
% Estimate autoregressive coefficients and observation additive noise components 
Yp = cell2mat(Ypcell');
for k = 1:K
    idx = find(labelsall == k);
    Ytil = yauxall(idx)';
    Xtil = Yp(idx, :)';
    if HMModel.robustIni
        mdl = fitlm(Xtil', Ytil', 'Intercept', false, 'RobustOpts', 'welsch');
    else
        mdl = fitlm(Xtil', Ytil', 'Intercept', false);
    end    
    HMModel.ObsParameters.meanParameters(k).Coefficients = table2array(mdl.Coefficients(:,1));
    % Residual model, i.e. errors 
    err = Xtil'*HMModel.ObsParameters.meanParameters(k).Coefficients - Ytil;      
    pd = fitdist(err, 'tLocationScale');
    HMModel.ObsParameters.sigma(k) = pd.sigma;
    HMModel.ObsParameters.nu(k) = pd.nu;
end

%% Parameters of hidden markov chain (labels/regimes and durations)
HMModel.StateParameters.pi = [1 0]';        % always start with non-sleep spindle mode
if ~isfield(HMModel, 'DurationParameters')
    HMModel.DurationParameters.dmin = 0;
end
if ~isfield(HMModel.DurationParameters, 'dmax')
    % No dmax provided
    % This case implies no self-transitions at all, BUT the duration
    % distribution of the non-spindle regime will be heavily skewed (i.e., 
    % long tailed) which increases the complexity of the algorithm in both 
    % time and memory, so beware...
    fprintf('Maximum duration of regimes not provided, inference might take a while...\n')
    % Best results were obtained when dmax = 30 seconds because it is a good
    % compromise for complexity while generalizing quite well. It also
    % mimics the well-known 30-second-long-epoch EEG analysis paradigm
    A = [0 1;1 0];                      % state transition matrix
    z1 = zeros(1, N);
    z2 = zeros(1, N);
    for train_i = 1:nSeq
        N = TrainingStructure(train_i).N;
        z = TrainingStructure(train_i).z;
        ct = 1;
        for i = 2:N
            % durations
            if z(i) == z(i-1)
                ct = ct + 1;
            else
                if z(i-1) == 1
                    z1(ct) = z1(ct) + 1;
                elseif z(i-1) == 2
                    z2(ct) = z2(ct) + 1;
                end
                ct = 1;
            end
        end
    end
    idx = find(z1 ~= 0);
    HMModel.DurationParameters.dmax = idx(end);
    z1 = z1(1: HMModel.DurationParameters.dmax);
    z2 = z2(1: HMModel.DurationParameters.dmax);
    z2Seq = [];
    idxz2 = find(z2 > 0);
    for i = 1:numel(idxz2)
        z2Seq = [z2Seq idxz2(i)*ones(1, z2(idxz2(i)))];
    end
    HMModel.DurationParameters.PNonParametric(1, :) = z1./sum(z1);
    HMModel.DurationParameters.PNonParametric(2, :) = z2./sum(z2);
    % This might be useful for initial conditions of EM - TODO
    HMModel.DurationParameters.Ini = [median(z2Seq) mad(z2Seq, 0)];
    if ~isfield(HMModel.DurationParameters, 'dmin')
        % no dmin provided, set it larger than the AR order
        dmin = HMModel.ARorder + 1;
        fprintf('Minimum duration of regimes not provided, setting it to AR model plus 1: %d samples \n', dmin)
        HMModel.DurationParameters.dmin = dmin;
    else
        dmin = HMModel.DurationParameters.dmin;
        if dmin <= HMModel.ARorder
            dmin = HMModel.ARorder + 1;
            fprintf('Minimum duration of regimes must be larger than AR model, setting it to: %d samples \n', dmin)
            HMModel.DurationParameters.dmin = dmin;
        end
    end
else
    % dmax is provided
    % This case admits self-transitions, although you should not set dmax
    % smaller than the largest sleep spindle duration (e.g., 3 seconds),
    % otherwise the duration distributions will be heavily distorted and
    % not reflect the actual dynamical regimes
    dmax = HMModel.DurationParameters.dmax;
    dState = cell(nSeq, K);
    A = zeros(K, K);
    for train_i = 1:nSeq
        fl = 1;
        i = iIni;
        N = TrainingStructure(train_i).N;
        z = TrainingStructure(train_i).z;
        while fl
            aux = z(i: min([i+dmax-1, N]));
            idx1 = find(aux == 2, 1);
            if ~isempty(idx1)
                if idx1 == 1
                    A(1, 2) = A(1, 2) + 1;
                    idx2 = find(z(i+idx1-1:end) == 1, 1);
                    A(2, 1) = A(2, 1) + 1;
                    dState{train_i, 2} = [dState{train_i, 2} idx2 - 1];
                    i = idx1 + idx2 + i - 2;
                else
                    A(1, 2) = A(1, 2) + 1;
                    dState{train_i, 1} = [dState{train_i, 1} idx1-1];
                    idx2 = find(z(i+idx1-1:end) == 1, 1);
                    A(2, 1) = A(2, 1) + 1;
                    dState{train_i, 2} = [dState{train_i, 2} idx2 - 1];
                    i = idx1 + idx2 + i - 2;
                end
            else
                dState{train_i, 1} = [dState{train_i, 1} dmax];
                A(1, 1) = A(1, 1) + 1;
                i = i + dmax;
            end
            if i > numel(z)
                break
            end
        end
    end
    if ~isfield(HMModel.DurationParameters, 'dmin')
        % no dmin provided, set it larger than the AR order
        dmin = HMModel.ARorder + 1;
        fprintf('Minimum duration of regimes not provided, setting it to AR model plus 1: %d samples \n', dmin)
        HMModel.DurationParameters.dmin = dmin;
    else
        dmin = HMModel.DurationParameters.dmin;
        if dmin <= HMModel.ARorder
            dmin = HMModel.ARorder + 1;
            fprintf('Minimum duration of regimes must be larger than AR model, setting it to: %d samples \n', dmin)
            HMModel.DurationParameters.dmin = dmin;
        end
    end
    
    for k = 1:K
        zSeq = cell2mat(dState(:,k)');
        zSeq(zSeq < dmin) = dmin;
        counts = histcounts(zSeq, 1:HMModel.DurationParameters.dmax + 1);
        HMModel.DurationParameters.PNonParametric(k, :) = counts./sum(counts);
        if k == 2
            % This might be useful for initial conditions of EM - TODO
            HMModel.DurationParameters.Ini = [median(zSeq) mad(zSeq, 0)];
        end
    end
    % State transition matrix
    for i = 1:2
        A(i, :) = A(i, :)./sum(A(i, :));
    end    
end
HMModel.StateParameters.A(:, :, 1) = A;

%% Sort fields
sortCell = {'StateParameters', 'DurationParameters', 'ObsParameters',...
    'ARorder', 'robustIni', 'normalize', 'Fs'};
HMModel = orderfields(HMModel, sortCell);
end