function HMModel = InitialConditionsSleepSpindles(ySeq, HMModel)
%INITIALCONDITIONSSLEEPSPINDLES Sets initial conditions for parameter
% estimation under robust autoregressive hidden semi-Markov model (RARHSMM)
% via EM algorithm for sequences sampled at 50 Hz
%
%   Parameters
%   ----------
%   ySeq :                  single row vector, size (1, N) or cell, size (1, NSeq),
%                           Single univariate time series (sequence) or batch
%                           of univariate sequences (not necessarily of the same
%                           length). In bath case, sequences must be row vectors
%   HMModel :               structure
%                           Hyperparameters such as number of regimes,
%                           normalization flag, robut initialization flag,
%                           minimum and maximum regime durations and so on.
%                           It has the same format as the HMModel structure
%                           output. This structure is meant to be a concise
%                           way to provide hyperparameters
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
%Example: HMModel = InitialConditionsSleepSpindles(ySeq, HMModel)
%Important: Initial conditions are set to work properly for sequences sampled
%at 50 Hz. Other sampling frequencies might require different parameters
%Note: Requires Statistics and Machine Learning Toolbox and
%Signal Processing Toolbox
%Author: Carlos Loza (carlos.loza@utexas.edu)
%https://github.com/carlosloza/spindles-HMM

%%
nSeq = HMModel.nSeq;                % Number of contionally iid input sequences given the model parameters
K = HMModel.StateParameters.K;      % Number of regimes/modes 
% Observation parameters (i.e., emissions)
if ~isfield(HMModel, 'ObsParameters')
    HMModel.ObsParameters.meanParameters = [];
    if ~isfield(HMModel.ObsParameters.meanParameters, 'Coefficients')
        p = HMModel.ARorder;
        % Build auxiliary matrix of AR predictors (embedding matrix)
        % Take first few seconds of each sequences to estimate non-spindle
        % observation parameters
        sigmaIniaux = 10;
        Ypini = zeros(sigmaIniaux*p*nSeq, p);
        Yini = zeros(sigmaIniaux*p*nSeq, 1);
        for iSeq = 1:nSeq
            y = ySeq{iSeq};
            Ypaux = HMModel.DelayARMatrix(iSeq).Seq;
            Ypini((iSeq - 1)*sigmaIniaux*p + 1: iSeq*sigmaIniaux*p, :) = ...
                Ypaux(1:sigmaIniaux*p, :);
            Yini((iSeq - 1)*sigmaIniaux*p + 1: iSeq*sigmaIniaux*p) = ...
                y(p+1:(sigmaIniaux + 1)*p)';
        end
        if HMModel.robustIni
            mdl = fitlm(Ypini, Yini, 'Intercept', false, 'RobustOpts', 'welsch');
        else
            mdl = fitlm(Ypini, Yini, 'Intercept', false);
        end       
        ARnonSpindle = table2array(mdl.Coefficients(:,1));      
        sig = ones(1, K)*std(Ypini*ARnonSpindle - Yini, 1);
        % Build transfer function
        a1 = [1; ARnonSpindle]';
        b1 = [zeros(1, p-1) 1];
        [z1, p1, k1] = tf2zp(b1, a1);       % Transfer function to zero-pole representation
        % Modify zero-pole representation to emphasize a single peak in the
        % EEG sigma band
        z2 = z1;
        k2 = k1;
        p2 = p1;
        p2(1:2) = 1*abs(p2(1:2)).*exp(1i*angle(p2(1:2)));       % peak at DC
        p2(3:4) = 1.7*abs(p2(3:4)).*exp(1i*1.1*angle(p2(3:4))); % peak at sigma (sleep spindles) band
        p2(5) = 0.05*p2(5);                                     % "peak" at Nyquist frequency
        [b2, a2] = zp2tf(z2, p2, k2);       % Zero-pole representation to transfer function
        % "Final" initial AR coefficients
        C(1, :) = a1(2:end);
        C(2, :) = a2(2:end);
        % Additive noise components (scale and degrees of freedom)
        sig(1) = 0.8*sig(2);
        nuGent = [4, 10];
        for k = 1:K
            HMModel.ObsParameters.meanParameters(k).Coefficients = C(k, :)';
            HMModel.ObsParameters.sigma(k) = sig(k);
            HMModel.ObsParameters.nu(k) = nuGent(k);
        end
    end
end

%% Parameters of hidden state (labels, z)
HMModel.StateParameters.pi = [1 0]';            % always start with non-sleep spindle regime
% State transition matrix
A = [0.5 0.5; 1 0];
HMModel.StateParameters.A = A;

%% Parameters of regime durations
% Check if dmin was provided
if ~isfield(HMModel.DurationParameters, 'dmin')
    % no dmin provided, set it larger than the AR order
    dmin = HMModel.ARorder + 1;
    fprintf('Minimum duration of regimes not provided, setting it to AR model plus 1: %d samples \n', dmin)
    HMModel.DurationParameters.dmin = dmin;
else 
    if HMModel.DurationParameters.dmin <= HMModel.ARorder
        dmin = HMModel.DurationParameters.dmin;
        fprintf('Minimum duration of regimes must be larger than AR model, setting it to: %d samples \n', dmin)
        HMModel.DurationParameters.dmin = dmin;
    else
        dmin = HMModel.DurationParameters.dmin;
    end
end
% Check if dmax was provided
if ~isfield(HMModel.DurationParameters, 'dmax')
    % Other more subtle heuristics can be added here
    error('error: dmax (maximum duration of regimes/modes) must be provided')
end
dmax = HMModel.DurationParameters.dmax;
for k = 1:K
    if k == 1
        % Non-sleep spindle regime: Uniform prior plus censoring on
        % durations smaller than dmin
        f = ones(1, dmax - dmin + 1);
        f = f/sum(f);
        HMModel.DurationParameters.PNonParametric(k, :) = [zeros(1, dmin-1) f];
    else
        % Sleep spindles regime
        if isfield(HMModel.DurationParameters, 'Ini')
            % Experimental - to combine with previously labeled data in the
            % future
            f1 = [zeros(1, 24) normpdf(25:dmax, HMModel.DurationParameters.Ini(1), HMModel.DurationParameters.Ini(2))];
        else
            % Gaussian prior as implemented in paper
            % KEY: this is tuned to 50 Hz sampling rate 
            f1 = [zeros(1, 24) normpdf(25:dmax, 50, 5)];
        end
        f1 = f1/sum(f1);
        f = [f1 zeros(1, dmax - numel(f1))];
        HMModel.DurationParameters.PNonParametric(k, :) = f;
    end
    clear f
end
    
end