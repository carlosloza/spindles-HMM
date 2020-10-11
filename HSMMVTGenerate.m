function [HSMMSeq, HSMModel] = HSMMVTGenerate(N, K, varargin)
%% Self transitions allowed!
% a vector with duration edges is necessary - starting at d = 1 and ending
% at d = dmax. dmax is required
% A total of (D-1) KxK transition matrices are required where D is the
% number of elements in the vector of transition edges

%% Check inputs and initial conditions
for i = 1:length(varargin)
    if strcmpi(varargin{i}, 'type')
        HSMModel.type = varargin{i + 1};      % options: HSMMED, ARHSMMED
    elseif strcmpi(varargin{i}, 'StateParameters')    
        HSMModel.StateParameters = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'ObsParameters')
        HSMModel.ObsParameters = varargin{i + 1};       % structure with observations means and standard deviations
    elseif strcmpi(varargin{i}, 'DurationParameters')
        HSMModel.DurationParameters = varargin{i + 1};
    end
end

HSMModel.StateParameters.K = K;
distInt = HSMModel.DurationParameters.DurationIntervals;
dmax = HSMModel.DurationParameters.dmax;

%%
y = zeros(1, N);
z = zeros(1, N);
if strcmpi(HSMModel.DurationParameters.model, 'Poisson')
    % First hidden state
    zaux = find(mnrnd(1, HSMModel.StateParameters.pi));
    dState(1) = min([max([poissrnd(HSMModel.DurationParameters.lambda(zaux)), 1]), dmax]);
    dStatePrev = dState(1);
    z(1:min([dState(1), N])) = zaux*ones(1, min([dState(1), N]));
    i = min([dState(1), N]) + 1;
    ct = 2;   
    % Subsequent hidden states
    while i < N
        idxA = discretize(dStatePrev, distInt);
        A = HSMModel.StateParameters.A(:, :, idxA);
        zaux = find(mnrnd(1, A(z(i-1), :)));
        %zaux = mnrnd(1, HSMModel.A(z(i-1), :));
        dState(ct) = min([max([poissrnd(HSMModel.DurationParameters.lambda(zaux)), 1]), dmax]);
        dStatePrev = dState(ct);
        idxState = i:min([i + dState(ct) - 1, N]);
        z(idxState) = zaux*ones(1,numel(idxState));
        i = i + min([dState(ct), N]);     
        ct = ct + 1;
         
    end
elseif strcmpi(HSMModel.DurationParameters.model, 'Gaussian')
    % First hidden state
    zaux = find(mnrnd(1, HSMModel.StateParameters.pi));
    dState(1) = min([max([round(normrnd(HSMModel.DurationParameters.mu(zaux), ...
        HSMModel.DurationParameters.sigma(zaux))), 1]), dmax]);
    dStatePrev = dState(1);
    z(1:min([dState(1), N])) = zaux*ones(1, min([dState(1), N]));
    i = min([dState(1), N]) + 1;
    ct = 2;
    % Subsequent hidden states
    while i < N
        idxA = discretize(dStatePrev, distInt);
        A = HSMModel.StateParameters.A(:, :, idxA);
        zaux = find(mnrnd(1, A(z(i-1), :)));
        dState(ct) = min([max([round(normrnd(HSMModel.DurationParameters.mu(zaux), ...
            HSMModel.DurationParameters.sigma(zaux))), 1]), dmax]);
        dStatePrev = dState(ct);
        idxState = i:min([i + dState(ct) - 1, N]);
        z(idxState) = zaux*ones(1,numel(idxState));
        i = i + min([dState(ct), N]);
        ct = ct + 1;
    end
end
%% Generate observations
switch HSMModel.ObsParameters.model
    case 'Gaussian'
        for k = 1:K
            idx = find(z == k);
            y(idx) = HSMModel.ObsParameters.mu(k) + ...
                HSMModel.ObsParameters.sigma(k)*randn(1, numel(idx));
        end
    case 'Generalizedt'
        for k = 1:K
            idx = find(z == k);
            y(idx) = random('tLocationScale', HSMModel.ObsParameters.mu(k),...
                HSMModel.ObsParameters.sigma(k), HSMModel.ObsParameters.nu(k), 1, numel(idx));
        end       
    case 'MultivariateGaussian'
        d = size(HMModel.ObsParameters.mu, 1);
        y = zeros(d, N);
        for k = 1:K
            idx = find(z == k);
            y(:, idx) = mvnrnd(HSMModel.ObsParameters.mu(:, k)', ...
                HSMModel.ObsParameters.sigma(:,:,k)', numel(idx))';
        end
end

%%
HSMMSeq.z = z;
HSMMSeq.dState = dState;
HSMMSeq.y = y;
end