function [HMMSeq, HMModel] = HMMGenerate(N, K, varargin)
%% TODO - switch/case instead of elseifs (like HSMMGenerate)

%% Check inputs and initial conditions
for i = 1:length(varargin)
    if strcmpi(varargin{i}, 'type')
        HMModel.type = varargin{i + 1};      % options: HMM, ARHMM
    elseif strcmpi(varargin{i}, 'StateParameters')    
        HMModel.StateParameters = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'ObsParameters')
        HMModel.ObsParameters = varargin{i + 1};       % structure with observations means and standard deviations
    end
end

HMModel.StateParameters.K = K;

%% Types
if strcmpi(HMModel.ObsParameters.model, 'Gaussian')
    % univariate Gaussian
    y = zeros(1, N);
    z = zeros(1, N);
    % First hidden state
    z(1) = find(mnrnd(1, HMModel.StateParameters.pi));
    % Subsequent hidden states
    for n = 2:N
        z(n) = find(mnrnd(1, HMModel.StateParameters.A(z(n-1), :)));
    end
    % Generate observations
    for k = 1:K
        idx = find(z == k);
        y(idx) = HMModel.ObsParameters.mu(k) + ...
            HMModel.ObsParameters.sigma(k)*randn(1, numel(idx));
    end
elseif strcmpi(HMModel.ObsParameters.model, 'Generalizedt')
    % univariate generalized t distribution
    y = zeros(1, N);
    z = zeros(1, N);
    % First hidden state
    z(1) = find(mnrnd(1, HMModel.StateParameters.pi));
    % Subsequent hidden states
    for n = 2:N
        z(n) = find(mnrnd(1, HMModel.StateParameters.A(z(n-1), :)));
    end
    % Generate observations
    for k = 1:K
        idx = find(z == k);
        y(idx) = random('tLocationScale', HMModel.ObsParameters.mu(k),...
           HMModel.ObsParameters.sigma(k), HMModel.ObsParameters.nu(k), 1, numel(idx));
    end
elseif strcmpi(HMModel.ObsParameters.model, 'MultivariateGaussian')
    % multivariate Gaussian
    d = size(HMModel.ObsParameters.mu, 1);
    y = zeros(d, N);
    z = zeros(1, N);
    % First hidden state
    z(1) = find(mnrnd(1, HMModel.StateParameters.pi));
    % Subsequent hidden states
    for n = 2:N
        z(n) = find(mnrnd(1, HMModel.StateParameters.A(z(n-1), :)));
    end
    % Generate observations
    for k = 1:K
        idx = find(z == k);
        y(:, idx) = mvnrnd(HMModel.ObsParameters.mu(:, k)', ...
            HMModel.ObsParameters.sigma(:,:,k)', numel(idx))';
    end
end


%%
HMMSeq.z = z;
HMMSeq.x = z;
HMMSeq.y = y;
end