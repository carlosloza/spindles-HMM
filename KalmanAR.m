function KalmanARModel = KalmanAR(x, p, varargin)
% Estimation of parameters of a Kalman Filter Autoregressive (AR) Model given
% univariate observations
% Batch implementation. State noise variance is estimated online using the
% algorithm proposed by Jazwinski, 1969
% For further reference see "Dynamic Linear Models, Recursive Least Squares
% and Steepest-Descent Learning" by Penny and Roberts, 1998 and "Dynamic Models
% for Nonstationary Signal Segmentation" by Penny and Roberts, 1998.
% Author: Carlos Loza
%
% Parameters
% ----------
% x :                   vector, size (N, 1) or (1, N)
%                       Observations from model
% p :                   int
%                       Order of autoregressive model
% alph :                float
%                       Tracking parameter. Smaller values favor tracking
%                       whereas larger values yield smooth state stimates
%                       Default: 0.1
% sig2 :                float
%                       Observation noise variance
%                       Default: 0.2
%
% Returns
% -------
% KalmanARModel :       structure array. Fields:
% a :                   matrix, size (p, N)
%                       State stimates with autoregressive coefficients
%                       over time
% evX :                 vector, size (1, N)
%                       Evidence of observations (likelihood) given the 
%                       model parameters 
% LRate :               vector, size (1, N)
%                       Average learning rate of Kalman Filter AR model
% StateVar :            vector, size (1, N)
%                       Estimated state noise variance
%
% Examples :
% KalmanARModel = KalmanAR(x, 'p', 10, 'alph', 0.1, 'sig2', 0.15)

%% Check inputs and initial conditions
% Defaults
alph = 0.1;
sig2 = 0.2;
for i = 1:length(varargin)
    if strcmpi(varargin{i}, 'p')
        p = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'alph')
        alph = varargin{i + 1};
    elseif strcmpi(varargin{i}, 'sig2')
        sig2 = varargin{i + 1};
    end
end
if ~exist('p', 'var')
    fprintf('AR model order is required \n')
end

% Observations must be row vectors
if iscolumn(x)
    x = x';
end
x = zscore(x);                          % Normalize observations
N = numel(x);

%% Kalman Filter solution 
[U, S, V] = svd(fliplr(x(1:p)));
aux = U'*x(p + 1);
a_ini = V(:, 1)*(aux./S(1));

% Initial estimate - SVD solution
a = zeros(p, N);
a(:, p) = a_ini;
q = zeros(1, N);                        % Estimated state noise variance
Fn = -fliplr(x(1:p));
SigCovtprev = sig2*(Fn*Fn');
e = zeros(1, N);                        % Estimation error
sig2_q0 = zeros(1, N);                  % Estimated prediction variance when q = 0
sig2_yt = zeros(1, N);                  % Estimated prediction variance
K = zeros(p, N);                        % Kalman gain
SigStaten = zeros(p, p, N);             % Estimated state covariance             
LRate = zeros(1, N);                    % Average learning rate
pred_y = zeros(1, N);                   % Predicted observation
evX = zeros(1, N);                      % Evidence of observations given model parameters
for i = p + 1:N
    Fn = -fliplr(x(i-p: i-1));
    e(i) = x(i) - Fn*a(:, i-1);
    sig2_q0(i) = sig2 + Fn*SigCovtprev*Fn';
    % State noise estimation (variance)
    h_arg = (e(i)^2 - sig2_q0(i))/(Fn*Fn');
    % Ramp function
    if h_arg >= 0
        h = h_arg;
    else
        h = 0;
    end
    % State noise variance update
    q(i) = alph*q(i-1) + (1 - alph)*h;    
    if numel(SigCovtprev) == 1
        SigStaten(:, :, i) = SigCovtprev*eye(p) + q(i)*eye(p);
    else
        SigStaten(:,:,i) = SigCovtprev + q(i)*eye(p);
    end    
    sig2_a = Fn*SigStaten(:, :, i)*Fn';
    sig2_yt(i) = sig2 + sig2_a;    
    pred_y(i) = Fn*a(:, i-1);
    evX(i) = (1/sqrt(2*pi*sig2_yt(i)))*(exp(-e(i)^2/(2*sig2_yt(i))));   
    LRate(i) = (1/p)*(trace(SigStaten(:,:,i))/sig2_yt(i));   
    K(:, i) = (SigStaten(:,:,i)*Fn')/sig2_yt(i);    
    a(:, i) = a(:, i - 1) + K(:, i)*e(i);
    SigCovtprev = SigStaten(:,:,i) - K(:, i)*Fn*SigStaten(:,:,i) ;
end

KalmanARModel.a = a;
KalmanARModel.evX = evX;
KalmanARModel.LRate = LRate;
KalmanARModel.StateVar = q;

end