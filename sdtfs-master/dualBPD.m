function [x1, x2, c1, c2, cost] = dualBPD(x, A1, A1H, A2, A2H, lam1, lam2, mu, Nit, plot_flag)

% [x1, x2, c1, c2, cost] = dualBPD(x, A1, A1H, A2, A2H, lam1, lam2, mu, Nit)
%
% DUAL BASIS PURSUIT DENOISING
% Signal decomposition using two transforms (Parseval frames)
% Minimize with respect to (c1, c2):
% 0.5 ||x - A1 c1 + A2 c2||_2^2 + lam1 ||c1||_1 + lam2 ||c2||_2  
%
% Requirements:
%    A1*A1H = I
%    A2*A2H = I
%    lam1 > 0
%    lam2 > 0
%
% INPUT
%   x                : input signal signal
%   A1H, A1, A2, A2H : functions handles for Parseval frames and inverses
%   lam1, lam2       : regularization parameters
%   mu               : Augmented Lagrangian parameter
%   Nit              : number of iterations
%
% OUTPUT
%   x1, x2 : components
%   c1, c2 : coefficients of components
%
% Use [x1, x2, c1, c2, cost] = dualBPD(...) to return cost function history.
%
% Use [...] = dualBPD(...,'plots') to plot progress of algorithm.

% Ivan Selesnick
% Polytechnic Institute of New York University
% May 2010
% Revised August 2011

% Reference:
% I. W. Selesnick. Sparse signal representations using the tunable Q-factor wavelet transform.
% Proc. SPIE 8138 (Wavelets and Sparsity XIV), August 2011. doi:10.1117/12.894280.

% Reference
% M. V. Afonso, J. M. Bioucas-Dias, and M. A. T. Figueiredo.
% Fast image recovery using variable splitting and constrained optimization.
% IEEE Trans. Image Process., 19(9):2345–2356, September 2010.


% By default do not compute cost function (to reduce computation)
if nargout > 4
    COST = true;
    cost = zeros(1,Nit);     % cost function
else
    COST = false;
    cost = [];
end

GOPLOTS = false;
% if nargin == 10
%     if strcmp(plot_flag,'plots')
%         GOPLOTS = true;
%     end
% end

% Initialize:

c1 = A1H(x);
c2 = A2H(x);

d1 = A1H(zeros(size(x)));
d2 = A2H(zeros(size(x)));

T1 = lam1/mu;
T2 = lam2/mu;

N = length(x);
A = 1.1*max(abs(x));
%progressbar
for k = 1:Nit
    % fprintf('Iteration %d\n', k)
    
    u1 = soft(c1 + d1, T1) - d1;
    u2 = soft(c2 + d2, T2) - d2;
    c = x - A1(u1) - A2(u2);
    % c = 0.5 * c;
    
    d1 = A1H(c)/(mu+2);
    d2 = A2H(c)/(mu+2);
    
    c1 = d1 + u1;
    c2 = d2 + u2;
    
    if COST | GOPLOTS        
        x1 = A1(c1);
        x2 = A2(c2);
        res = x - x1 - x2;
    end
    
    if COST
        cost(k) = 0.5*sum(abs(res(:)).^2) + lam1*sum(abs(c1(:))) + lam2*sum(abs(c2(:)));
    end
    
    if GOPLOTS
        
        figure(gcf)
        clf
        subplot(3,1,1)
        plot(real(x1))
        xlim([0 N])
        ylim([-A A])
        title({sprintf('ITERATION %d',k),'COMPONENT 1'})
        box off
        subplot(3,1,2)
        plot(real(x2))
        xlim([0 N])
        ylim([-A A])
        box off
        title('COMPONENT 2')
        subplot(3,1,3)
        plot(real(res))
        xlim([0 N])
        ylim([-A A])
        title('RESIDUAL')
        box off
        drawnow
    end
    %progressbar(k/Nit)
end

x1 = A1(c1);
x2 = A2(c2);

