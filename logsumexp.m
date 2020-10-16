function y = logsumexp(x, dim)
%LOGSUMEXP  Log-sum-exp implementation for alpha, beta and other estimators 
%   of E-step (expectation) of EM algorithm.
%   y = LOGSUMEXP(x, dim) reduces the array x across the dimension dim
%   (i.e., marginalization over discrete random variable in dimension dim)
%Author: Carlos Loza (carlos.loza@utexas.edu)
%https://github.com/carlosloza/spindles-HMM
[b, ~] = max(x, [], dim);
y = b + reallog(sum(exp(x - b), dim));
i = find(isinf(b));
if ~isempty(i)
  y(i) = b(i);
end
end