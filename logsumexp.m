function logs = logsumexp(a, dim)
[b, ~] = max(a,[],dim);
% idx = isinf(b);
% if sum(idx) ~= numel(b)
%     logs = b + reallog(sum(exp(a - b), dim));
%     logs(idx) = -Inf;
% else
%     logs = -Inf(size(b));
% end

logs = b + reallog(sum(exp(a - b), dim));
i = find(isinf(b));
if ~isempty(i)
  logs(i) = b(i);
end
end