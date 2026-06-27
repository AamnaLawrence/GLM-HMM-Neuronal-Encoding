function [f,df] = logsumexp(x,idx)
% Computes the function:
%     f(x) = log(sum(exp(x))
% and its 1st derivative, where sum is along 'idx' dimension of x
% 
% Default is idx = 2 (columns) if unspecified

if nargin < 2
    idx = 2;
end

rowmax = max(x,[],idx); % get max along this index (or "row")
f = log(sum(exp(x-rowmax),idx))+rowmax; % compute log-sum-exp

% Compute 1st derivative:
if nargout > 1
    df = exp(x-rowmax)./sum(exp(x-rowmax),idx);
end
