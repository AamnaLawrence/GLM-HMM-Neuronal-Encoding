function xi = compute_xi(alpha, beta, ll_tz, A, c)
% COMPUTE_XI  Expected transitions for HMM (sum over time)
%
% alpha, beta : [T x state] scaled forward/backward probs
% ll_tz       : [T x state] log emissions
% A           : [state x state] transition matrix
% c           : [T x 1] scaling factors from forward pass
%
% xi          : [state x state x T-1] expected transitions

[T, K] = size(ll_tz);
xi = zeros(K,K,T-1);
logA = log(A + eps);

for t=1:T-1
    % log joint for all i->j at time t
    M = log(alpha(t,:))' + logA + ll_tz(t+1,:) + log(beta(t+1,:));
    % normalize
    M = M - logsumexp_matrix(M);
    xi(:,:,t) = exp(M);
end

end

%% helper for log-sum-exp over a matrix
function s = logsumexp_matrix(M)
m = max(M,[],'all');
s = m + log(sum(exp(M - m),'all') + eps);
end
