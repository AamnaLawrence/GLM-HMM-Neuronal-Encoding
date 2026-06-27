function [f, g, H] = neglogli_Mstep_poissonGLM_ridge(w, X, y, gam, lambda)
% Applies ridge penalty to the negative log-likelihood, its gradient and
% Hessian
% Inputs:
% -------
%      w [predictors x 1] - regression weights
%      X [T x feature] - regressors
%      y [T x 1] - neuron spikes
%      gam [T x 1] - probability of being in state at each time bin
%      lambda [1 x 1] - per-neuron ridge penalty
%
% Outputs:
% -------
%   f [1 x 1] - negative log-likelihood after penalty
%   g [predictors x 1] - gradient of negative log-likelihood after penalty
%   H [predictors x predictors] - Hessian (2nd derivative matrix) after
%   penalty

    w = w(:);  % ensure column

    % --- Base function (needs to return Hessian too) ---
    [f, g, H] = neglogli_Mstep_poissonGLM(w, X, y, gam); 

    % --- Ridge penalty ---
    % not penalize the intercept
    mask = [0;ones(length(w)-1,1)];  
    
    f = f + lambda * sum((mask .* w).^2);
    g = g + 2 * lambda * (mask .* w);
    H = H + 2 * lambda * diag(mask);   % Hessian of ridge

    % NaN/Inf check
    if any(isnan(f)) || any(isinf(f)) || any(isnan(g)) || any(isinf(g)) || any(any(isnan(H))) || any(any(isinf(H)))
        f = 1e12;
        g = zeros(size(w));
        H = 2*lambda*diag(mask);  % fallback
    end
end
