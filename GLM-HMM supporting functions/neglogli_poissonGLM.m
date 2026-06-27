function [negL,dnegL,H] = neglogli_poissonGLM(wts,X,Y)
% Compute negative log-likelihood of data under logistic regression model,
% plus gradient and Hessian
%
% Inputs:
% -------
% wts [feature x 1] - regression weights
%   X [T x feature] - regressors
%   Y [T x 1] - output (binary vector of 1s and 0s).
%
% Outputs:
% --------
%    negL [1 x 1] - negative loglikelihood
%   dnegL [feature x 1] - gradient
%       H [feature x feature] - Hessian (2nd deriv matrix)

% Compute projection of inputs onto GLM weights for each class

xproj = X*wts;
lambda = exp(xproj);


if nargout <= 1     % Evaluate neglogli only
    
    %negL = -Y'*xproj + sum(softplus(xproj)); % neg log-likelihood
    negL = -sum(Y.*xproj - lambda - gammaln(Y+1));
elseif nargout == 2 % Evaluate gradient

    % [f,df] = softplus(xproj);  % evaluate log-normalizer & deriv
    % negL = -Y'*xproj + sum(f); % neg log-likelihood
    % dnegL = X'*(df-Y);         % gradient
    negL  = -sum(Y.*xproj - lambda - gammaln(Y+1));
    dnegL = X' * (lambda - Y);

elseif nargout == 3 % Evaluate Hessian
    
    % [f,df,ddf] = softplus(xproj); % evaluate log-normalizer & derivs
    % negL = -Y'*xproj + sum(f);    % neg log-likelihood
    % dnegL = X'*(df-Y);            % gradient
    % H = X'*bsxfun(@times,X,ddf);  % Hessian

    negL  = -sum(Y.*xproj - lambda - gammaln(Y+1));
    dnegL = X' * (lambda - Y);
    H     = X' * bsxfun(@times, X, lambda);

end
