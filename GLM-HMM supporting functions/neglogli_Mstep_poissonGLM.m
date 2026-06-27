function [negL,dnegL,H] = neglogli_Mstep_poissonGLM(wts,X,Y,pstate)
% Compute negative log-likelihood of data under Poisson regression model,
% plus gradient and Hessian
% Inputs:
% -------
%    wts [predictors x 1] - regression weights
%      X [T x feature] - regressors
%      Y [T x 1] - neuron spikes
% pstate [T x 1] - probability of being in state at each time bin
%
% Outputs:
% -------
%   negL [1 x 1] - negative log-likelihood
%  dnegL [predictors x 1] - gradient of negative log-likelihood
%      H [predictors x predictors] - Hessian (2nd derivative matrix)

% Compute projection of stimuli onto weights
xproj = X*wts;
lambda = exp(xproj);

if nargout <= 1    % Evaluate neglogli only

    negL = -(pstate.*Y)'*xproj + sum(pstate.*lambda);

elseif nargout == 2 % Evaluate gradient
 
    negL  = -(pstate.*Y)'*xproj + sum(pstate.*lambda);
    dnegL = X' * ((lambda - Y).*pstate);

elseif nargout == 3 % Evaluate gradient & Hessian
    
    negL  = -(pstate.*Y)'*xproj + sum(pstate.*lambda);
    dnegL = X' * ((lambda - Y).*pstate);
    H     = X' * bsxfun(@times,X,pstate.*lambda);
end


