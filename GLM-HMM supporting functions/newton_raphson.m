function [w, logp_trace] = newton_raphson(w0, lfunc, maxIter, tol)
% Pure Newton-Raphson convergence 
% Inputs
% -------
%  w0 [predictors x 1]- Per-neuron GLM weights for each state
%  lfunc - function computing per-neuron log-likelihood, with first and second gradient
%  maxIter - # of iterations to run to reach convergence
%  tol - tolerance to break the optimization loop if change in
%  log-likelihood is below the value

% Outputs
% -------
% w [predictors x 1]- Updated per-neuron GLM weights for each state
% logp_trace [maxIter x 1] - Negative log-likelihood for each iteration

    w = w0;
    logp_trace = zeros(maxIter,1);

    for k = 1:maxIter

        [f, g, H] = lfunc(w);  

        logp_trace(k) = -f;     % store log-likelihood

        % Newton step
        step = -H \ g;

        % Update
        w = w + step;

        % Convergence check
        if norm(g) < tol
            logp_trace = logp_trace(1:k);
            break;
        end
    end
end