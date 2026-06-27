function [Ahat,What,logp,logpTrace,sqErr,jj,dlogp,gams,all_weights_iter] = runEMforGLMHMM_multi(A0,W0,xdat,ydat,lambda, optsEM)
% EM run with M-step convergence using Trust-region method
% Inputs:
% -------
%    A0 [state x state] - initial state transition matrix
%    W0 [1 x neurons] - each cell array is initial weights per state [predictors x state]
%  xdat [1 x neurons] - each cell array is a predictor matrix [predictors x T]
%  ydat [1 x neurons] - each cell array is a spike vector [1 x T]
%  opts [struct] - optimization params (optional)
%       .maxiter - maximum # of iterations 
%       .dlogptol - stopping tol for change in log-likelihood 
%       .display - how often to report log-likelihood
%
% Outputs:
% --------
%      Ahat [state x state] - estimated state transition matrix
%      What [1 x neurons] - each cell array is the estimated GLM regression per state [predictor x state]
%      logp [1 x 1] - final log-likelihood
% logpTrace [1 x maxiter] - trace of log-likelihood during EM
%        jj [1 x 1] - final iteration
%     dlogp [1 x 1] - final change in log-likelihood
%      gams [state x T] - marginal state probabilities from last E step
%     all_weights_iter [neurons x maxiter]- weights across all states per neuron for every iteration

% Set EM optimization params if necessary
if nargin < 6
    optsEM.maxiter = 100; 
    optsEM.dlogptol = 0.1; 
    optsEM.display = inf;
end
if ~isfield(optsEM,'display') || isempty(optsEM.display)
    optsEM.display = inf;
end

% Initialize params
nStates = size(A0,1);
nNeurons= length(ydat);
Ahat = A0;
What = W0;
all_weights_iter = zeros(nNeurons*length(What{1}(:)),optsEM.maxiter);
% Set up variables for EM
logpTrace = zeros(optsEM.maxiter,1); % trace of log-likelihood
dlogp = inf; % change in log-likelihood
logpPrev = -inf; % prev value of log-likelihood
jj = 1; % counter
thetaPrev = []; %added for convergence check

% Set up loss function for GLM log-likelihood
optsFminunc = optimoptions(@fminunc,'Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective','display','off');

while (jj <= optsEM.maxiter) %&& (abs(dlogp)>optsEM.dlogptol)
    
    % --- run E step  -------
  
    [logp,gams,xisum] = runFB_GLMHMM_multi_stable(Ahat,What,xdat,ydat);
    logpTrace(jj) = logp;
    
    % --- run M step  -------
   
    % Update transition matrix
    Ahat = xisum ./ sum(xisum,2); % normalize each row to sum to 1
    
    % Update GLM weights (could be parallelized over states)
   
    for neuron_id=1:nNeurons 
        for iistate = 1:nStates
                init_w = What{neuron_id}(:,iistate)
                init_w = init_w(:);
                lfunc = @(w) neglogli_Mstep_poissonGLM_ridge(w, ...
                              xdat{neuron_id}', ...
                              ydat{neuron_id}', ...
                              gams(iistate,:)', ...
                              lambda(neuron_id));
                
                What{neuron_id}(:,iistate) = fminunc(lfunc, init_w, optsFminunc);
        end

    end
    
    % --- Compute squared error --- added for convergence check
    theta = [Ahat(:)]; 
    all_weights=[];
    for n = 1:nNeurons
        theta = [theta; What{n}(:)];
        all_weights = [all_weights; What{n}(:)];
    end
    all_weights_iter(:, jj)=all_weights;
    if jj > 1
        sqErr(jj) = sum((theta - thetaPrev).^2);
    end
    thetaPrev = theta;

    % ---  Display progress ----
    if mod(jj,optsEM.display)==0
        fprintf('EM iter %d:  logli = %-.6g\n',jj,logp);

    end
    
    % Update counter and log-likelihood change
    dlogp = logp-logpPrev; % change in log-likelihood
    logpPrev = logp; % previous log-likelihood

    jj = jj+1;  
    
    if dlogp<-1e-6
        warning('Log-likelihood decreased during EM!');
        fprintf('dlogp = %.5g\n', dlogp);
    end

end
jj = jj-1;



