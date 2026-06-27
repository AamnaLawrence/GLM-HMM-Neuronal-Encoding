function [logp] = runF_test_GLMHMM_multi_stable(Ahat, WhatAll, xdatAll, ydatAll)
% Forward step run for testing log-likelihood on test trial
% 
% Inputs:
%   Ahat [state x state]- Transition matrix
%   WhatAll [1 x neurons] - cell array such that each cell is weight matrix [predictors x states] 
%   xdatAll [1 x neurons] - cell array such that each cell is predictor matrix [predictors x T]
%   ydatAll [1 x neurons] - cell array such that each cell is a spike train [1 x T] 
% Outputs:
%   logp     : total log-likelihood

nNeurons = numel(WhatAll);
nStates  = size(Ahat,1);
nT       = size(xdatAll{1},2);

%% 1. Compute log-likelihood per state per time bin
logpy_total = zeros(nT, nStates);
for iNeuron = 1:nNeurons
    What = WhatAll{iNeuron};   
    xdat = xdatAll{iNeuron};  
    ydat = ydatAll{iNeuron};  

    eta = What' * xdat;        
    eta = min(max(eta, -30), 50); % clamp
    lambda = exp(eta);
    lambda = max(lambda, 1e-12);  % avoid log(0)

    logpy = ydat .* log(lambda) - lambda; 
    logpy_total = logpy_total + logpy';   % sum across neurons
end

%% 2. Forward pass with scaling
logalpha = zeros(nStates, nT);
logcs = zeros(1,nT);  % scaling factors

logpi0 = log(ones(1,nStates)/nStates); % log initial probs
logalpha(:,1) = logpi0' + logpy_total(1,:)';
logcs(1) = logsumexp(logalpha(:,1),1);
logalpha(:,1) = logalpha(:,1) - logcs(1);

for t = 2:nT
    % logsumexp over previous states
    tmp = logalpha(:,t-1) + log(Ahat)';  
    logalpha(:,t) = logsumexp(tmp,1)' + logpy_total(t,:)';
    logcs(t) = logsumexp(logalpha(:,t),1);
    logalpha(:,t) = logalpha(:,t) - logcs(t); % scale
end

logp = sum(logcs);

end


