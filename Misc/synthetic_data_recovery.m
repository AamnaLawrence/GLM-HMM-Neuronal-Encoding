%% Main script to generate fake data and test parameter recovery
clc;
clear all;


%% Generate simulated dataset
% Parameters
rng(1);                  

nStates  = 4;
nNeurons = 74;
nX       = 57;                  % Number of GLM predictors
nT       = 96000;                % Number of time bins
binSize  = 0.05;                 % 50 ms

% Ground-truth transition matrix A
alpha_diag = 10;
alpha_full = 1;

A_true = gamrnd(alpha_full*ones(nStates) + ...
                alpha_diag*eye(nStates), 1);

A_true = A_true ./ sum(A_true, 2);

A_true = 1*eye(nStates)+.2*rand(nStates); %from Pillow lab
A_true = A_true ./ sum(A_true,2);  % each row sums to 1

% Ground-truth GLM weights W
% Dimensions: predictors x neurons x states
W_true = zeros(nX, nNeurons, nStates);

% Baseline firing rates in Hz
FR_base = logspace(log10(0.3), log10(10), nNeurons);

% Small state-dependent offsets to help distinguish states
stateBias = linspace(-0.25, 0.25, nStates);

for k = 1:nStates

    % Intercept: log expected counts per bin
    W_true(1,:,k) = log(FR_base * binSize) + stateBias(k);

    % Remaining predictor weights
    W_true(2:end,:,k) = 0.10 * randn(nX-1, nNeurons);

end

% Generate synthetic predictors X
% X dimensions: predictors x time bins
X_true = randn(nX, nT);

% First predictor is the intercept
X_true(1,:) = 1;
X_true=X_true';
% 4. Generate ground-truth latent state sequence
z_true = zeros(1, nT);

% Initial state
z_true(1) = randsample(nStates,1);

for t = 2:nT
    p = A_true(z_true(t-1), :);
    z_true(t) = randsample(nStates,1,true,p);
end

% Generate synthetic Poisson spike counts
% Y_true dimensions: time bins x neurons
Y_true = zeros(nT, nNeurons);

for k = 1:nStates

    idx = find(z_true == k);
    eta = X_true(idx,:) * W_true(:,:,k);
    lambda = exp(eta);
    Y_true(idx,:) = poissrnd(lambda);

end

% Inspect the simulated dataset
FR_true = mean(Y_true, 1) / binSize;

fprintf('Mean firing rate: %.3f Hz\n', mean(FR_true));
fprintf('Minimum neuron rate: %.3f Hz\n', min(FR_true));
fprintf('Maximum neuron rate: %.3f Hz\n', max(FR_true));

fprintf('Maximum observed count per bin: %d\n', max(Y_true(:)));

figure;
histogram(FR_true);
xlabel('Mean firing rate (Hz)');
ylabel('Number of neurons');
title('Synthetic firing-rate distribution');

figure;
imagesc(z_true);
colormap(parula)
xlabel('Time bin');
ylabel('Ground-truth state');
title('Synthetic latent state sequence');
% Count bins assigned to each state
stateCounts = histcounts(z_true, 0.5:1:(nStates+0.5));

% Convert counts to percentages
statePercent = 100 * stateCounts / numel(z_true);

% Display results
T_state = table((1:nStates)', stateCounts', statePercent', ...
    'VariableNames', {'State', 'NumBins', 'Percent'});
disp(T_state);

% Plot distribution
figure;
bar(1:nStates, statePercent);
xticks(1:nStates);
xlabel('Ground-truth state');
ylabel('Time bins (%)');
title('Distribution of latent states');
%% Fit a sharded-state GLM-HMM on the fake data

numStates=nStates;  
results_fake = struct();
A0 = 1*eye(numStates)+.2*rand(numStates); 
A0 = A0 ./ sum(A0,2);  % each row sums to 1
X_all_neuron = cell(1,nNeurons);
y_neuron = cell(1,nNeurons);
W0_neuron = cell(1,nNeurons);
optimal_lambda=nan(1,nNeurons);
for neuron_id=1:nNeurons
    
    y= Y_true(:,neuron_id);
    % === Prepare the feature matrix by combining all predictors ===
    X_all = X_true;
    
    X_all_neuron{neuron_id}=X_all';

    % === Initialize params ===
    
    opts = struct('alpha', 0, 'nfolds', 5,'nlambda',50,'standardize',false);
    cv_fit = cvglmnet(X_all(:,2:end), y, 'poisson', opts);
    optimal_lambda(neuron_id) = cv_fit.lambda_min;
    Wglm_ml = glmnetCoef(cv_fit.glmnet_fit, optimal_lambda(neuron_id));
    W0 = Wglm_ml + randn(size(X_all,2),numStates)*norm(Wglm_ml)*.25; %smart initialization    
    W0_neuron{neuron_id}=W0;
    y_neuron{neuron_id}=y';
end

X_all_neuron = X_all_neuron(~cellfun(@isempty,X_all_neuron));
y_neuron = y_neuron(~cellfun(@isempty,y_neuron));
W0_neuron = W0_neuron(~cellfun(@isempty,W0_neuron));
optimal_lambda = optimal_lambda(~isnan(optimal_lambda));


[Ahat, What, logp,logpTrace,sqErr,jj,dlogp,gams,all_weights_iter]=runEMforGLMHMM_multi(A0,W0_neuron,X_all_neuron,y_neuron, optimal_lambda);
finalLogLikelihood=logpTrace(end);
logLs = finalLogLikelihood;

% Compute # of params 
nCov = size(X_all,2);
p = numStates*(numStates-1) + numStates*nCov*length(y_neuron);  % transition + GLM weights

% Compute AIC/BIC 
T = size(X_all,1);  % number of bins
AICs = -2*finalLogLikelihood + 2*p;
BICs = -2*finalLogLikelihood + p*log(T);
% save results
results_fake.K = numStates;
results_fake.Ahat = Ahat;
results_fake.What = What;
results_fake.gams = gams;
results_fake.logpTrace = logpTrace;
results_fake.finalLogL = logpTrace(end);
results_fake.AIC = AICs;
results_fake.BIC = BICs;
results_fake.sqErr = sqErr;
results_fake.nParams = p;
results_fake.A0=A0;
results_fake.optimal_lambda=optimal_lambda;
results_fake.W0=W0_neuron;
results_fake.all_weights_iter=all_weights_iter;
results_fake.z_true=z_true;
results_fake.A_true=A_true;
results_fake.W_true=W_true;
%% plotting logp trace
close all;
fig = figure;
logpTrace=results_fake.logpTrace(2:end);
plot(2:numel(results_fake.logpTrace), -logpTrace, 'o-','LineWidth',2);
xlabel('Iteration');
ylabel('Negative log-likelihood');
box off
manuscriptify(fig,3,4)

fig=figure;
sqErr=results_fake.sqErr(2:end);
plot(2:numel(results_fake.sqErr), sqErr, 'o-','LineWidth',2);
xlabel('Iteration');
ylabel('Squared parameter change');
box off
manuscriptify(fig,3,4)

%% State correlation
K = nStates; %
[~, z_rec] = max(results_fake.gams, [], 1);
T = numel(z_true);

% One-hot encode the true and recovered state sequences
Z_true = zeros(T,K);
Z_rec  = zeros(T,K);

for k = 1:K
    Z_true(:,k) = (z_true(:) == k);
    Z_rec(:,k)  = (z_rec(:) == k);
end

% Correlation between every true and recovered state
R_states = corr(Z_true, Z_rec);

figure;
imagesc(R_states, [-1 1]);
axis square;
colorbar;
xlabel('Recovered state');
ylabel('True state');
xticks(1:K);
yticks(1:K);
title('True vs. recovered state correspondence');
P = perms(1:K);
meanCorr = zeros(size(P,1),1);

for i = 1:size(P,1)
    meanCorr(i) = mean(diag(R_states(:,P(i,:))));
end

[bestCorr, bestIdx] = max(meanCorr);
bestPerm = P(bestIdx,:);

R_aligned = R_states(:,bestPerm);

fprintf('Mean state-wise correlation after alignment: %.3f\n', ...
    mean(diag(R_aligned)));

figure;
imagesc(R_aligned, [-1 1]);
axis square;
colorbar;
xlabel('Recovered state (aligned)');
ylabel('True state');
xticks(1:K);
yticks(1:K);
title('Aligned true vs. recovered state correspondence');
bestPerm
%% State alignment accuracy

invPerm = zeros(1,K);
invPerm(bestPerm) = 1:K;

z_rec_aligned = invPerm(z_rec);

accuracy = mean(z_rec_aligned(:) == z_true(:));
fprintf('Aligned state accuracy: %.2f%%\n', 100*accuracy);
C = confusionmat(z_true(:), z_rec_aligned(:), ...
    'Order', 1:K);

disp(C);
%% A recovery
A_rec=results_fake.Ahat;
A_rec_aligned = A_rec(bestPerm,bestPerm);
A_corr = corr(A_true(:), A_rec_aligned(:), ...
              'Type','Pearson');
A_rel_error = norm(A_true - A_rec_aligned,'fro') / ...
              norm(A_true,'fro');
A_Frobenius = norm(A_true - A_rec_aligned,'fro');
fprintf('A correlation = %.4f\n', A_corr);
fprintf('A relative error = %.4f\n', A_rel_error);
fprintf('A Frobenius error = %.4f\n', A_Frobenius);

%% GLM weight recovery

weight_score = zeros(size(permsK,1),1);
W_rec=results_fake.What;


perm = bestPerm;
score = 0;

for n = 1:nNeurons

    Wt = squeeze(W_true(:,n,:));
    Wr = W_rec{n}(:,perm);

    score = score + corr(Wt(:),Wr(:),'Type','Pearson');

end
weight_score = score;

fprintf('Mean W correlation with W-optimal permutation: %.4f\n', ...
    weight_score/nNeurons);
figure;
histogram(weight_score./nNeurons);
xlabel('W coefficient correlation');
ylabel('Number of neurons');
xline(0,'--');
W_rec_aligned = cell(size(W_rec));
for n = 1:nNeurons
    W_rec_aligned{n} = W_rec{n}(:,bestPerm);
end

%% State occupancy comparison

K = size(A_true, 1);

% Ground-truth state sequence
z_true = z_true(:);

% Recovered state sequence: most probable state at each time bin
[~, z_rec] = max(results_fake.gams, [], 1);
z_rec = z_rec(:);

% Calculate number of bins per state
trueCounts = accumarray(z_true, 1, [K 1]);
recCounts  = accumarray(z_rec,  1, [K 1]);

% Convert to percentages
truePct = 100 * trueCounts / numel(z_true);
recPct  = 100 * recCounts  / numel(z_rec);

% Display results
T_occ = table((1:K)', trueCounts, truePct, recCounts, recPct, ...
    'VariableNames', {'State', 'TrueBins', 'TruePercent', ...
                      'RecoveredBins', 'RecoveredPercent'});

disp(T_occ);

% Plot occupancy comparison
figure;
bar([truePct, recPct]);
xticks(1:K);
xlabel('State');
ylabel('Time bins (%)');
legend('Ground truth', 'Recovered', 'Location', 'best');
title('Ground-truth vs recovered state occupancy');
