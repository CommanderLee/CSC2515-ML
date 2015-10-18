function [f, df, y] = logistic(weights, data, targets, hyperparameters)
% Calculate log likelihood and derivatives with respect to weights.
%
% Note: N is the number of examples and 
%       M is the number of features per example.
%
% Inputs:
% 	weights:    (M+1) x 1 vector of weights, where the last element
%               corresponds to bias (intercepts).
% 	data:       N x M data matrix where each row corresponds 
%               to one data point.
%	targets:    N x 1 vector of binary targets. Values should be either 0 or 1.
%   hyperparameters: The hyperparameter structure
%
% Outputs:
%	f:             The scalar error value.
%	df:            (M+1) x 1 vector of derivatives of error w.r.t. weights.
%   y:             N x 1 vector of probabilities. This is the output of the classifier.
%

%TODO: finish this function
%% Get y and z
[N M] = size(data);

% z = w^T * x + w_0
z = data * weights(1:M) + weights(M+1);
% y(x) = sigma(z)
y = sigmoid(z);

%% Get f, df
[f, frac_correct] = evaluate(targets, y);
% f = sum((targets - 1) .* (-z) - log(y));

% dE(w,b)/dw_i = /sum_n { x_i^(n) * (-targets^(n)-y^(n)) }
dE_dwi = data' * (-targets - y);
dE_db = sum(-targets - y);
df = [dE_dwi; dE_db];

end
