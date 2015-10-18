%% Clear workspace.
clear all;
close all;

%% Load data.
load mnist_train;
load mnist_train_small;
load mnist_valid;

%% TODO: Initialize hyperparameters.
% Learning rate
hyperparameters.learning_rate = 1;
% Weight regularization parameter
hyperparameters.weight_regularization = 1;
% Number of iterations
hyperparameters.num_iterations = 500;
% Logistics regression weights
% TODO: Set random weights.
[N M] = size(train_inputs);
weights = randn(M+1, 1);

%% Verify that your logistic function produces the right gradient, diff should be very close to 0
% this creates small random data with 20 examples and 10 dimensions and checks the gradient on
% that data.
nexamples = 20;
ndimensions = 10;
diff = checkgrad('logistic', ...
	             randn((ndimensions + 1), 1), ...   % weights
                 0.001,...                          % perturbation
                 randn(nexamples, ndimensions), ... % data        
                 rand(nexamples, 1), ...            % targets
                 hyperparameters)                   % other hyperparameters

%% Begin learning with gradient descent.
% N = size(mnist_train, 0);
learn_rate_set = [0.5 0.75 1];
num_iter_set = [100 300 500];
results = zeros(9, 6);

for hyperID = 1:9
    hyperparameters.learning_rate = learn_rate_set(fix((hyperID-1)/3)+1);
    hyperparameters.num_iterations = num_iter_set(mod(hyperID, 3)+1);
    
    ce_t_sum = 0.0;
    error_t_sum = 0.0;
    ce_v_sum = 0.0;
    error_v_sum = 0.0;
    
    multi_times = 20;
    for mt = 1:multi_times
        for t = 1:hyperparameters.num_iterations

            %% TODO: You will need to modify this loop to create plots etc.

            % Find the negative log likelihood and derivative w.r.t. weights.
            [f, df, predictions] = logistic(weights, ...
                                            train_inputs, ...
                                            train_targets, ...
                                            hyperparameters);

            [cross_entropy_train, frac_correct_train] = evaluate(train_targets, predictions);

            % Find the fraction of correctly classified validation examples.
            [temp, temp2, frac_correct_valid] = logistic(weights, ...
                                                         valid_inputs, ...
                                                         valid_targets, ...
                                                         hyperparameters);

            if isnan(f) || isinf(f)
                error('nan/inf error');
            end

            %% Update parameters.
            weights = weights - hyperparameters.learning_rate .* df / N;

            predictions_valid = logistic_predict(weights, valid_inputs);
            [cross_entropy_valid, frac_correct_valid] = evaluate(valid_targets, predictions_valid);

            %% Print some stats.
%             fprintf(1, 'ITERATION:%4i   NLOGL:%4.2f TRAIN CE %.6f TRAIN FRAC:%2.2f VALIC_CE %.6f VALID FRAC:%2.2f\n',...
%                     t, f/N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100);
        end
        ce_t_sum = ce_t_sum + cross_entropy_train;
        error_t_sum = error_t_sum + (1 - frac_correct_train);
        ce_v_sum = ce_v_sum + cross_entropy_valid;
        error_v_sum = error_v_sum + (1 - frac_correct_valid);
    end
    results(hyperID, :) = [hyperparameters.learning_rate, hyperparameters.num_iterations, ...
        ce_t_sum / multi_times, error_t_sum / multi_times, ce_v_sum / multi_times, error_v_sum / multi_times];
end
disp(results);


