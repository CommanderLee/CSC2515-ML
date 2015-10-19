%% Clear workspace.
clear all;
close all;

%% Load data.
load mnist_train;
load mnist_train_small;
load mnist_valid;

%% TODO: Initialize hyperparameters.
% Learning rate
hyperparameters.learning_rate = 0.5;
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
learn_rate_set = [0.25 0.5 1];
num_iter_set = [100 300 500];
hyperNum = 9;
results = zeros(hyperNum, 6);

testing = 1;
if testing
   hyperNum = 1; 
end

for hyperID = 1:hyperNum
    if ~testing
        hyperparameters.learning_rate = learn_rate_set(fix((hyperID-1)/3)+1);
        hyperparameters.num_iterations = num_iter_set(mod(hyperID, 3)+1);
    end
    
    ce_train = [];
    er_train = [];
    ce_valid = [];
    er_valid = [];
    ce_test = [];
    er_test = [];
    
    multi_times = 5;
    colors = ['r', 'b', 'k', 'g', 'c'];
    for mt = 1:multi_times
        weights = randn(M+1, 1) * 0.1;
        
        y_train = zeros(1, hyperparameters.num_iterations);
        y_valid = zeros(1, hyperparameters.num_iterations);
        for t = 1:hyperparameters.num_iterations

            % Find the negative log likelihood and derivative w.r.t. weights.
            [f, df, predictions] = logistic(weights, ...
                                            train_inputs, ...
                                            train_targets, ...                                     
                                            hyperparameters);

            [cross_entropy_train, frac_correct_train] = evaluate(train_targets, predictions);
%             [cross_entropy_train, frac_correct_train] = evaluate(train_targets_small, predictions);
            
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
%              fprintf(1, 'ITERATION:%4i   NLOGL:%4.2f TRAIN CE %.6f TRAIN FRAC:%2.2f VALIC_CE %.6f VALID FRAC:%2.2f\n',...
%                      t, f/N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100);
            y_train(t) = cross_entropy_train;
            y_valid(t) = cross_entropy_valid;
        end
        ce_train = [ce_train, cross_entropy_train];
        er_train = [er_train, (1 - frac_correct_train)];
        ce_valid = [ce_valid, cross_entropy_valid];
        er_valid = [er_valid, (1 - frac_correct_valid)];
        
        if testing
            load mnist_test;
            predictions_test = logistic_predict(weights, test_inputs);
            [cross_entropy_test, frac_correct_test] = evaluate(test_targets, predictions_test);
            ce_test = [ce_test cross_entropy_test];
            er_test = [er_test 1 - frac_correct_test];
        end
        
        %% Plot train & validation
%         figure(mt);

        x = 1:hyperparameters.num_iterations;
        plot(x, y_train, strcat(colors(mt),'-'), x, y_valid, strcat(colors(mt), ':'));
        xlabel('iteration');
        ylabel('cross entropy');
        legend('train','validation', -1);
%         title('mnist\_train\_small');
        title('mnist\_train');
        hold on;
    end
    results(hyperID, :) = [hyperparameters.learning_rate, hyperparameters.num_iterations, ...
        mean(ce_train), mean(er_train), mean(ce_valid), mean(er_valid)];
end
disp(results);

%% Plot test
if testing
    disp(ce_test');
    disp(er_test');
    
%     figure(multi_times+1);
%     x = 1:multi_times;
%     y1 = ce_test;
%     y2 = er_test;
%     [AX,H1,H2] = plotyy(x, y1, x, y2, 'plot');
%     set(AX(1), 'XColor','k','YColor','b');
%     set(AX(2), 'XColor','k','YColor','r');
%     HH1 = get(AX(1), 'Ylabel');
%     set(HH1, 'String', 'test\_cross\_entropy');
%     set(HH1, 'color', 'b');
%     HH2=get(AX(2),'Ylabel');
%     set(HH2, 'String', 'test\_classification\_error');
%     set(HH2, 'color', 'r');
%     set(H1, 'LineStyle', '-');
%     set(H1, 'color', 'b');
%     set(H2, 'LineStyle', ':');
%     set(H2, 'color', 'r');
% 	legend([H1,H2], {'test\_cross\_entropy';'test\_classification\_error'});
%     xlabel('test id(1 to 20)');
end
