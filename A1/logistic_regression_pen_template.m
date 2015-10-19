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
hyperparameters.weight_regularization = 0.5;
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
diff = checkgrad('logistic_pen', ...
	             randn((ndimensions + 1), 1), ...   % weights
                 0.001,...                          % perturbation
                 randn(nexamples, ndimensions), ... % data        
                 rand(nexamples, 1), ...            % targets
                 hyperparameters)                   % other hyperparameters

%% Begin learning with gradient descent.
% N = size(mnist_train, 0);
weight_reg_set = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0];
hyperNum = size(weight_reg_set, 2);
results = zeros(hyperNum, 5);

testing = 1;
if testing
   hyperNum = 1; 
end

for hyperID = 1:hyperNum
    if ~testing
        hyperparameters.weight_regularization = weight_reg_set(hyperID);
    end

    multi_times = 20;
    
    ce_train = zeros(1, multi_times);
    er_train = zeros(1, multi_times);
    ce_valid = zeros(1, multi_times);
    er_valid = zeros(1, multi_times);
    ce_test = zeros(1, multi_times);
    er_test = zeros(1, multi_times);
%     colors = ['r', 'b', 'k', 'g', 'c'];
    for mt = 1:multi_times
        weights = randn(M+1, 1) * 0.1;
        
%         y_train = zeros(1, hyperparameters.num_iterations);
%         y_valid = zeros(1, hyperparameters.num_iterations);
        for t = 1:hyperparameters.num_iterations

            % Find the negative log likelihood and derivative w.r.t. weights.
            [f, df, predictions] = logistic_pen(weights, ...
                                            train_inputs, ...
                                            train_targets, ...                                     
                                            hyperparameters);

            [cross_entropy_train, frac_correct_train] = evaluate(train_targets, predictions);
%             [cross_entropy_train, frac_correct_train] = evaluate(train_targets_small, predictions);
            
            % Find the fraction of correctly classified validation examples.
            [temp, temp2, frac_correct_valid] = logistic_pen(weights, ...
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
%             y_train(t) = cross_entropy_train;
%             y_valid(t) = cross_entropy_valid;
        end
        ce_train(mt) = cross_entropy_train;
        er_train(mt) = 1 - frac_correct_train;
        ce_valid(mt) = cross_entropy_valid;
        er_valid(mt) = 1 - frac_correct_valid;
        
        if testing
            load mnist_test;
            predictions_test = logistic_predict(weights, test_inputs);
            [cross_entropy_test, frac_correct_test] = evaluate(test_targets, predictions_test);
            ce_test(mt) = cross_entropy_test;
            er_test(mt) = 1 - frac_correct_test;
        end
        
        %% Plot train & validation
%         figure(mt);
% 
%         x = 1:hyperparameters.num_iterations;
%         plot(x, y_train, strcat(colors(mt),'-'), x, y_valid, strcat(colors(mt), ':'));
%         xlabel('iteration');
%         ylabel('cross entropy');
%         legend('train','validation', -1);
% %         title('mnist\_train\_small');
%         title('mnist\_train');
%         hold on;
    end
    results(hyperID, :) = [hyperparameters.weight_regularization, ...
        mean(ce_train), mean(er_train), mean(ce_valid), mean(er_valid)];
end
disp(results);

%% Plot train & valid
x = 1:hyperNum;

% figure(3);
% plot(x, results(:,2), 'r.-', x, results(:,4), 'bo:');
% xlabel('\lambda = 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0');
% ylabel('cross entropy');
% legend('train','validation', -1);
% title('mnist\_train: cross entropy');
% % title('mnist\_train\_small: cross entropy');
% 
% figure(4);
% plot(x, results(:,3), 'r.-', x, results(:,5), 'bo:');
% xlabel('\lambda = 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0');
% ylabel('classification error');
% legend('train','validation', -1);
% title('mnist\_train: classification error');
% % title('mnist\_train\_small: classification error');

%% Plot test
if testing
    disp(ce_test');
    disp(er_test');
    
    figure(multi_times+1);
    x = 1:multi_times;
    y1 = ce_test;
    y2 = er_test;
    [AX,H1,H2] = plotyy(x, y1, x, y2, 'plot');
    set(AX(1), 'XColor','k','YColor','b');
    set(AX(2), 'XColor','k','YColor','r');
    HH1 = get(AX(1), 'Ylabel');
    set(HH1, 'String', 'test\_cross\_entropy');
    set(HH1, 'color', 'b');
    HH2=get(AX(2),'Ylabel');
    set(HH2, 'String', 'test\_classification\_error');
    set(HH2, 'color', 'r');
    set(H1, 'LineStyle', '-');
    set(H1, 'color', 'b');
    set(H2, 'LineStyle', ':');
    set(H2, 'color', 'r');
	legend([H1,H2], {'test\_cross\_entropy';'test\_classification\_error'});
    xlabel('test id(1 to 20)');
end
