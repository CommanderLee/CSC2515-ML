clear all;
load digits;

errorTrain = zeros(1, 4);
errorValidation = zeros(1, 4);
errorTest = zeros(1, 4);
numComponent = [2, 5, 15, 25];

for i = 1 : 4
    K = numComponent(i);
% Train a MoG model with K components for digit 2
%-------------------- Add your code here --------------------------------
    iters = 100;
    minVary = 0.01;
    plotFlag = 0;
    
    x1 = [train2];
    [p1, mu1, vary1, logProbX1] = mogEM(x1, K, iters, minVary, plotFlag);

% Train a MoG model with K components for digit 3
%-------------------- Add your code here --------------------------------
    x2 = [train3];
    [p2, mu2, vary2, logProbX2] = mogEM(x2, K, iters, minVary, plotFlag);

% Caculate the probability P(d=1|x) and P(d=2|x), 
% classify examples, and compute the error rate
% Hints: you may want to use mogLogProb function
%-------------------- Add your code here --------------------------------
    [inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test] = load_data();
    %% Training data set:
    % logP(x|d=1)
    [logProb_trainxd1] = mogLogProb(p1, mu1, vary1, inputs_train);
    % logP(x|d=2)
    [logProb_trainxd2] = mogLogProb(p2, mu2, vary2, inputs_train);
    % P(d=1|x) and P(d=2|x)
    prob_traindx1 = exp(logProb_trainxd1) ./ (exp(logProb_trainxd1) + exp(logProb_trainxd2));
    prob_traindx2 = 1 - prob_traindx1;
    % Error: target: 0:2, 1:3, prob_traindx1: 0:not 2, 1:is 2.
    errorNum = sum(target_train == prob_traindx1);
    errorTrain(i) = errorNum / size(inputs_train, 2);
    fprintf('i=%d, train: errorNum=%d, errorRate=%.4f\n', i, errorNum, errorTrain(i));
    
    %% Validation data set:
    % logP(x|d=1)
    [logProb_validxd1] = mogLogProb(p1, mu1, vary1, inputs_valid);
    % logP(x|d=2)
    [logProb_validxd2] = mogLogProb(p2, mu2, vary2, inputs_valid);
    % P(d=1|x) and P(d=2|x)
    prob_validdx1 = exp(logProb_validxd1) ./ (exp(logProb_validxd1) + exp(logProb_validxd2));
    prob_validdx2 = 1 - prob_validdx1;
    % Error: target: 0:2, 1:3, prob_validdx1: 0:not 2, 1:is 2.
    errorNum = sum(target_valid == prob_validdx1);
    errorValidation(i) = errorNum / size(inputs_valid, 2);
    fprintf('     valid: errorNum=%d, errorRate=%.4f\n', errorNum, errorValidation(i));
    
    %% Testing data set:
    % logP(x|d=1)
    [logProb_testxd1] = mogLogProb(p1, mu1, vary1, inputs_test);
    % logP(x|d=2)
    [logProb_testxd2] = mogLogProb(p2, mu2, vary2, inputs_test);
    % P(d=1|x) and P(d=2|x)
    prob_testdx1 = exp(logProb_testxd1) ./ (exp(logProb_testxd1) + exp(logProb_testxd2));
    prob_testdx2 = 1 - prob_testdx1;
    % Error: target: 0:2, 1:3, prob_testdx1: 0:not 2, 1:is 2.
    errorNum = sum(target_test == prob_testdx1);
    errorTest(i) = errorNum / size(inputs_test, 2);
    fprintf('     test: errorNum=%d, errorRate=%.4f\n', errorNum, errorTest(i));
end

%% Plot the error rate
%-------------------- Add your code here --------------------------------
figure(2);
hold on;
plot([1:4], errorTrain, 'g-*');
plot([1:4], errorValidation, 'r-*');
plot([1:4], errorTest, 'b-*');
legend('training set', 'validation set', 'testing set');
ylabel('classification error rate');
xlabel('number of mixture components = 2, 5, 15, 25');
