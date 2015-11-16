% Choose the best mixture of Gaussian classifier you have, compare this
% mixture of Gaussian classifier with the neural network you implemented in
% the last assignment.


% Train neural network classifier. The number of hidden units should be
% equal to the number of mixture components.

% Show the error rate comparison.

%-------------------- Add your code here --------------------------------
%% Run NN
init_nn();

num_hiddens = 30;
eps = 0.02;
momentum = 0.5;

train_num = 10;
for i=1:train_num
    train_nn();
end

%% Show hidden weights of NN
visualize_digits(dW1);

%% Test on NN
fprintf('Neural Networks: ');
test_nn();

%% Run MoG
load digits;

K = 15;
iters = 100;
minVary = 0.01;
plotFlag = 0;

x1 = [train2];
[p1, mu1, vary1, logProbX1] = mogEM(x1, K, iters, minVary, plotFlag);
x2 = [train3];
[p2, mu2, vary2, logProbX2] = mogEM(x2, K, iters, minVary, plotFlag);

%% Show hidden weights of MoG
mu12 = zeros(256, 30);
mu12(:, 1:15) = mu1;
mu12(:, 16:30) = mu2;
visualize_digits(mu12);

%% Test on MoG
[inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test] = load_data();

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
fprintf('Mixture of Gaussians: errorNum=%d, errorRate=%.4f\n', errorNum, errorTest(i));