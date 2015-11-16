clear all;
load digits;
x = [train2, train3];
%-------------------- Add your code here --------------------------------
% Train a MoG model with 20 components on all 600 training vectors
% with both original initialization and your kmeans initialization. 

%% Set parameters
K = 20;
iters = 100;
minVary = 0.01;
plotFlag = 0;

%% Run mog
train_num = 10;
logProbList = zeros(train_num, 1);
muList = zeros(256, 20 * train_num);
varyList = zeros(256, 20 * train_num);
for t=1:train_num
    [p, mu, vary, logProbX] = mogEM(x, K, iters, minVary, plotFlag);
    disp(logProbX(iters));
    logProbList(t) = logProbX(iters);
    muList(:, (t-1)*20+1:t*20) = mu;
    varyList(:, (t-1)*20+1:t*20) = vary;
end
fprintf('Mean logProb = %f\n', mean(logProbList));