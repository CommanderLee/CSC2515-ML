%% Initialize
clear all;
load digits;

%% Set parameters
inputs_train = train3;
K = 2;
iters = 100;
minVary = 0.01;
plotFlag = 0;

%% Run mog
train_num = 1;
logProbList = zeros(train_num, 1);
muList = zeros(256, 2 * train_num);
varyList = zeros(256, 2 * train_num);
for t=1:train_num
    [p,mu,vary,logProbX] = mogEM(inputs_train,K,iters,minVary,plotFlag);
    disp(logProbX(iters));
    logProbList(t) = logProbX(iters);
    muList(:,t*2-1:t*2) = mu;
    varyList(:,t*2-1:t*2) = vary;
end
fprintf('Mean logProb = %f\n', mean(logProbList));

%% Plot digit
bestNum = 1;
mu = muList(:, 2*bestNum-1:2*bestNum);
vary = varyList(:, 2*bestNum-1:2*bestNum);
visualize_digits(mu);
visualize_digits(vary);