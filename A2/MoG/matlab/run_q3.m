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
