%% Run NN with exist code
%% Initialize
init_nn();

%% Change parameters
eps = 0.02;
momentum = 0.5;

%% Train
train_num = 10;
for i=1:train_num
    train_nn();
end
fprintf('num_hiddens:%d: ', num_hiddens);
test_nn();
