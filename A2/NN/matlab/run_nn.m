%% Run NN with exist code
%% Initialize
init_nn();

%% Change parameters
eps = 0.2;
momentum = 0.3;

%% Train
test_num = 10;
for i=1:test_num
    train_nn();
end
fprintf('num_hiddens:%d: ', num_hiddens);
test_nn();
