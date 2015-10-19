function calc_knn()
%% Calculate the results required by A1-2.1

%% Clear workspace.
clear all;
close all;

%% Load data.
load mnist_train;
load mnist_valid;
load mnist_train_small;

%% Initialize
k_set = [1,3,5,7,9];
class_rate = [];

%% Validate
for k = k_set
    [valid_labels] = run_knn(k, train_inputs, train_targets, valid_inputs);
%     [valid_labels] = run_knn(k, train_inputs_small, train_targets_small, valid_inputs);
    data_num = size(valid_targets, 1);
    corr_num = sum(valid_targets == valid_labels);
    class_rate = [class_rate corr_num/data_num];
end
disp(class_rate);
plot(k_set, class_rate, 'b*-');
legend('validation', 'Location', 'northwest');
xlabel('k');
ylabel('classification rate');

%% Test
kk = 5;
load mnist_test;
kk_set = [kk-2 kk kk+2];
corr_rate = [];

for kk = kk_set
    [valid_labels] = run_knn(kk, train_inputs, train_targets, test_inputs);
%     [valid_labels] = run_knn(kk, train_inputs_small, train_targets_small, test_inputs);
    data_num = size(test_targets, 1);
    corr_num = sum(test_targets == valid_labels);
    corr_rate = [corr_rate corr_num / data_num];
end
disp(corr_rate);

%% Summary
figure();
plot(k_set, class_rate, 'b*-', kk_set, corr_rate, 'ro-');
legend('validation','test', -1);
xlabel('k');
ylabel('classification rate');
end