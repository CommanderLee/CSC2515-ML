function calc_knn()
%% Calculate the results required by A2-2.5

%% Initialize
init_nn();
k_set = [1,3,5,7,9];
error_rate = [];

%% Validate
for k = k_set
    [valid_labels] = run_knn(k, inputs_train', target_train', inputs_valid');
    data_num = size(target_valid', 1);
    corr_num = sum(target_valid' == valid_labels);
    error_rate = [error_rate 1 - corr_num/data_num];
end
disp(error_rate);
plot(k_set, error_rate, 'b*-');
legend('validation', 'Location', 'northwest');
xlabel('k=1,3,5,7,9');
ylabel('classification error');

%% Test
kk = 3;
kk_set = [kk-2 kk kk+2];
error_rate = [];

for kk = kk_set
    [test_labels] = run_knn(kk, inputs_train', target_train', inputs_test');
    data_num = size(target_test', 1);
    corr_num = sum(target_test' == test_labels);
    error_rate = [error_rate 1 - corr_num / data_num];
end
disp(error_rate);

end