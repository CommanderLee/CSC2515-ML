% Learn a Naive Bayes classifier on the digit dataset, evaluate its
% performance on training and test sets, then visualize the mean and variance
% for each class.

load mnist_train;
load mnist_test;

% Add your code here (it should be less than 10 lines)
[log_prior, class_mean, class_var] = train_nb(train_inputs, train_targets);

[prediction_train, accuracy_train] = test_nb(train_inputs, train_targets, log_prior, class_mean, class_var);
[prediction_test, accuracy_test] = test_nb(test_inputs, test_targets, log_prior, class_mean, class_var);
fprintf('accuracy: train(%f), test(%f)\n', accuracy_train, accuracy_test);

plot_digits([class_mean;class_var]);

load mnist_train_small;
[log_prior_s, class_mean_s, class_var_s] = train_nb(train_inputs_small, train_targets_small);

[prediction_train_s, accuracy_train_s] = test_nb(train_inputs_small, train_targets_small, log_prior_s, class_mean_s, class_var_s);
[prediction_test_s, accuracy_test_s] = test_nb(test_inputs, test_targets, log_prior_s, class_mean_s, class_var_s);
fprintf('accuracy-small: train(%f), test(%f)\n', accuracy_train_s, accuracy_test_s);

plot_digits([class_mean_s;class_var_s]);
