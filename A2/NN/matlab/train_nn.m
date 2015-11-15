%% To run this program:
%%   First run initnn.m
%%   Then repeatedly call train_nn.m until convergence.

train_CE_list = zeros(1, num_epochs);
valid_CE_list = zeros(1, num_epochs);
train_err_list = zeros(1, num_epochs);
valid_err_list = zeros(1, num_epochs);

start_epoch = total_epochs + 1;


num_train_cases = size(inputs_train, 2);
num_valid_cases = size(inputs_valid, 2);

for epoch = 1:num_epochs
  % Fprop
  h_input = W1' * inputs_train + repmat(b1, 1, num_train_cases);  % Input to hidden layer.
  h_output = 1 ./ (1 + exp(-h_input));  % Output of hidden layer.
  logit = W2' * h_output + repmat(b2, 1, num_train_cases);  % Input to output layer.
  prediction = 1 ./ (1 + exp(-logit));  % Output prediction.

  % Compute cross entropy
  train_CE = -mean(mean(target_train .* log(prediction) + (1 - target_train) .* log(1 - prediction)));

  % Compute the classificatio error
  train_err = sum(target_train ~= round(prediction)) / size(prediction, 2);
  
  % Compute deriv
  dEbydlogit = prediction - target_train;

  % Backprop
  dEbydh_output = W2 * dEbydlogit;
  dEbydh_input = dEbydh_output .* h_output .* (1 - h_output) ;

  % Gradients for weights and biases.
  dEbydW2 = h_output * dEbydlogit';
  dEbydb2 = sum(dEbydlogit, 2);
  dEbydW1 = inputs_train * dEbydh_input';
  dEbydb1 = sum(dEbydh_input, 2);

  %%%%% Update the weights at the end of the epoch %%%%%%
  dW1 = momentum * dW1 - (eps / num_train_cases) * dEbydW1;
  dW2 = momentum * dW2 - (eps / num_train_cases) * dEbydW2;
  db1 = momentum * db1 - (eps / num_train_cases) * dEbydb1;
  db2 = momentum * db2 - (eps / num_train_cases) * dEbydb2;

  W1 = W1 + dW1;
  W2 = W2 + dW2;
  b1 = b1 + db1;
  b2 = b2 + db2;

  %%%%% Test network's performance on the valid patterns %%%%%
  h_input = W1' * inputs_valid + repmat(b1, 1, num_valid_cases);  % Input to hidden layer.
  h_output = 1 ./ (1 + exp(-h_input));  % Output of hidden layer.
  logit = W2' * h_output + repmat(b2, 1, num_valid_cases);  % Input to output layer.
  prediction = 1 ./ (1 + exp(-logit));  % Output prediction.
  valid_CE = -mean(mean(target_valid .* log(prediction) + (1 - target_valid) .* log(1 - prediction)));
  valid_err = sum(target_valid ~= round(prediction)) / size(prediction, 2);
  
  %%%%%% Print out summary statistics at the end of the epoch %%%%%
  total_epochs = total_epochs + 1;
  if total_epochs == 1
      start_error = train_CE;
  end
  train_CE_list(1, epoch) = train_CE;
  valid_CE_list(1, epoch) = valid_CE;
  train_err_list(1, epoch) = train_err;
  valid_err_list(1, epoch) = valid_err;
  fprintf(1,'%d  eps=%f, mom=%f: Train CE=%f, Valid CE=%f, Train Error=%f, Valid Error=%f\n',...
            total_epochs, eps, momentum, train_CE, valid_CE, train_err, valid_err);
end

clf; 
if total_epochs > min_epochs_per_plot
  epochs = [1 : total_epochs];
end

% Note: the variable called 'train_errors' is actually confusing, since it
% is using the cross entropy. But I didn't change it, just in case, to
% avoid conflit in the future. To specify, I will use c_err instead. -Zhen
%%%%%%%%% Plot the learning curve for the training set patterns %%%%%%%%%
train_errors(1, start_epoch : total_epochs) = train_CE_list;
valid_errors(1, start_epoch : total_epochs) = valid_CE_list;
train_c_err(1, start_epoch : total_epochs) = train_err_list;
valid_c_err(1, start_epoch : total_epochs) = valid_err_list;
  hold on, ...
  plot(epochs(1, 1 : total_epochs), train_errors(1, 1 : total_epochs), 'b'),...
  plot(epochs(1, 1 : total_epochs), valid_errors(1, 1 : total_epochs), 'g'),...
  plot(epochs(1, 1 : total_epochs), train_c_err(1, 1 : total_epochs), 'k'),...
  plot(epochs(1, 1 : total_epochs), valid_c_err(1, 1 : total_epochs), 'r'),...
  legend('Train CE', 'Valid CE', 'Train Error', 'Valid Error'),...
  title(sprintf('Cross Entropy & Classification Error - eps=%.2f, momentum=%.2f, number of hidden units:%d', eps, momentum, num_hiddens)), ...
  xlabel('Epoch'), ...
  ylabel('Cross Entropy & Classification Error');
