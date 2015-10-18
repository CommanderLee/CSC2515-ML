function [ce, frac_correct] = evaluate(targets, y)
%    Compute evaluation metrics.
%    Inputs:
%        targets : N x 1 vector of binary targets. Values should be either 0 or 1.
%        y       : N x 1 vector of probabilities.
%    Outputs:
%        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we
%                       want to compute CE(targets, y).
%        frac_correct : (scalar) Fraction of inputs classified correctly.

% TODO: Finish this function
% z = w^T * x + w_0
% y(x) = sigma(z)  => e^{-z} = 1/y - 1
exp_Z = 1.0 ./ y - 1;
% ce = E(w,b) = \sum{ [(y^(n)-1]*ln(exp_Z) + ln(1 + exp_Z) }
ce = sum((targets - 1) .* log(exp_Z) + log(1 + exp_Z));

y_bool = y > 0.5;
num_correct = sum(y_bool == targets);
frac_correct = num_correct / size(y,1);
end
