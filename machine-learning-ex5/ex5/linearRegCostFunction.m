function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% X: every row is an example, bias padded
% X;

% no need:
% X_i = [ones(m, 1), X];
X_i = X;

% no need to add bias param to theta
% theta_i = [1; theta];
theta_i = theta;

% for all examples
H = X_i * theta_i;

% calc cost function
J_el = H - y;
% J = sum(J_el .* J_el) / (2 * m);
J = J_el' * J_el / (2 * m);

% add regulariation
% bias param should not be added for regularization
J = J + lambda * sum(theta_i(2:end) .* theta_i(2:end)) / (2 * m);


% calc grad
grad = ((H - y)' * X_i)' / m;

% add regul
grad_reg = grad + lambda * theta_i / m;
grad_reg(1) = grad(1);

% =========================================================================

grad = grad_reg(:);

end
