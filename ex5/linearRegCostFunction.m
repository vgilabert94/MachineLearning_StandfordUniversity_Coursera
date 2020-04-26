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
%% COST FUNCTION
H_theta = X * theta;            % (12x1)(2,1) -> (12,1)
predict = (H_theta - y).^2;     % (12x1)
cost = (1/(2*m))*sum(predict);  % (1x1)
reg = (lambda/(2*m))*sum(theta(2:end).^2);  % (1x1)
J = cost + reg;     %(1x1)

%% GRADIENT
grad = (1/m)*sum(((H_theta - y).*X),1);     %(1x2)
grad(:,2:end) = grad(:,2:end) + ((lambda/m)*theta(2:end)');

% =========================================================================

grad = grad(:); % 2x1

end
