function [J,grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================

% Obtain de output in matrix format.
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

% Alternative code:
% y_matrix=zeros(m,num_labels);
% for i=1:length(y)
%     y_matrix(i,y(i)) = 1;
% end

%% FEEDFORWARD PASS (no "for loop")
% Adding 1's in the firts colum
a1 = [ones(1,m)' X];        % (5000x401)
%Obtain z2 = theta1 * a1
z2 = a1 * Theta1';          % (5000x401)(401x25) -> (5000x25)
% Calculate activation -> a2
a2 = sigmoid(z2);           % (5000x25)
% Adding 1's in the firts colum
a2 = [ones(1,size(a2,1))' a2];      % (5000x26)
%Obtain z3 = theta2 * a2
z3 = a2 * Theta2';          % (5000x26)(26x10)-> (5000x10)
% Calculate activation -> a3 = h_theta(x)
a3 = sigmoid(z3);           % (5000x10)
H_theta = a3;


%% COST FUNCTION NON REGULARIZED
% predict0 = -y_matrix .* log(H_theta);
% predict1 = (1-y_matrix) .* log(1-H_theta);
% J = (1/m) * sum(sum(predict0 - predict1));

%% COST FUNCTION REGULARIZED
predict0 = -y_matrix .* log(H_theta);
predict1 = (1-y_matrix) .* log(1-H_theta);
Reg = ((lambda /(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))));
J = (1/m) * sum(sum(predict0 - predict1));
J = J + Reg;

%% BACKPROPAGATION ALGORITHM

s3 = a3 - y_matrix;             % (5000x10)

grad_z2 = sigmoidGradient(z2);  % (5000x25)

s2 =  (s3 * Theta2(:,2:end)) .* grad_z2;    % (5000x10)(10x25) .* (5000x25)

Delta1 = s2' * a1;               %(25x5000)(5000x401) -> 25x401
Delta2 = s3' * a2;               %(10x5000)(5000*26) -> 10x26

Theta1_grad = Delta1 * (1/m);    % (25*401)
Theta2_grad = Delta2 * (1/m);    % (10*26)

%% Regularization for j > 0 // I just calculate the regularization for 2:end values. (1 -> bias)
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (Theta1(:,2:end) * (lambda/m));    % (25x401) + ((25x401)*cte) -> (25x401) 
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (Theta2(:,2:end) * (lambda/m));    % (10x26) + ((10x26)*cte) -> (10x26) 

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
