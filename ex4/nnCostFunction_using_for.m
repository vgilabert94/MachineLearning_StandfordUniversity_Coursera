function [J,grad] = nnCostFunction_using_for(nn_params, ...
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

%% FEEDFORWARD PASS & BACKPROPAGATION ALGORITHM

X = [ones(1,m)' X];

for t=1:m
    a1 = X(t,:);            % (1x401)
    z2 = a1 * Theta1';       % (1x401)(401x25) -> (1x25)
    a2 = sigmoid(z2);        % (1x25)
    a2 = [1 a2];             % (1x26)
    z3 = a2 * Theta2';       % (1x26)(26x10) -> (1x10)
    a3 = sigmoid(z3);        % (1x10)
   
    H_theta(t,:) = a3;
   
    s3 = a3 - y_matrix(t,:);        %(1x10)
    grad_z2 = sigmoidGradient(z2);  %(1x10)
    s2 =  (s3 * Theta2(:,2:end)) .* grad_z2;    % (1x10)(10x25) .* %(1x25)
    
    Delta2 = s2' * a1;               %(25x1)(1x401) -> (25x401)
    Delta3 = s3' * a2;               %(10x1)(1*26) -> 10x26
                                        
    Theta1_grad = Theta1_grad + (Delta2 * (1/m));
    Theta2_grad = Theta2_grad + (Delta3 * (1/m));

end

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end));
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); 

%% COST FUNCTION REGULARIZED
predict0 = -y_matrix .* log(H_theta);
predict1 = (1-y_matrix) .* log(1-H_theta);
Reg = ((lambda /(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))));
J = (1/m) * sum(sum(predict0 - predict1));
J = J + Reg;

% =========================================================================
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
