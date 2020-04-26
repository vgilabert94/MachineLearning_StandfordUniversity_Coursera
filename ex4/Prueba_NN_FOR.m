clear all

load('ex4data1.mat');
m = size(X, 1);
% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

% Weight regularization parameter (we set this to 0 here).
lambda = 0;

%Obtain Cost
J = nnCostFunction_using_for(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
fprintf('\nCost at parameters (loaded from ex4weights): %f\n', J);


% Weight regularization parameter (we set this to 1 here).
lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
fprintf('\nCost at parameters (loaded from ex4weights) REGULARIZED: %f\n', J);

% Inital theta's ramdomly
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
lambda = 0;
checkNNGradientsFOR;




