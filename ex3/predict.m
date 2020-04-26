function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Adding 1's in the firts colum
X = [ones(1,m)' X];
%Obtain z2 = theta1 * a1
z2 = X * Theta1';
% Calculate activation -> a2
a2 = sigmoid(z2);
% Adding 1's in the firts colum
a2 = [ones(1,size(a2,1))' a2];
%Obtain z3 = theta2 * a2
z3 = a2 * Theta2';
% Calculate activation -> a3 = h_theta(x)
a3 = sigmoid(z3);

%Obtain the predition. We get the max value for each row (p) and the index
%of this max value (k).
[k,p]=max(a3,[],2);

% =========================================================================

end
