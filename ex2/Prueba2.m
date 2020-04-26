clear all
%  The first two columns contains the X values and the third column
%  contains the label (y).
data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

% Add Polynomial Features
% Note that mapFeature also adds a column of ones for us, so the intercept term is handled
X = mapFeature(X(:,1), X(:,2));
% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic regression
[cost2, grad2] = costFunction(initial_theta, X, y);
fprintf('COST FUNCTION: Cost at initial theta (zeros): %f\n', cost2);
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
fprintf('COST FUNCTION REG: Cost at initial theta (zeros): %f\n', cost);

% Plot Boundary
plotDecisionBoundary(grad, X, y);
% Add some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')
% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;