% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Scale features and set them to zero mean
[X, mu, sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

num_iters = 100;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[~, J_history_1] = gradientDescentMulti(X, y, theta, 1, num_iters);

[~, J_history_01] = gradientDescentMulti(X, y, theta, 0.1, num_iters);

[~, J_history_001] = gradientDescentMulti(X, y, theta, 0.01, num_iters);

[~, J_history_0001] = gradientDescentMulti(X, y, theta, 0.001, num_iters);

[~, J_history_03] = gradientDescentMulti(X, y, theta, 0.3, num_iters);

[~, J_history_003] = gradientDescentMulti(X, y, theta, 0.03, num_iters);

[~, J_history_05] = gradientDescentMulti(X, y, theta, 0.5, num_iters);

[~, J_history_005] = gradientDescentMulti(X, y, theta, 0.05, num_iters);

figure
plot(1:num_iters, J_history_1, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

hold on
plot(1:num_iters, J_history_01, 'LineWidth', 2);
plot(1:num_iters, J_history_001, 'LineWidth', 2);
plot(1:num_iters, J_history_0001, 'LineWidth', 2);
plot(1:num_iters, J_history_03, 'LineWidth', 2);
plot(1:num_iters, J_history_003, 'LineWidth', 2);
plot(1:num_iters, J_history_05, 'LineWidth', 2);
plot(1:num_iters, J_history_005, 'LineWidth', 2);
legend('alpha = 1', 'alpha = 0.1','alpha = 0.01','alpha = 0.001','alpha = 0.3','alpha = 0.03','alpha = 0.5','alpha = 0.05','Location','northeast')
hold off
