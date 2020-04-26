% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

X_original = X;

% figure
% plot(X(:,1),y,'rx'); 
% title('size houses / price')
% xlabel('Size')
% ylabel('Price')
% legend('Training data','Location','southeast')
% 
% figure
% plot(X(:,2),y,'rx'); 
% title('bedroms house / price')
% xlabel('bedroms')
% ylabel('Price')
% legend('Training data','Location','southeast')
% 
% figure
% plot(X(:,2),X(:,1),'rx'); 
% title('bedroms house / size')
% xlabel('bedroms')
% ylabel('size')
% legend('Training data','Location','southeast')


% Scale features and set them to zero mean
[X, mu, sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Run gradient descent
% Choose some alpha value
alpha = 0.1;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, ~] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Display gradient descent's result
fprintf('Theta computed from gradient descent:\n%f,\n%f',theta(1),theta(2))


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
newdata= [1650 3];
price = [1,(newdata(1)-mu(1))/sigma(1),(newdata(2)-mu(2))/sigma(2)]*theta; % You should change this
% ============================================================
fprintf('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f', round(price));

figure
plot(X(:,2),y,'rx'); 
title('size houses / price')
xlabel('Size')
ylabel('Price')
legend('Training data','Location','southeast')
hold on
plot(X(:,2),X*theta,'-')
legend('Training data', 'Prediction','Location','southeast')
hold off
