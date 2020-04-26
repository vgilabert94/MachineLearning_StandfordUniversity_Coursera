function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;
C_list = [0.001 0.003 0.007 0.01 0.03 0.07 0.1 0.3 0.7 1 3 7 10 30 70];
sigma_list = [0.001 0.003 0.007 0.01 0.03 0.07 0.1 0.3 0.7 1 3 7 10 30 70];
%C_list = [0.001 0.003 1];
%sigma_list = [0.001 0.003 1];
results = zeros(length(C_list) * length(sigma_list), 3);
row = 1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

for c=1:length(C_list)
    C = C_list(c);
    
    for s=1:length(sigma_list)
        sigma = sigma_list(s);
        
        model = svmTrain(X, y, C, @(x1, x2)gaussianKernel(x1, x2, sigma));
        pred = svmPredict(model,Xval);
        pred_error = mean(double(pred ~= yval));
        
        results(row,:) = [C sigma pred_error];
        row = row + 1;
    end
end

[opt,i_opt] = min(results(:,3));
sigma = results(i_opt,2);
C = results(i_opt,1);

% =========================================================================

end
