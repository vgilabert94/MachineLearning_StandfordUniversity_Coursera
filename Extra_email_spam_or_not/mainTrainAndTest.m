%% SPAM OR NOT. Train and test model.

%Load full process dataset and test dataset.
load('fulldataset.mat');
load('spamTest.mat');

C = 0.01;
sigma = 2;
%[C, sigma] = dataset3Params(X, Y, Xtest, ytest);
model = svmTrain(X, Y, C, @(x1, x2)gaussianKernel(x1, x2, sigma));
%model = svmTrain(X, Y, C, @linearKernel);

p = svmPredict(model, X);
fprintf('Training Accuracy: %f\n', mean(double(p == Y)) * 100);

% Load the test dataset
p = svmPredict(model, Xtest);
fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);

save Trained_model_spam_or_not model

% load('fulldataset.mat');
% opts = struct('Holdout',0.3);
% linSVMmdl = fitclinear(X,Y,'Learner','svm','Regularization','ridge','OptimizeHyperparameters',{'lambda'},'HyperparameterOptimizationOptions',opts)
% load('spamTest.mat');
% fprintf('Training Accuracy: %f | Test Accuracy: %f\n', mean(predict(linSVMmdl,X)==Y)*100,mean(predict(linSVMmdl,Xtest)==ytest)*100)
