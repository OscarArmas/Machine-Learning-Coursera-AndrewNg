function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
Test_values = [0.01,0.03,0.1,0.3,1,3,10,30];
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
a = 1;
predictions =zeros(1,length(Test_values)^2);
sigmas =zeros(1,length(Test_values)^2);
Cs=zeros(1,length(Test_values)^2);
for i = 1:length(Test_values)
  for j =1:length(Test_values)
    C = Test_values(i);
    sigma = Test_values(j);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    result = svmPredict(model, Xval);
    predictions(a) =mean(double(result ~= yval));
    sigmas(a)=sigma;
    Cs(a) = C;
    ++a;
  endfor
endfor
index = find( predictions == min(predictions));
C = Cs(index);
sigma = sigmas(index);







% =========================================================================

end
