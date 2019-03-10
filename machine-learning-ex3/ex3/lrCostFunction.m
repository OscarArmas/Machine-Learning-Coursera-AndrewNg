function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
theta_lambda = 0;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 

%====Cost fucntion====
hyp= X * theta;
sigmoid_params = sigmoid(hyp);
temp = sum(-y.*log(sigmoid_params)-(1.-y).*log(1.-sigmoid_params));
J = J +temp;
%====End Cost fucntion====

%====Factor regularized cost function====
theta_not_one = theta(2:end);
temp_ = theta_not_one.^2;
theta_lambda =sum(temp_);

reg_factor = (lambda/(2*m))* theta_lambda;
%====End Factor regularized cost function====
J= (J /m) + reg_factor;
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
hypdiffY = sigmoid_params-y;
temp =(X')*(hypdiffY);

factor_regularized = (lambda/m) .* theta;
factor_regularized(1,1) = 0
temp = (1/m).*(temp);
grad= temp.+factor_regularized;

% =============================================================

grad = grad(:);

end
