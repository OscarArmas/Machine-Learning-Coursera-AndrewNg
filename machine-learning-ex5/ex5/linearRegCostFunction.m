function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
reg_factor = lambda/(2*m)*sum(theta(2:end,:).^2);
predict = X *theta;
optim =1/m;
sqrError = (predict - y).^2;
 J= 1/(2*m) * sum(sqrError) + reg_factor;


 temp0 =optim* sum((X * theta-y));
         grad(1) = temp0;
 for iter = 2:size(theta,1)
     temp1 =(optim* sum((X * theta-y).*X(:,iter))) +((lambda/m)*theta(iter,:));
      grad(iter)= temp1;
  end
       

% =========================================================================

grad = grad(:);

end
