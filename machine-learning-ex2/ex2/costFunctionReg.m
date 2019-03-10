function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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
hyp= X * theta;
sigmoid_params = sigmoid(hyp);
for i=1:m
  temp = (-y(i)*log(sigmoid_params(i))-(1-y(i))*log(1-sigmoid_params(i)));
  J = J +temp;
end
for j=2:n
temp_ = theta(j,1)^2;
theta_lambda = theta_lambda + temp_;
end
reg_factor = (lambda/(2*m))* theta_lambda;
J= (J /m) + reg_factor;
for i =1:n
   if i<2
     temp =sum(((sigmoid_params-y).*X(:,i)));
     grad(i)= (temp/m);
   else
    temp =sum(((sigmoid_params-y).*X(:,i)));
    grad(i)= (temp/m)+((lambda/m)* theta(i,1));
   end
end

% =============================================================

end
