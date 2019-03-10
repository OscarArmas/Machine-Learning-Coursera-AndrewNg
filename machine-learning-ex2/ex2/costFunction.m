function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
hyp= X * theta;
sigmoid_params = sigmoid(hyp);
for i=1:m
  temp = (-y(i)*log(sigmoid_params(i))-(1-y(i))*log(1-sigmoid_params(i)));
  J = J +temp;

end
J= J /m;
% =============================================================
    temp0 =sum(((sigmoid_params-y).*X(:,1)));
    temp1 =sum(((sigmoid_params-y).*X(:,2)));
    temp2 =sum(((sigmoid_params-y).*X(:,3)));
        grad(1) = (temp0/m);
        grad(2)= (temp1/m);
        grad(3)= (temp2/m);
end
